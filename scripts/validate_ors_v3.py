#!/usr/bin/env python3
"""Validates correctness and speed of the ORS v3 implementation.

This script runs five checks:
1. numerical table validation for DP-prefix error tables
2. stage-1 candidate cost consistency
3. stage-1 microbenchmarks for legacy vs v3
4. notebook-style end-to-end ORS timing on a real session bundle
5. MT4XAI pipeline-style teaching-pool smoke run under /tmp

The script does not modify repository-tracked files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch


def _repo_root() -> Path:
    """Return repository root from this script location."""
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4xai import ors as legacy_ors  # noqa: E402
from mt4xai import ors_v3  # noqa: E402
from mt4xai.data import apply_scalers, build_loader, fit_scalers_on_train, split_data  # noqa: E402
from mt4xai.inference import make_bundle_from_session_df  # noqa: E402
from mt4xai.model import load_lstm_model  # noqa: E402
from mt4xai.teach import TeachingPool, TeachingPoolConfig  # noqa: E402
from project_config import load_config  # noqa: E402


def _print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _assert_all_finite(*arrays: np.ndarray) -> None:
    """Assert arrays contain only finite values."""
    for arr in arrays:
        if not np.isfinite(arr).all():
            raise AssertionError("found NaN or Inf values in error tables")


def _table_validation() -> Dict[str, float]:
    """Compare v3 DP-prefix tables against direct-sum references."""
    _print_section("1) Correctness: Numerical table validation")
    rng = np.random.default_rng(42)
    max_diffs = {"err": 0.0, "errL": 0.0, "errR": 0.0, "errA": 0.0}
    lengths = [16, 32, 48, 64, 80, 96]

    for n in lengths:
        y = rng.normal(size=n).astype(float)
        err_ref, errL_ref, errR_ref, errA_ref = legacy_ors.build_error_tables_no_prefix(y)
        err_v3, errL_v3, errR_v3, errA_v3 = ors_v3.build_error_tables_with_prefix(y)

        _assert_all_finite(err_v3, errL_v3, errR_v3, errA_v3)
        max_diffs["err"] = max(max_diffs["err"], float(np.max(np.abs(err_ref - err_v3))))
        max_diffs["errL"] = max(max_diffs["errL"], float(np.max(np.abs(errL_ref - errL_v3))))
        max_diffs["errR"] = max(max_diffs["errR"], float(np.max(np.abs(errR_ref - errR_v3))))
        max_diffs["errA"] = max(max_diffs["errA"], float(np.max(np.abs(errA_ref - errA_v3))))

    print("max abs diffs:", {k: f"{v:.3e}" for k, v in max_diffs.items()})
    tol = 1e-8
    if any(v > tol for v in max_diffs.values()):
        raise AssertionError(f"table validation failed with tolerance {tol}: {max_diffs}")
    print(f"[PASS] table differences are within tolerance {tol}")
    return max_diffs


def _candidate_consistency(alpha: float = 0.01, beta: float = 3.0, q: int = 120) -> Dict[str, float]:
    """Check stage-1 candidate cost consistency and legacy mismatch reproduction."""
    _print_section("2) Correctness: Candidate-cost consistency")
    rng = np.random.default_rng(123)
    y = rng.normal(size=90).astype(float)

    cands_v3 = ors_v3.stage1_dp_prefix(y, q=q, alpha=alpha, beta=beta)
    max_abs_gap_v3 = 0.0
    for cost_es, piv, piv_vals in cands_v3:
        vals = np.asarray(piv_vals, dtype=float) if piv_vals is not None else y[piv]
        sts = ors_v3.interpolate_from_pivots(
            T=y.size,
            pivots=piv,
            values=vals,
            t_min_eval=0,
            anchor_endpoints="both",
        )
        k = int(len(piv) - 1)
        lhs = float(cost_es)
        rhs = float(alpha * np.sum((y - sts) ** 2) + beta * k)
        max_abs_gap_v3 = max(max_abs_gap_v3, abs(lhs - rhs))

    print(f"v3 max |cost_es - (alpha*l2 + beta*k)| = {max_abs_gap_v3:.3e}")
    if max_abs_gap_v3 > 1e-8:
        raise AssertionError(f"v3 candidate consistency failed: gap={max_abs_gap_v3}")

    cands_legacy = legacy_ors.stage1_dp_prefix(y, q=q, alpha=alpha, beta=beta)
    legacy_mismatch = 0.0
    legacy_checked = 0
    for cost_es, piv in cands_legacy:
        if len(piv) == 3 and int(piv[0]) == 0 and int(piv[-1]) == y.size - 1:
            sts = legacy_ors.interpolate_from_pivots(
                T=y.size,
                pivots=piv,
                values=y[piv],
                t_min_eval=0,
                anchor_endpoints="both",
            )
            k = int(len(piv) - 1)
            rhs = float(alpha * np.sum((y - sts) ** 2) + beta * k)
            legacy_mismatch = max(legacy_mismatch, abs(float(cost_es) - rhs))
            legacy_checked += 1

    print(f"legacy checked candidates with piv=[0,*,T-1]: {legacy_checked}")
    print(f"legacy max mismatch on those candidates: {legacy_mismatch:.3e}")
    if legacy_checked == 0:
        print("[WARN] no legacy [0,*,T-1] candidates found in this sample")
    elif legacy_mismatch <= 1e-8:
        raise AssertionError("legacy mismatch was not reproduced as expected")
    print("[PASS] v3 consistency verified and legacy mismatch reproduced")

    return {
        "v3_max_abs_gap": max_abs_gap_v3,
        "legacy_max_abs_gap": legacy_mismatch,
        "legacy_checked": float(legacy_checked),
    }


def _median_time(fn, reps: int = 3) -> float:
    """Return median runtime in seconds over repeated calls."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(times, dtype=float)))


def _microbench(alpha: float = 0.01, beta: float = 3.0, q: int = 150, reps: int = 3) -> pd.DataFrame:
    """Benchmark legacy vs v3 stage-1 routines."""
    _print_section("3) Speed proof: Stage-1 microbenchmarks")
    rng = np.random.default_rng(7)
    rows = []
    for n in [60, 90, 120, 150, 200, 260]:
        y = rng.normal(size=n).astype(float)

        t_old_tbl = _median_time(lambda: legacy_ors.build_error_tables_with_prefix_legacy(y), reps=reps)
        t_v3_tbl = _median_time(lambda: ors_v3.build_error_tables_with_prefix(y), reps=reps)

        t_old_stage1 = _median_time(lambda: legacy_ors.stage1_dp_prefix(y, q=q, alpha=alpha, beta=beta), reps=reps)
        t_v3_stage1 = _median_time(lambda: ors_v3.stage1_dp_prefix(y, q=q, alpha=alpha, beta=beta), reps=reps)

        row = {
            "T": n,
            "table_old_s": t_old_tbl,
            "table_v3_s": t_v3_tbl,
            "table_speedup_x": t_old_tbl / max(t_v3_tbl, 1e-12),
            "stage1_old_s": t_old_stage1,
            "stage1_v3_s": t_v3_stage1,
            "stage1_speedup_x": t_old_stage1 / max(t_v3_stage1, 1e-12),
        }
        rows.append(row)
        print(
            f"T={n:3d} | table {row['table_old_s']:.4f}s -> {row['table_v3_s']:.4f}s "
            f"({row['table_speedup_x']:.2f}x) | stage1 {row['stage1_old_s']:.4f}s -> "
            f"{row['stage1_v3_s']:.4f}s ({row['stage1_speedup_x']:.2f}x)"
        )

    df = pd.DataFrame(rows)
    if not (df["table_speedup_x"] > 1.0).all():
        raise AssertionError("v3 table build is not consistently faster than legacy")
    if not (df["stage1_speedup_x"] > 1.0).all():
        raise AssertionError("v3 stage1 candidate generation is not consistently faster than legacy")
    print("[PASS] v3 is faster than legacy in all benchmark sizes")
    return df


def _load_notebook_style_context(device: torch.device):
    """Load model, data, scalers and one bundle using notebook-style steps."""
    cfg = load_config()
    model_fp = ROOT / cfg.paths.final_model
    data_fp = ROOT / cfg.paths.dataset

    model, ckpt = load_lstm_model(str(model_fp), device=device)
    model.eval()
    input_features = ckpt["input_features"]
    target_features = ckpt["target_features"]
    horizon = int(ckpt["config"]["horizon"])

    idx_power_inp = input_features.index("power")
    idx_soc_inp = input_features.index("soc")

    df_cleaned = pd.read_parquet(data_fp)
    drop_cols = [
        "energy",
        "charger_category",
        "timestamp",
        "nearest_weather_station",
        "timestamp_d",
        "lat",
        "lon",
        "timestamp_H",
    ]
    drop_cols = [c for c in drop_cols if c in df_cleaned.columns]
    df = df_cleaned.drop(columns=drop_cols).copy()

    train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1, random_seed=int(cfg.project.random_seed))
    cols_to_scale = list(set(input_features) | set(target_features))
    scalers = fit_scalers_on_train(train_df, cols_to_scale)
    power_scaler = scalers["power"]
    soc_scaler = scalers["soc"]
    test_s = apply_scalers(test_df, scalers)

    # notebook session id used in 05__Curve_Simplification
    sid = 5649864
    bundle = make_bundle_from_session_df(
        model=model,
        df_scaled=test_s,
        sid=int(sid),
        device=device,
        input_features=input_features,
        target_features=target_features,
        horizon=horizon,
        power_scaler=power_scaler,
        soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp,
        idx_soc_inp=idx_soc_inp,
    )

    return {
        "cfg": cfg,
        "model": model,
        "input_features": input_features,
        "target_features": target_features,
        "horizon": horizon,
        "idx_power_inp": idx_power_inp,
        "power_scaler": power_scaler,
        "val_df": val_df,
        "test_s": test_s,
        "bundle": bundle,
    }


def _end_to_end_notebook_style(ctx: Dict, reps: int = 2) -> Dict[str, float]:
    """Compare end-to-end ORS call timing using notebook-style invocation."""
    _print_section("4) Speed proof: Notebook-style end-to-end ORS timing")
    cfg = ctx["cfg"]
    model = ctx["model"]
    bundle = ctx["bundle"]
    power_scaler = ctx["power_scaler"]
    idx_power_inp = ctx["idx_power_inp"]

    params_legacy = legacy_ors.ORSParams(
        stage1_mode="dp_prefix",
        stage2_err_metric="l2",
        dp_q=220,
        rdp_stage1_candidates=40,
        dp_alpha=0.001,
        beta=3.0,
        gamma=0.05,
        R=600,
        epsilon_mode="fraction",
        epsilon_value=0.3,
        t_min_eval=int(cfg.inference.t_min_eval),
        anchor_endpoints="last",
        min_k=1,
        max_k=15,
        random_seed=int(cfg.project.random_seed),
        soc_stage1_mode="rdp",
        soc_rdp_epsilon=0.75,
    )
    params_v3 = replace(params_legacy, stage1_mode="dp_prefix_v3")

    def run_legacy():
        return legacy_ors.ors(
            bundle,
            model,
            params_legacy,
            power_scaler=power_scaler,
            idx_power_inp=idx_power_inp,
            decay_lambda=float(cfg.inference.horizon_decay_lambda),
            threshold=float(cfg.inference.ad_rmse_threshold),
        )

    def run_v3():
        return ors_v3.ors(
            bundle,
            model,
            params_v3,
            power_scaler=power_scaler,
            idx_power_inp=idx_power_inp,
            decay_lambda=float(cfg.inference.horizon_decay_lambda),
            threshold=float(cfg.inference.ad_rmse_threshold),
        )

    legacy_time = _median_time(run_legacy, reps=reps)
    v3_time = _median_time(run_v3, reps=reps)
    res_legacy = run_legacy()
    res_v3 = run_v3()

    if res_legacy is None or res_v3 is None:
        raise AssertionError("end-to-end ORS returned None for legacy or v3")

    legacy_keys = set(res_legacy.keys())
    v3_keys = set(res_v3.keys())
    if legacy_keys != v3_keys:
        raise AssertionError(f"result schema mismatch:\nlegacy={legacy_keys}\nv3={v3_keys}")

    print(f"legacy ORS time: {legacy_time:.4f}s")
    print(f"v3 ORS time:     {v3_time:.4f}s")
    print(f"speedup:         {legacy_time / max(v3_time, 1e-12):.2f}x")
    print(
        "legacy summary:",
        {"k": int(res_legacy["k"]), "obj": float(res_legacy["obj"]), "label": str(res_legacy["label"])},
    )
    print(
        "v3 summary:",
        {"k": int(res_v3["k"]), "obj": float(res_v3["obj"]), "label": str(res_v3["label"])},
    )

    if not (legacy_time > v3_time):
        raise AssertionError("v3 did not beat legacy in notebook-style ORS timing")

    return {
        "legacy_time_s": legacy_time,
        "v3_time_s": v3_time,
        "speedup_x": legacy_time / max(v3_time, 1e-12),
    }


def _pipeline_smoke(ctx: Dict, out_dir: Path) -> Dict[str, float]:
    """Run a small TeachingPool construction using v3 ORS path."""
    _print_section("5) Pipeline-style compatibility smoke test")
    cfg = ctx["cfg"]
    model = ctx["model"]
    input_features = ctx["input_features"]
    target_features = ctx["target_features"]
    horizon = ctx["horizon"]
    idx_power_inp = ctx["idx_power_inp"]
    power_scaler = ctx["power_scaler"]

    val_df = ctx["val_df"]
    cols_to_scale = list(set(input_features) | set(target_features))
    train_df, _, _ = split_data(val_df, test_size=0.2, validation_size=0.1, random_seed=int(cfg.project.random_seed))
    scalers = fit_scalers_on_train(train_df, cols_to_scale)
    val_s = apply_scalers(val_df, scalers)

    # small subset for smoke execution
    keep_ids = val_s["charging_id"].drop_duplicates().head(80).to_numpy()
    subset = val_s[val_s["charging_id"].isin(keep_ids)].copy()
    loader = build_loader(
        subset,
        input_features,
        target_features,
        horizon,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # monkey patch teach module symbols to use v3 API-compatible entrypoint
    import mt4xai.teach as teach_mod

    old_ors_fn = teach_mod.ors
    old_ors_params_cls = teach_mod.ORSParams
    teach_mod.ors = ors_v3.ors
    teach_mod.ORSParams = ors_v3.ORSParams

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ors_params = ors_v3.ORSParams(
            stage1_mode="dp_prefix_v3",
            stage2_err_metric="l2",
            dp_q=120,
            rdp_stage1_candidates=25,
            dp_alpha=0.0075,
            beta=4.0,
            gamma=0.05,
            R=160,
            epsilon_mode="fraction",
            epsilon_value=0.2,
            t_min_eval=2,
            anchor_endpoints="last",
            min_k=1,
            max_k=12,
            random_seed=int(cfg.project.random_seed),
            soc_stage1_mode="rdp",
            soc_rdp_epsilon=0.75,
            model_id=str(ROOT / cfg.paths.final_model),
        )
        tp_cfg = TeachingPoolConfig(
            model_path=str(ROOT / cfg.paths.final_model),
            output_dir=str(out_dir),
            ad_threshold=float(cfg.inference.ad_rmse_threshold),
            random_seed=int(cfg.project.random_seed),
            length_range=(11, 60),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            export_every=5,
            L=128,
            P=4,
            decay_lambda=float(cfg.inference.horizon_decay_lambda),
            ors_params=ors_params,
        )

        t0 = time.perf_counter()
        pool = TeachingPool.construct_from_cfg(
            model=model,
            config=tp_cfg,
            loader=loader,
            df_scaled=subset,
            power_scaler=power_scaler,
            soc_scaler=scalers["soc"],
            idx_power_inp=idx_power_inp,
            idx_soc_inp=input_features.index("soc"),
        )
        elapsed = time.perf_counter() - t0
    finally:
        teach_mod.ors = old_ors_fn
        teach_mod.ORSParams = old_ors_params_cls

    if pool.pool_df is None or pool.pool_df.empty:
        raise AssertionError("pipeline smoke run produced an empty pool")

    required_cols = {"session_id", "k", "label_int", "label_text"}
    missing = required_cols.difference(set(pool.pool_df.columns))
    if missing:
        raise AssertionError(f"pool output misses expected columns: {sorted(missing)}")

    print(f"pool rows: {len(pool.pool_df)}")
    print(f"smoke runtime: {elapsed:.2f}s")
    print(f"output dir: {out_dir}")
    print("[PASS] pipeline-style smoke run completed with expected ORS fields")
    return {"pool_rows": float(len(pool.pool_df)), "elapsed_s": elapsed}


def main() -> None:
    """Run all validations and print a concise summary."""
    parser = argparse.ArgumentParser(description="Validate ORS v3 correctness and speed.")
    parser.add_argument("--device", default=None, help="torch device, default chooses cuda if available else cpu")
    parser.add_argument("--reps", type=int, default=2, help="repetitions for runtime medians")
    parser.add_argument("--out-dir", default=None, help="optional output directory for pipeline smoke test")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tmp_base = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="ors_v3_validation_"))
    smoke_dir = tmp_base / "teaching_pool_smoke_v3"
    print(f"[info] device={device}")
    print(f"[info] temporary output root={tmp_base}")

    summary: Dict[str, object] = {}
    summary["table_validation"] = _table_validation()
    summary["candidate_consistency"] = _candidate_consistency()
    summary["microbench"] = _microbench(reps=max(1, int(args.reps)))
    ctx = _load_notebook_style_context(device=device)
    summary["end_to_end"] = _end_to_end_notebook_style(ctx, reps=max(1, int(args.reps)))
    summary["pipeline_smoke"] = _pipeline_smoke(ctx, out_dir=smoke_dir)

    _print_section("Validation summary")
    print("table validation:", summary["table_validation"])
    print("candidate consistency:", summary["candidate_consistency"])
    print("microbench rows:")
    print(summary["microbench"].to_string(index=False))
    print("end-to-end:", summary["end_to_end"])
    print("pipeline smoke:", summary["pipeline_smoke"])
    print("\n[PASS] All ORS v3 validations completed successfully.")


if __name__ == "__main__":
    main()
