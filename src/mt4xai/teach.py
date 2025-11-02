# src/mt4xai/teach.py
# machine teaching utilities: pool construction, binning, selection and analytics.
from __future__ import annotations
import time
import json
import math
import os
import copy
import heapq
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Sequence, Optional, Tuple
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths
import matplotlib.pyplot as plt
import sklearn
import torch
from IPython.display import display

# ---- mt4xai project modules ----
from .ors import ORSParams, ors, build_true_abs_from_series, macro_rmse_from_abs
from .inference import predict_residuals, inverse_targets_np, reconstruct_abs_from_residuals_batch
from .data import fetch_session_preds_bundle, ChargingSession, ChargingSessionSimplification
from .model import load_lstm_model


# ----------------- Teaching Pool ------------------------- #

@dataclass
class TeachingPool:
    """Container for the ORS teaching pool and its derived strata/bins.

    This class owns the raw pool (all candidate sessions), the stratified view
    after binning, and file-system paths for persisted artefacts.

    Attributes:
        pool_df: DataFrame with one row per session in the teaching pool.
        bins_df: Stratified DataFrame produced by `bin_pool(...)` (adds k/k_bin).
        bins_meta: Metadata from the most recent binning run (edges, counts, etc.).
        paths: Key file paths: `root`, `pool_parquet`, `cache_sqlite`, `sample_plan`.
        meta: Lightweight provenance (e.g. `model_id`, `threshold`, `random_seed`, `timestamp`).
    """

    pool_df: pd.DataFrame | None = None
    bins_df: pd.DataFrame | None = None
    bins_meta: dict | None = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    config: TeachingPoolConfig = None

    @classmethod
    def construct_from_cfg(
        cls,
        model: torch.nn.Module | None,
        *,
        config: TeachingPoolConfig,
        test_loader: torch.utils.data.DataLoader,
        power_scaler: sklearn.preprocessing.MinMaxScaler,
        soc_scaler: sklearn.preprocessing.MinMaxScaler,
        idx_power_inp: int,
        idx_soc_inp: int,
    ) -> "TeachingPool":
        """Constructs the teaching pool on disk and returns a loaded `TeachingPool`.

        This wraps `TeachingPool.build(...)`: it runs the pipeline that
        computes base labels, applies ORS, writes the SQLite cache and Parquet
        snapshot, then loads the Parquet into memory and wires up paths/metadata.

        Args:
            model: Forecasting model (LSTM). If None, a loader inside `build` loads it from `config.model_path`.
            config: TeachingPoolConfig controlling thresholds, dirs and ORS params.
            test_loader: DataLoader for the test set sessions.
            power_scaler: Fitted scaler for the power channel (inverse-transform).
            soc_scaler: Fitted scaler for the SOC channel (inverse-transform).
            idx_power_inp: Index of the power feature in the input tensor.
            idx_soc_inp: Index of the SOC feature in the input tensor.

        Returns:
            TeachingPool: a ready-to-use pool object with `pool_df` loaded and standard `paths` initialised.
        """
        # build artefacts to disk
        cls.build(
            model=model,
            config=config,
            test_loader=test_loader,
            power_scaler=power_scaler,
            soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp,
            idx_soc_inp=idx_soc_inp,
        )

        # load snapshot
        root = Path(config.output_dir)
        pool_parquet = root / "pool.parquet"
        pool_df = pd.read_parquet(pool_parquet)

        paths = {
            "root": root,
            "pool_parquet": pool_parquet,
            "cache_sqlite": root / "pool_cache.db",
            "sample_plan": root / "sampled_normals.json",
        }
        meta = {
            "model_id": Path(config.model_path).name,
            "threshold": config.threshold,
            "random_seed": config.random_seed,
            "timestamp": int(time.time()),
        }
        return cls(pool_df=pool_df, paths=paths, meta=meta, config=config)

    @classmethod
    def load_from_parquet(
        cls, 
        pool_parquet: str | Path, *, 
        root_dir: str | Path | None = None, 
        config: Optional[TeachingPoolConfig] = None
    ) -> "TeachingPool":
        """Loads a pool from an existing Parquet snapshot.
        Args: 
            config: Pass in the config to set the optional TeachingPool.config attr"""
        pq = Path(pool_parquet)
        root = Path(root_dir) if root_dir is not None else pq.parent
        pool_df = pd.read_parquet(pq)
        paths = {
            "root": root,
            "pool_parquet": pq,
            "cache_sqlite": root / "pool_cache.db",
            "sample_plan": root / "sampled_normals.json",
        }
        return cls(pool_df=pool_df, paths=paths, meta={}, config=config)

    # ---------- building (moved from free function) ----------

    @classmethod
    def build(
        cls,
        model: torch.nn.Module | None,
        config: "TeachingPoolConfig",
        test_loader: torch.utils.data.DataLoader,
        power_scaler: sklearn.preprocessing.MinMaxScaler,
        soc_scaler: sklearn.preprocessing.MinMaxScaler,
        idx_power_inp: int,
        idx_soc_inp: int,
    ) -> None:
        """Builds and caches the ORS teaching pool used in the MT4XAI pipeline.

        Writes:
            - SQLite cache (`pool_cache.db`) with metadata and embeddings.
            - Parquet snapshot (`pool.parquet`) for downstream analysis.
            - Sampling plan (`sampled_normals.json`).
            - JSON config with provenance.

        Raises:
            RuntimeError: If SOC is missing/invalid when saving raw SOC arrays.
            ValueError: If not enough normal sessions are available to match abnormals.
        """
        if model is None:
            model, ckpt = load_lstm_model(config.model_path, device=config.device)
            model.eval()
            cfg_model = ckpt["config"]
            config.power_weight = float(cfg_model.get("power_weight", config.power_weight))
            config.decay_lambda = float(cfg_model.get("decay_lambda", config.decay_lambda))
            print(f"[teach] loaded model config: power_weight={config.power_weight}, decay_lambda={config.decay_lambda}")

        dirs = cls.ensure_dirs(Path(config.output_dir))
        db_path = dirs["root"] / "pool_cache.db"
        conn = cls.db_connect(db_path)
        cls.create_main_table(conn)

        sp_json = dirs["root"] / "sampled_normals.json"
        if sp_json.exists():
            plan = json.loads(sp_json.read_text())
            abn_ids = plan.get("abnormal", [])
            norm_ids_sampled = plan.get("normal", [])
            print(f"[teach] loaded existing sampling plan: abnormal={len(abn_ids)}, normal={len(norm_ids_sampled)}")
        else:
            print("[teach] computing base labels on test set ...")
            abn_ids, norm_ids, err_by_id = cls.compute_base_labels(
                test_loader,
                model,
                config.device,
                power_scaler=power_scaler,
                soc_scaler=soc_scaler,
                idx_power_inp=idx_power_inp,
                idx_soc_inp=idx_soc_inp,
                power_weight=config.power_weight,
                decay_lambda=config.decay_lambda,
                t_min_eval=config.ors_params.t_min_eval,
                threshold=config.threshold,
            )
            print(f"[teach] base labels: abnormal={len(abn_ids)}, normal={len(norm_ids)}")
            plan = cls.sample_sessions(abn_ids, norm_ids, random_seed=config.random_seed)
            (dirs["root"] / "sampled_normals.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            print(f"[teach] wrote sampling plan → {sp_json}")

        target_ids = list(sorted(set(plan["abnormal"]) | set(plan["normal"])))
        print(f"[teach] sampling plan: abnormal={len(plan['abnormal'])}, normal={len(plan['normal'])}, total={len(target_ids)}")

        cached_ids = set(int(x[0]) for x in conn.execute("SELECT session_id FROM ors_pool").fetchall())
        to_process = [sid for sid in target_ids if sid not in cached_ids]
        if cached_ids:
            print(f"[teach] found {len(cached_ids)} already cached; skipping those.")
        else:
            print("[teach] no cached rows yet.")

        start_time = time.time()
        last_export = 0
        total = len(to_process)

        params = config.ors_params
        params.model_id = Path(config.model_path).name
        for n_done, sid in enumerate(to_process, start=1):
            # fetch bundle by session_id
            b = fetch_session_preds_bundle(
                model,
                test_loader,
                batch_index=None,
                sample_index=None,
                session_id=int(sid),
                device=config.device,
                power_scaler=power_scaler,
                soc_scaler=soc_scaler,
                idx_power_inp=idx_power_inp,
                idx_soc_inp=idx_soc_inp,
            )

            # persist raw power and raw SOC for overlays
            try:
                np.save(dirs["raw_power"] / f"{sid}.npy", np.asarray(b.true_power_unscaled, dtype=np.float32))
            except Exception as e:
                print(f"[teach][warn] cannot save raw power sid={sid}: {e}")
            try:
                soc_arr = np.asarray(b.true_soc_unscaled, dtype=np.float32).reshape(-1)
                if soc_arr.size == 0 or not np.all(np.isfinite(soc_arr)):
                    raise ValueError("missing or invalid SOC series")
                np.save(dirs["raw_soc"] / f"{sid}.npy", soc_arr)
            except Exception as e:
                raise RuntimeError(f"[teach] session {sid}: SOC missing or cannot save: {e}") from e

            # clamp ORS params to sequence length
            try:
                T_seq = int(getattr(b, "T", None) or len(b.true_power_unscaled))
            except Exception:
                T_seq = 0
            p_local = cls.prepare_ors_params_for_T(params, T_seq if T_seq > 0 else 2)

            # run ORS safely
            try:
                res = ors(
                    b,
                    model,
                    p_local,
                    power_scaler=power_scaler,
                    soc_scaler=soc_scaler,
                    idx_power_inp=idx_power_inp,
                    idx_soc_inp=idx_soc_inp,
                    power_weight=config.power_weight,
                    decay_lambda=config.decay_lambda,
                    threshold=config.threshold,
                )
            except Exception as e:
                print(f"[teach][warn] ORS raised for sid={sid}: {type(e).__name__}: {e}")
                continue

            ok, why = cls.validate_ors_result(res)
            if not ok:
                print(f"[teach][warn] ORS invalid for sid={sid} ({why}); skipping.")
                continue

            # metrics
            try:
                robust_prob = float(1.0 - float(res["frag"]))
                lbl_text = str(res["label"])
                lbl_int = 1 if lbl_text == "abnormal" else 0
                margin = float(config.threshold - res["err"]) if lbl_int == 0 else float(res["err"] - config.threshold)
            except Exception as e:
                print(f"[teach][warn] metric cast failure sid={sid}: {e}; skipping.")
                continue

            # save power arrays
            try:
                sts = np.asarray(res["sts"], dtype=np.float32)
                piv = np.asarray(res["piv"], dtype=np.int32)
                sts_path = dirs["sts_full"] / f"{sid}.npy"
                piv_path = dirs["piv"] / f"{sid}.npy"
                np.save(sts_path, sts)
                np.save(piv_path, piv)
            except Exception as e:
                print(f"[teach][warn] saving arrays failed sid={sid}: {e}; skipping.")
                continue

            # save soc arrays (optional)
            piv_soc_path: Path | None = None
            sts_soc_path: Path | None = None
            try:
                piv_soc = res.get("piv_soc", None)
                sts_soc = res.get("sts_soc", None)
                if piv_soc is not None and sts_soc is not None:
                    piv_soc = np.asarray(piv_soc, dtype=np.int32)
                    sts_soc = np.asarray(sts_soc, dtype=np.float32)
                    piv_soc_path = dirs["piv_soc"] / f"{sid}.npy"
                    sts_soc_path = dirs["sts_soc"] / f"{sid}.npy"
                    np.save(piv_soc_path, piv_soc)
                    np.save(sts_soc_path, sts_soc)
            except Exception:
                print(f"[teach][warn] saving soc simplification arrays failed for sid={sid}, proceeding with only raw SOC ")
                piv_soc_path, sts_soc_path = None, None

            # embedding
            try:
                emb = cls.compute_embedding(sts, L=config.L, P=config.P)
                emb_blob = emb.tobytes()
            except Exception as e:
                print(f"[teach][warn] embedding failed sid={sid}: {e}; skipping.")
                continue

            row = dict(
                session_id=int(sid),
                label_text=lbl_text,
                label_int=int(lbl_int),
                k=float(res["k"]),
                err=float(res["err"]),
                frag=float(res["frag"]),
                robust_prob=robust_prob,
                margin=margin,
                threshold=float(config.threshold),
                model_id=p_local.model_id,
                ts_unix=float(time.time()),
                raw_power_path=str((dirs["raw_power"] / f"{sid}.npy").as_posix()),
                raw_soc_path=str((dirs["raw_soc"] / f"{sid}.npy").as_posix()),
                sts_full_path=str(sts_path.as_posix()),
                sts_soc_path=(str(sts_soc_path.as_posix()) if sts_soc_path is not None else None),
                piv_path=str(piv_path.as_posix()),
                piv_soc_path=(str(piv_soc_path.as_posix()) if piv_soc_path is not None else None),
                emb_dim=int(2 * config.L + 2 * config.P),
                emb=emb_blob,
            )
            cls.upsert_row(conn, row)

            # progress and periodic export
            pct = (n_done / total) * 100.0 if total > 0 else 100.0
            print(f"[teach] ORS {n_done:4d}/{total} ({pct:4.1f}%) sid={sid}")
            if (n_done - last_export) >= config.export_every or n_done == total:
                print("[teach] exporting parquet snapshot ...")
                cls.rows_to_parquet(db_path, dirs["root"] / "pool.parquet")
                last_export = n_done

        cls.export_config(config, dirs["root"], n_abn=len(plan["abnormal"]), n_norm=len(plan["normal"]))
        cls.rows_to_parquet(db_path, dirs["root"] / "pool.parquet")

        dt = time.time() - start_time
        hh, rem = divmod(int(dt), 3600)
        mm, ss = divmod(rem, 60)
        print(f"[teach] done. processed {len(to_process)} sessions in {hh:d} h {mm:d} m.")
        print(f"parquet: {dirs['root'] / 'pool.parquet'}")

    # ---------- binning & budgets (inlined logic) ----------

    def bin_pool(
        self,
        *,
        label_source: Literal["base", "simplified"] = "base",
        binning: Literal["quantile", "fixed"] = "quantile",
        target_bins: int = 5,
        min_bins: int = 4,
        max_bins: int = 6,
        fixed_edges_per_class: dict[str, list[int]] | None = None,
        ensure_extrema: bool = True,
        save_outputs: bool = True,
        verbose: bool = True
    ) -> tuple[pd.DataFrame, dict]:
        """Runs binning on the pool by class and k and stores the result in `self.bins_df`.

        Args:
            binning: "quantile" (per-class qcut with robust fallbacks) or "fixed" (use fixed_edges_per_class).
            target_bins, min_bins, max_bins: knobs for quantile binning; ignored if binning="fixed".
            fixed_edges_per_class: optional dict {"0": [e0,...,eN], "1": [...]}. Only used if binning="fixed".
            ensure_extrema: if True, clamp first/last bin labels to each class's min_k/max_k (display only).
            save_outputs: if True, writes parquet + JSON meta to output_dir.
            verbose: If True prints summary statistics about the bins.
        Raises:
            ValueError: If `paths['pool_parquet']` is missing or `k` is missing.

        Returns:
            df_binned: DataFrame with at least ["session_id","class_label","k","k_bin_idx","k_bin_label"].
            meta: Metadata dict with edges per class, counts per bin, class-k histograms, etc.
        """
        if "pool_parquet" not in self.paths:
            raise ValueError("paths['pool_parquet'] not set; call TeachingPool.load_from_parquet(...) or construct(...) first.")
        out_dir = Path(self.paths["root"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = out_dir / "binned_pool.parquet"
        out_config = out_dir / "binned_pool_config.json"

        df = pd.read_parquet(self.paths["pool_parquet"])
        df = _coerce_types(df)  # keep external utility (shared elsewhere)

        # choose class labels
        if label_source == "base":
            base = self.load_base_labels(self.paths.get("sample_plan", out_dir / "sampled_normals.json"))
            df["class_label"] = df["session_id"].map(base).astype("Int64")
        elif label_source == "simplified":
            df["class_label"] = df["label_int"].astype("Int64")
        else:
            raise ValueError("label_source must be 'base' or 'simplified'")

        if "k" not in df.columns:
            raise ValueError("pool parquet must contain column 'k'")
        df["k"] = pd.to_numeric(df["k"], errors="coerce").round().astype("Int64")

        # diagnostics (per-class k hist)
        hist_per_class: dict[str, dict[int, int]] = {}
        for cls_id in [0, 1]:
            vc = df.loc[df["class_label"] == cls_id, "k"].value_counts(dropna=True).sort_index()
            hist_per_class[str(cls_id)] = {int(k): int(v) for k, v in vc.items()}

        # records the bin edges, number of bins, and counts per bin
        bin_edges_per_class: dict[str, list[float]] = {}
        bin_label_edges_int: dict[str, list[int]] = {}
        actual_bins: dict[str, int] = {}
        counts_per_bin: dict[str, list[int]] = {}

        # destination columns
        df["k_bin_idx"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["k_bin_label"] = pd.Series(pd.NA, index=df.index, dtype="object")

        for cls_id in [0, 1]:
            mask = df["class_label"] == cls_id
            k_series = df.loc[mask, "k"].dropna()
            if k_series.empty:
                bin_edges_per_class[str(cls_id)] = []
                bin_label_edges_int[str(cls_id)] = []
                actual_bins[str(cls_id)] = 0
                counts_per_bin[str(cls_id)] = []
                continue

            k_min, k_max = int(k_series.min()), int(k_series.max())

            # build bins
            if binning == "fixed":
                edges = fixed_edges_per_class.get(str(cls_id), []) if fixed_edges_per_class else []
                if not edges or len(edges) < 2:
                    rng = k_max - k_min
                    if rng <= 5:
                        edges = list(range(k_min, k_max + 1))
                    else:
                        edges = np.linspace(k_min, k_max, num=max(3, target_bins) + 1).tolist()
                edges = np.array(np.unique(np.array(edges, dtype=float)), dtype=float)
                if edges.size < 2:
                    edges = np.array([k_min, k_max], dtype=float)
                cats, used_edges = pd.cut(
                    k_series,
                    bins=edges,
                    include_lowest=True,
                    right=True,
                    ordered=True,
                    duplicates="drop",
                    retbins=True,
                )
            else:
                def try_q(q: int):
                    if q < 1:
                        return None
                    try:
                        return pd.qcut(k_series, q=q, duplicates="drop")
                    except Exception:
                        return None

                cats = try_q(target_bins)
                if (cats is None) or (cats.cat.categories.size < min_bins):
                    for q in range(min(target_bins - 1, max_bins), min_bins - 1, -1):
                        cand = try_q(q)
                        if cand is not None and cand.cat.categories.size >= min_bins:
                            cats = cand
                            break
                if (cats is None) or (cats.cat.categories.size < min_bins):
                    for q in range(target_bins + 1, max_bins + 1):
                        cand = try_q(q)
                        if cand is not None and cand.cat.categories.size >= min_bins:
                            cats = cand
                            break

                if cats is not None and cats.cat.categories.size >= 1:
                    cats_full = cats.reindex(df.index[mask])
                    c = cats.cat.categories
                    lefts = [float(iv.left) for iv in c]
                    rights = [float(iv.right) for iv in c]
                    used_edges = np.array([lefts[0], *rights], dtype=float)
                else:
                    fallback_edges = np.linspace(k_min, k_max, num=max(min_bins, 2), dtype=float)
                    cats_full, used_edges = pd.cut(
                        k_series,
                        bins=fallback_edges,
                        include_lowest=True,
                        right=True,
                        ordered=True,
                        duplicates="drop",
                        retbins=True,
                    )
                    cats = cats_full

            # align categoricals and build codes
            if isinstance(cats, pd.Categorical):
                cats_aligned = cats.reindex(k_series.index)
                categories = cats_aligned.cat.categories
                codes = cats_aligned.cat.codes.astype("Int64")
            else:
                cats_aligned = cats
                categories = cats_aligned.cat.categories
                codes = cats_aligned.cat.codes.astype("Int64")

            n_bins = int(len(categories))
            if n_bins == 0:
                bin_edges_per_class[str(cls_id)] = []
                bin_label_edges_int[str(cls_id)] = []
                actual_bins[str(cls_id)] = 0
                counts_per_bin[str(cls_id)] = []
                continue

            used_edges = np.asarray(used_edges, dtype=float)
            int_edges = [int(math.floor(used_edges[0]))] + [int(math.ceil(x)) for x in used_edges[1:]]
            if ensure_extrema:
                int_edges[0] = k_min
                int_edges[-1] = k_max

            df.loc[k_series.index, "k_bin_idx"] = codes
            labels_str = [f"[{int_edges[i]}, {int_edges[i+1]}]" for i in range(n_bins)]
            code_to_label = {i: labels_str[i] for i in range(n_bins)}
            df.loc[k_series.index, "k_bin_label"] = df.loc[k_series.index, "k_bin_idx"].map(code_to_label)

            counts = [int((codes == i).sum()) for i in range(n_bins)]

            bin_edges_per_class[str(cls_id)] = used_edges.tolist()
            bin_label_edges_int[str(cls_id)] = int_edges
            actual_bins[str(cls_id)] = n_bins
            counts_per_bin[str(cls_id)] = counts

            has_min = int((df.loc[mask, "k"] == k_min).sum())
            has_max = int((df.loc[mask, "k"] == k_max).sum())
            if has_min == 0 or has_max == 0:
                print(
                    f"[warn] class {cls_id}: observed k range [{k_min},{k_max}] but "
                    f"{'missing min_k ' if has_min==0 else ''}"
                    f"{'and ' if has_min==0 and has_max==0 else ''}"
                    f"{'missing max_k' if has_max==0 else ''} in raw pool."
                )

        cols = [
            "session_id", "class_label", "k", "k_bin_idx", "k_bin_label",
            "label_int", "label_text", "robust_prob", "margin", "threshold",
            "emb_dim", "emb", "raw_power_path", "sts_full_path", "piv_path",
            "raw_soc_path", "sts_soc_path", "piv_soc_path", "model_id",
        ]
        cols = [c for c in cols if c in df.columns]
        df_binned = df[cols].copy()

        meta = {
            "label_source": label_source,
            "binning": binning,
            "target_bins": int(target_bins),
            "actual_bins": actual_bins,
            "bin_edges_per_class_float": bin_edges_per_class,
            "bin_label_edges_per_class_int": bin_label_edges_int,
            "counts_per_bin": counts_per_bin,
            "k_hist_per_class": hist_per_class,
            "timestamp": int(time.time()),
            "pool_parquet": str(self.paths["pool_parquet"]),
            "sample_plan": str(self.paths.get("sample_plan", out_dir / "sampled_normals.json")),
        }

        if verbose:
            print(f"rows in pool: {len(df_binned)}")
            for cls_id in [0, 1]:
                print(f"class {cls_id}:")
                hist = hist_per_class.get(str(cls_id), {})
                if hist:
                    ks = sorted(hist.keys())
                    print(f"  unique k: {len(ks)}  range: [{ks[0]},{ks[-1]}]")
                    print(f"  k counts: {hist}")
                n_bins = actual_bins.get(str(cls_id), 0)
                if n_bins > 0:
                    print(f"  bins ({n_bins}): labels={bin_label_edges_int[str(cls_id)]}  counts={counts_per_bin[str(cls_id)]}")
                else:
                    print("  bins: none")

        if save_outputs:
            df_binned.to_parquet(out_parquet, index=False)
            with open(out_config, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.bins_df = df_binned
        self.bins_meta = meta
        return df_binned, meta

    def derive_per_bin_budget(
        self, *, per_class_target: int, bin_allocation: Literal["even", "proportional"] = "even",
    ) -> dict[str, dict[str, int]]:
        """Derives per-class × k-bin budgets from the *binned* pool.

        For "even", splits the per-class target evenly across that class's k-bins (remainder to
        densest bins). For "proportional", splits by per-bin availability and fixes rounding drift.

        Args:
            per_class_target: Target count per class (upper bound; may be reduced by availability).
            bin_allocation: "even" or "proportional".

        Returns:
            dict[str, dict[str, int]]: e.g. {"0": {"[1, 2]": 20, ...}, "1": {...}}.

        Raises:
            ValueError: If `self.bins_df` is empty or missing required columns.
        """
        if self.bins_df is None or len(self.bins_df) == 0:
            raise ValueError("bins_df is empty. Run pool.bin_pool(...) before deriving per-bin budgets.")

        df = self.bins_df.copy()
        if "class_label" not in df.columns:
            if "label_int" in df.columns:
                df = df.rename(columns={"label_int": "class_label"})
            else:
                raise ValueError("binned pool must include 'class_label' or 'label_int'")

        out: dict[str, dict[str, int]] = {"0": {}, "1": {}}
        for c in sorted(df["class_label"].dropna().unique().tolist()):
            c = int(c)
            g = df[df["class_label"] == c]

            per_bin_counts = (
                g.groupby(["k_bin_idx", "k_bin_label"], dropna=False)["session_id"]
                .count()
                .reset_index(name="n")
                .sort_values(["k_bin_idx"])
            )
            bins = per_bin_counts["k_bin_label"].astype(str).tolist()
            counts = per_bin_counts["n"].astype(int).tolist()
            if not bins:
                out[str(c)] = {}
                continue

            if bin_allocation == "even":
                base = per_class_target // len(bins)
                rem = per_class_target - base * len(bins)
                densest_order = np.argsort(-np.asarray(counts))
                budgets = [base] * len(bins)
                for i in densest_order[:rem]:
                    budgets[i] += 1
            elif bin_allocation == "proportional":
                total = sum(counts)
                if total <= 0:
                    budgets = [0] * len(bins)
                else:
                    props = [per_class_target * (n / total) for n in counts]
                    budgets = [int(round(x)) for x in props]
                    drift = per_class_target - sum(budgets)
                    if drift != 0:
                        order = np.argsort(-np.asarray(counts))
                        idxs = order if drift > 0 else order[::-1]
                        for i in idxs:
                            if drift == 0:
                                break
                            budgets[i] += 1 if drift > 0 else -1
                            drift += -1 if drift > 0 else 1
            else:
                raise ValueError("bin_allocation must be 'even' or 'proportional'")

            out[str(c)] = {str(b): int(max(0, B)) for b, B in zip(bins, budgets)}
        return out

    # ---------- reporting ----------

    def describe(self) -> None:
        """Prints summary statistics for the pool and (if present) the binned strata."""
        if self.pool_df is None:
            print("teaching pool: empty")
            return
        df = self.pool_df
        n = len(df)
        by_class = df.groupby("label_int")["session_id"].nunique().rename("n_sessions")
        print(f"[teaching pool] rows={n}, classes:\n{by_class.to_string()}")
        if "k" in df.columns:
            print("\n[k] stats:\n", df["k"].describe().to_string())
        if self.bins_df is not None:
            bd = (
                self.bins_df.groupby(["label_int", "k_bin_label"])["session_id"]
                .nunique()
                .rename("n")
                .reset_index()
            )
            print("\n[bins] counts per (class, k_bin_label):")
            print(bd.pivot(index="k_bin_label", columns="label_int", values="n").fillna(0).astype(int).to_string())

    # ---------- analysis helpers ----------

    @staticmethod
    def selection_vs_pool_report(pool_bins_df: pd.DataFrame, selected_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Produces per-class, per-k and per-bin comparison tables between pool and selection."""
        return selection_vs_pool_report(pool_bins_df, selected_df)

    # ---------- moved helpers (class-visible) ----------

    @staticmethod
    def ensure_dirs(root: Path) -> dict[str, Path]:
        """Creates output directories for arrays and returns their paths."""
        root.mkdir(parents=True, exist_ok=True)
        d = {
            "root": root,
            "sts_full": root / "sts_full",
            "piv": root / "piv",
            "raw_power": root / "raw_power",
            "raw_soc": root / "raw_soc",
            "piv_soc": root / "piv_soc",
            "sts_soc": root / "sts_soc",
        }
        for p in d.values():
            if p != root:
                p.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def db_connect(db_path: Path) -> sqlite3.Connection:
        """Opens a SQLite connection with pragmatic PRAGMA settings."""
        conn = sqlite3.connect(str(db_path), timeout=60)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        return conn

    @staticmethod
    def sample_sessions(abnormal_ids: Sequence[int], normal_ids: Sequence[int], *, random_seed: Optional[int] = None) -> dict[str, list[int]]:
        """Selects all abnormal sessions and samples the same number of normal sessions."""
        rng = np.random.default_rng(random_seed)
        n = len(abnormal_ids)
        if len(normal_ids) < n:
            raise ValueError(f"not enough normal sessions to sample {n} (have {len(normal_ids)} normal, need {n}).")
        sel_normals = sorted(rng.choice(normal_ids, size=n, replace=False).tolist())
        return {"abnormal": sorted(list(abnormal_ids)), "normal": sel_normals}

    @staticmethod
    def compute_base_labels(
        test_loader,
        model: torch.nn.Module,
        device: torch.device,
        *,
        power_scaler,
        soc_scaler,
        idx_power_inp: int,
        idx_soc_inp: int,
        power_weight: float,
        decay_lambda: float,
        t_min_eval: int,
        threshold: float,
    ) -> tuple[list[int], list[int], dict[int, float]]:
        """Runs one pass over the test set to compute Macro-RMSE and base labels.

        Returns:
            Tuple of (abnormal_ids, normal_ids, err_by_id).
        """
        abnormal, normal = [], []
        err_by_id: dict[int, float] = {}

        for b_idx, batch in enumerate(test_loader):
            if len(batch) == 4:
                sids, Xb, Yb, Ls = batch
                sids = [int(x) for x in (sids.tolist() if hasattr(sids, "tolist") else sids)]
            else:
                Xb, Yb, Ls = batch
                sids = [None] * Xb.shape[0]

            B = Xb.shape[0]
            X_dev = Xb.to(device)
            L_cpu = Ls.cpu() if hasattr(Ls, "cpu") else torch.tensor(Ls)

            # reconstruct absolute predictions in original units
            P_res = predict_residuals(model, X_dev, L_cpu, device=device)
            P_abs_scaled = reconstruct_abs_from_residuals_batch(Xb, P_res, idx_power_inp, idx_soc_inp)
            P_abs = inverse_targets_np(P_abs_scaled.numpy(), power_scaler, soc_scaler)

            for i in range(B):
                T = int(L_cpu[i].item())
                power_true = power_scaler.inverse_transform(Xb[i, :T, [idx_power_inp]].numpy()).ravel()
                soc_true = soc_scaler.inverse_transform(Xb[i, :T, [idx_soc_inp]].numpy()).ravel()
                H = P_abs.shape[2]
                Y_abs_true = build_true_abs_from_series(power_true, soc_true, H)
                err = macro_rmse_from_abs(P_abs[i, :T], Y_abs_true, power_weight, decay_lambda, t_min_eval)
                sid = sids[i]
                if sid is None:
                    b = fetch_session_preds_bundle(
                        model,
                        test_loader,
                        batch_index=b_idx,
                        sample_index=i,
                        device=device,
                        power_scaler=power_scaler,
                        soc_scaler=soc_scaler,
                        idx_power_inp=idx_power_inp,
                        idx_soc_inp=idx_soc_inp,
                        session_id=None,
                    )
                    sid = int(b.session_id)
                sid = int(sid)
                err_by_id[sid] = float(err)
                if err > float(threshold):
                    abnormal.append(sid)
                else:
                    normal.append(sid)

        return abnormal, normal, err_by_id

    @staticmethod
    def load_base_labels(sample_plan_json: str | Path) -> dict[int, int]:
        """Builds `{session_id -> base_label}` from `sampled_normals.json` (0 normal, 1 abnormal)."""
        plan = json.loads(Path(sample_plan_json).read_text(encoding="utf-8"))
        base: dict[int, int] = {}
        for sid in plan.get("abnormal", []):
            base[int(sid)] = 1
        for sid in plan.get("normal", []):
            base[int(sid)] = 0
        return base

    # ---------- wrappers for helper functions ----------
    @staticmethod
    def create_main_table(conn: sqlite3.Connection) -> None:
        """Creates the main SQLite table if missing (delegates to shared helper)."""
        _create_main_table(conn)

    @staticmethod
    def upsert_row(conn: sqlite3.Connection, row: dict) -> None:
        """Upserts a row into the SQLite cache (delegates to shared helper)."""
        _upsert_row(conn, row)

    @staticmethod
    def rows_to_parquet(db_path: Path, out_path: Path) -> None:
        """Exports cached rows to Parquet (delegates to shared helper)."""
        _rows_to_parquet(db_path, out_path)

    @staticmethod
    def export_config(config: "TeachingPoolConfig", root: Path, *, n_abn: int, n_norm: int) -> None:
        """Writes build-time config/provenance (delegates to shared helper)."""
        _export_config(config, root, n_abn=n_abn, n_norm=n_norm)

    @staticmethod
    def prepare_ors_params_for_T(params: ORSParams, T: int) -> ORSParams:
        """Returns ORSParams clamped/adapted to sequence length T (delegates to shared helper)."""
        return _prepare_ors_params_for_T(params, T)

    @staticmethod
    def validate_ors_result(res: dict) -> tuple[bool, str | None]:
        """Validates an ORS result structure (delegates to shared helper)."""
        return _validate_ors_result(res)

    @staticmethod
    def compute_embedding(sts: np.ndarray, *, L: int, P: int) -> np.ndarray:
        """Computes the spike-aware embedding used for facility-location (delegates to shared helper)."""
        return _compute_embedding(sts, L=L, P=P)



@dataclass
class TeachingPoolConfig:
    """Configures the ORS pool build task and file outputs.

    Attributes:
        model_path: Path to the trained model file.
        output_dir: Directory for parquet/sqlite/arrays.
        threshold: Macro-RMSE threshold used for base labels.
        random_seed: RNG seed for reproducibility.
        device: Device string accepted by torch.
        export_every: Write cache rows every n items.
        L: Resampling length used in embeddings.
        P: Number of top peaks used in embeddings.
        ors_params: Parameters for robust simplification.
        power_weight: Macro-RMSE weighting for power.
        decay_lambda: Exponential decay factor in Macro-RMSE.
    """
    model_path: str = "../Models/final/final_model.pth"
    output_dir: str = "../Data/teaching_pool"
    threshold: float = 8.5962
    random_seed: Optional[int] = None
    device: torch.device = torch.device("cuda"),
    export_every: int = 10
    L: int = 128
    P: int = 4
    power_weight: Optional[float] = 0.5
    decay_lambda: Optional[float] = 0.2
    ors_params: ORSParams = field(
        default_factory=lambda: ORSParams(
            stage1_mode="dp_prefix",
            stage2_err_metric="l2",
            dp_q=1000,
            rdp_stage1_candidates=100,
            dp_alpha=0.001,
            beta=3.0,
            gamma=0.05,
            R=10000,
            epsilon_mode="fraction",
            epsilon_value=0.3,
            t_min_eval=1,
            min_k=1,
            max_k=100,
            random_seed=None,  # defer seed assignment - set in post_init to reuse self.random_seed
            soc_stage1_mode="rdp",
            soc_rdp_epsilon=0.75,
            soc_rdp_candidates=5,
            soc_rdp_eps_min=1e-6,
            soc_rdp_eps_max=100.0,
        )
    )

    def __post_init__(self):
        """Synchronises ORSParams with TeachingPoolConfig RNG seed."""
        if self.ors_params.random_seed != self.random_seed:
            self.ors_params.random_seed = self.random_seed
        self.ors_params.model_id = getattr(self, "model_path", None)



# ---------------- Teaching Set construction (selection) --------------- #

class TeachingSet:
    """Represents a teaching set with methods for set construction, serving examples 
    (e.g. during a teaching session) and set analytics. The construct method builds 
    a diverse teaching subset from a binned `TeachingPool` using a lazy-greedy facility-location 
    objective over spike-aware embeddings with optional linear terms for decision margin and robustness. 
    It then exposes group-specific samplers (A/B/C) that differ in ordering and overlays, while sharing the
    same teaching set (selected charging session IDs) across groups.

    The selection objective is:
        F(S) = ∑_i max_{s ∈ S} sim(i, s)  +  λ_margin · margin(s)  +  λ_robust · robust_prob(s)
    where `sim` is cosine similarity on L2-normalised embeddings. Selection proceeds
    per (class, k-bin) subject to budgets, with optional length filtering and spillover
    to nearest k-bins when availability is insufficient.

    Args:
        pool: Teaching pool containing the stratified/binned candidates (`pool.bins_df` must exist).
        per_bin_budget: Optional explicit budget per (class, k-bin) as nested dict {"0":{"[1,2]":n,...},"1":{...}}.
        per_class_target: Target count per class if `per_bin_budget` is not provided (used to derive budgets).
        bin_allocation: "even" or "proportional" split of `per_class_target` across a class's k-bins.
        enforce_even_class_dist: If True, caps both classes to the minimum achievable total subject to availability.
        length_range: Optional (min_len, max_len) to filter raw session length before selection;
            use None for open bound, e.g. (10, None) keeps length ≥ 10.
        lambda_margin: Weight for the decision-margin linear term.
        lambda_robust: Weight for the robustness-probability linear term.
        normalize_embeddings: If True, L2-normalises embeddings before cosine similarity.
        lazy_prune: If True, uses lazy upper bounds to reduce recomputations in greedy selection.
        dtw_tie_refine: If True, applies an optional DTW-based tie-break; ignored if no ties occur.
        dtw_params: Optional DTW parameters (kept for compatibility; not required by the core objective).
        sim_clip_min: Lower clip for pairwise sim (e.g. 0.0 keeps only non-negative contributions).
        random_seed: RNG seed for deterministic budget splits and iterator sampling.
        min_per_k: Enforces at least this many seeds per distinct k in a (class, k-bin) before greedy fill.
        output_dir: Base directory for `selection.parquet` and `selection_config.json`.

    Attributes:
        pool: Source `TeachingPool`.
        teaching_set_df: Selected rows (DataFrame) with decision order and selection metadata.
        meta: Selection config and diagnostics (budgets, coverage, spillovers, diversity per bin).
        random_seed: Stored RNG seed used for iterators and reproducibility.

    Notes:
        - Embeddings are expected in column `emb` as a list/array with `emb_dim`.
        - Class label column is `class_label` (mapped from `label_int` if needed).
        - k-bin label column is `k_bin_label` (or `k_bin`, which is renamed).
        - Length filtering prefers `raw_power_path` to infer length; falls back to `sts_full_path`.
        - Group semantics:
            A → curriculum in iterator, shows simplification overlays,
            B → random in iterator, shows simplification overlays,
            C → random in iterator, raw-only (no overlays).
    """

    pool: TeachingPool = None
    teaching_set_df: pd.DataFrame = None
    meta: dict = None
    random_seed: int = None
    ors_params: ORSParams = None

    def __init__(self, pool: TeachingPool, *,
                 per_bin_budget: dict[str, dict[str, int]] | None = None,
                 per_class_target: int | None = 100,
                 bin_allocation: str = "even",
                 enforce_even_class_dist: bool = False,
                 length_range: Optional[tuple] = None,
                 lambda_margin: float = 0.10,
                 lambda_robust: float = 0.05,
                 normalize_embeddings: bool = True,
                 lazy_prune: bool = True,
                 dtw_tie_refine: bool = False,
                 dtw_params: dict | None = None,
                 sim_clip_min: float | None = 0.0,
                 random_seed: int | None = None,
                 min_per_k: int = 0,
                 output_dir: str | Path | None = None) -> None:

        if pool.bins_df is None:
            raise ValueError("pool.bins_df is None; run pool.bin_pool(...) first.")
        out_dir = Path(output_dir) if output_dir is not None else pool.paths.get("root", Path("."))

        selected_df, meta = self.construct(
            pool.bins_df,
            per_bin_budget=per_bin_budget,
            per_class_target=per_class_target,
            min_per_k=min_per_k,
            bin_allocation=bin_allocation,
            enforce_even_class_dist=enforce_even_class_dist,
            length_range=length_range,
            lambda_margin=lambda_margin,
            lambda_robust=lambda_robust,
            normalize_embeddings=normalize_embeddings,
            lazy_prune=lazy_prune,
            dtw_tie_refine=dtw_tie_refine,
            dtw_params=dtw_params,
            sim_clip_min=sim_clip_min,
            random_seed=random_seed,
            output_dir=(Path(output_dir) if output_dir else pool.paths["root"]),
        )

        self.pool = pool
        self.teaching_set_df = selected_df
        self.meta = meta
        self.random_seed = random_seed
        self.ors_params = pool.config.ors_params

        # iterators are attached lazily by build_group_iterators()
        self.iter_A = None
        self.iter_B = None
        self.iter_C = None
        self.last_figs: dict[str, list[Figure]] | None = None
        self.last_meta: dict[str, list[dict]] | None = None

    def construct(self,
                  strata_df: pd.DataFrame,
                  *,
                  per_bin_budget: dict[str, dict[str, int]] | None = None,
                  per_class_target: int | None = 100,
                  min_per_k: int = 0,
                  bin_allocation: str = "even",
                  enforce_even_class_dist: bool = False,
                  length_range: Optional[tuple] = None,
                  lambda_margin: float = 0.10,
                  lambda_robust: float = 0.05,
                  normalize_embeddings: bool = True,
                  lazy_prune: bool = True,
                  dtw_tie_refine: bool = False,
                  dtw_params: dict | None = None,
                  sim_clip_min: float | None = 0.0,
                  random_seed: Optional[int] = None,
                  output_dir: str | Path = "Data/teaching_pool",
                  ) -> tuple[pd.DataFrame, dict]:
        """Constructs the diverse subset per (class, k-bin) and returns (selection, meta).

        Selection uses lazy-greedy facility location over L2-normalised embeddings with
        optional linear bonus λ_margin·margin + λ_robust·robust_prob, subject to per-bin
        budgets (explicit or derived from per-class targets). Shortfalls spill to nearest
        k-bins within the same class.

        Args:
            strata_df: Stratified pool (typically `pool.bins_df`) with columns: session_id, class_label/label_int,
                k/k_bin_label (or k_bin), emb_dim, emb, margin, robust_prob, and optional paths for length inference.
            per_bin_budget: Explicit budgets per (class, k-bin). If None, derived from `per_class_target`.
            per_class_target: Per-class target used to derive per-bin budgets when `per_bin_budget` is None.
            min_per_k: Seeds at least this many selections per distinct k inside each (class, k-bin) (up to budget).
            bin_allocation: "even" or "proportional" when deriving budgets from `per_class_target`.
            enforce_even_class_dist: Caps totals so both classes select the same achievable count.
            length_range: Optional (min_len, max_len) filter on raw-session length.
            lambda_margin: Weight for decision-margin bonus.
            lambda_robust: Weight for robustness-probability bonus.
            normalize_embeddings: If True, applies L2 normalisation before cosine similarity.
            lazy_prune: Enables lazy upper bounds for faster greedy selection.
            dtw_tie_refine: Optional DTW-based tie-break (kept for compatibility).
            dtw_params: DTW parameters if `dtw_tie_refine` is True.
            sim_clip_min: Lower bound clip for similarities (e.g. 0.0 discards negative sim).
            random_seed: RNG seed for deterministic behaviour.
            output_dir: Output directory for `selection.parquet` and `selection_config.json`.

        Returns:
            tuple[pd.DataFrame, dict]: `(selected_df, meta)` where:
                - `selected_df` holds chosen rows with selection metadata (rank, scores, neighbours).
                - `meta` contains config, budgets, coverage/diversity diagnostics, and spillover info.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # schema + embeddings
        df = strata_df.copy()
        df = _coerce_types(df)
        if "class_label" not in df.columns and "label_int" in df.columns:
            df = df.rename(columns={"label_int": "class_label"})
        if "k_bin_label" not in df.columns:
            if "k_bin" in df.columns:
                df = df.rename(columns={"k_bin": "k_bin_label"})
            else:
                raise ValueError("expected 'k_bin_label' (or 'k_bin') in strata_df")
        df = _normalise_embedding_column(
            df, int(df["emb_dim"].dropna().iloc[0] if "emb_dim" in df.columns else 264)
        )
        emb_dim = int(df["emb_dim"].dropna().mode().iloc[0]) if "emb_dim" in df.columns else 264

        mask_ok = df["emb"].map(lambda x: isinstance(x, list) and len(x) == emb_dim)
        n_bad = int((~mask_ok).sum())
        if n_bad > 0:
            print(f"[teach][warn] skipping {n_bad} rows with invalid embeddings for selection.")
        df = df[mask_ok].reset_index(drop=True)

        # optional: filter by raw-session length
        if length_range is not None:
            lo_raw, hi_raw = length_range
            if "length" not in df.columns:
                def _len_from_row(row) -> int | None:
                    p = row.get("raw_power_path") or row.get("sts_full_path")
                    if isinstance(p, str) and os.path.exists(p):
                        try:
                            return int(np.load(p, mmap_mode="r").shape[0])
                        except Exception:
                            return None
                    return None
                df["length"] = df.apply(_len_from_row, axis=1)
            pool_max = int(pd.to_numeric(df["length"], errors="coerce").max())
            lo = 0 if lo_raw is None else int(lo_raw)
            hi = pool_max if hi_raw is None else int(hi_raw)
            before = len(df)
            df = df[(pd.to_numeric(df["length"], errors="coerce") >= lo) &
                    (pd.to_numeric(df["length"], errors="coerce") <= hi)].copy()
            print(f"[teach] length filter [{lo}, {hi}] kept {len(df)} / {before} rows.")

        # budgets
        if per_bin_budget is None:
            if per_class_target is None:
                raise ValueError("either provide per_bin_budget or set per_class_target")
            per_bin_budget = derive_per_bin_budget(
                df, per_class_target=per_class_target, bin_allocation=bin_allocation
            )
        per_bin_budget = self.spillover_by_counts(df, per_bin_budget)
        if enforce_even_class_dist:
            per_bin_budget = self.cap_budgets_to_even_totals(df, per_bin_budget)

        # selection containers
        sel_rows: list[dict] = []
        spill_diag: dict[str, dict[str, dict[str, int]]] = {"0": {}, "1": {}}
        per_bin_selected: dict[str, dict[str, int]] = {"0": {}, "1": {}}
        per_class_rank = {0: 0, 1: 0}

        # iterate classes then k-bins in canonical order
        for c in (0, 1):
            g_class = df[df["class_label"] == c]

            if "k_bin_idx" in g_class.columns:
                bin_order = (
                    g_class.groupby(["k_bin_idx", "k_bin_label"], dropna=False)["session_id"]
                    .count()
                    .reset_index()
                    .sort_values("k_bin_idx")["k_bin_label"]
                    .astype(str)
                    .tolist()
                )
            else:
                bin_order = sorted(g_class["k_bin_label"].dropna().astype(str).unique().tolist())

            for bin_label in bin_order:
                budget = int(per_bin_budget.get(str(c), {}).get(str(bin_label), 0))
                g = g_class[g_class["k_bin_label"].astype(str) == str(bin_label)]
                n_avail = len(g)
                if budget <= 0 or n_avail == 0:
                    spill_diag[str(c)][str(bin_label)] = {
                        "available": int(n_avail),
                        "budget": int(budget),
                        "selected": 0,
                        "spillover_in": 0,
                        "spillover_out": int(max(0, budget - n_avail)),
                    }
                    per_bin_selected[str(c)][str(bin_label)] = 0
                    continue

                # build matrices/vectors
                E = np.asarray(g["emb"].to_list(), dtype=np.float32)
                if normalize_embeddings:
                    norms = np.linalg.norm(E, axis=1, keepdims=True)
                    norms[norms == 0.0] = 1.0
                    E = E / norms
                g_cov = np.zeros(E.shape[0], dtype=np.float32)

                margins = g.get("margin", pd.Series(0.0, index=g.index)).fillna(0.0).to_numpy(dtype=np.float32)
                robusts = g.get("robust_prob", pd.Series(0.0, index=g.index)).fillna(0.0).to_numpy(dtype=np.float32)
                lin_term = lambda_margin * margins + lambda_robust * robusts

                k_vals = g.get("k", pd.Series(np.inf, index=g.index)).fillna(np.inf).to_numpy(dtype=np.float32)
                sids = g["session_id"].to_numpy(dtype=np.int64)

                selected_local: list[int] = []

                # seeding: ensure at least `min_per_k` per distinct k
                if min_per_k > 0 and budget > 0:
                    k_to_idxs: dict[float, np.ndarray] = {}
                    for kv in np.unique(k_vals[~np.isnan(k_vals)]):
                        k_to_idxs[float(kv)] = np.where(k_vals == kv)[0]

                    seed_pairs: list[tuple[float, int]] = []
                    for kv, idxs in k_to_idxs.items():
                        sc = []
                        for idx in idxs:
                            sim_vec = E @ E[idx]
                            if sim_clip_min is not None and sim_clip_min > -1.0:
                                sim_vec = np.clip(sim_vec, sim_clip_min, 1.0)
                            delta0 = float(np.maximum(0.0, sim_vec).sum())
                            ub = delta0 + float(lin_term[idx])
                            tie = (-margins[idx], -robusts[idx], float(k_vals[idx]), int(sids[idx]))
                            sc.append((ub, tie, idx))
                        sc.sort(key=lambda t: (t[0], t[1]), reverse=True)
                        for j in range(min(min_per_k, len(sc))):
                            seed_pairs.append((sc[j][0], sc[j][2]))

                    seed_pairs.sort(key=lambda t: t[0], reverse=True)
                    used: set[int] = set()
                    for _, idx in seed_pairs:
                        if len(selected_local) >= budget:
                            break
                        if idx in used:
                            continue
                        sim_vec = E @ E[idx]
                        if sim_clip_min is not None and sim_clip_min > -1.0:
                            sim_vec = np.clip(sim_vec, sim_clip_min, 1.0)
                        delta = float(np.maximum(0.0, sim_vec - g_cov).sum())
                        score = delta + float(lin_term[idx])

                        g_cov = np.maximum(g_cov, sim_vec)
                        selected_local.append(idx)
                        used.add(idx)

                        topn = int(min(5, sim_vec.shape[0]))
                        nn_idx = np.argpartition(-sim_vec, kth=topn - 1)[:topn]
                        nn_order = nn_idx[np.argsort(-sim_vec[nn_idx])]
                        chosen_neighbors = [int(sids[j]) for j in nn_order.tolist()]

                        row = g.iloc[idx].to_dict()
                        row["gain_coverage"] = float(delta)
                        row["score_total"] = float(score)
                        row["rank_in_bin"] = int(len(selected_local))
                        per_class_rank[c] += 1
                        row["rank_in_class"] = int(per_class_rank[c])
                        row["k_bin"] = str(bin_label)
                        row["chosen_neighbors"] = chosen_neighbors
                        sel_rows.append(row)

                # remaining budget via lazy-greedy with bounds
                remaining_budget = max(0, budget - len(selected_local))
                bounds = []
                for idx in range(E.shape[0]):
                    if idx in selected_local:
                        continue
                    sim_vec = E @ E[idx]
                    if sim_clip_min is not None and sim_clip_min > -1.0:
                        sim_vec = np.clip(sim_vec, sim_clip_min, 1.0)
                    delta_hat = float(np.maximum(0.0, sim_vec - g_cov).sum())
                    ub = delta_hat + float(lin_term[idx])
                    tie = (-margins[idx], -robusts[idx], float(k_vals[idx]), int(sids[idx]))
                    heapq.heappush(bounds, (-ub, tie, idx))

                step = 0
                while step < min(remaining_budget, n_avail - len(selected_local)) and bounds:
                    ub_neg, tie, idx = heapq.heappop(bounds)

                    sim_vec = E @ E[idx]
                    if sim_clip_min is not None and sim_clip_min > -1.0:
                        sim_vec = np.clip(sim_vec, sim_clip_min, 1.0)
                    delta = float(np.maximum(0.0, sim_vec - g_cov).sum())
                    score = delta + float(lin_term[idx])

                    accept = True
                    if lazy_prune and bounds:
                        next_best = -bounds[0][0]
                        if score + 1e-12 < next_best:
                            accept = False
                            heapq.heappush(bounds, (-score, tie, idx))

                    if accept:
                        g_cov = np.maximum(g_cov, sim_vec)
                        selected_local.append(idx)
                        step += 1

                        topn = int(min(5, sim_vec.shape[0]))
                        nn_idx = np.argpartition(-sim_vec, kth=topn - 1)[:topn]
                        nn_order = nn_idx[np.argsort(-sim_vec[nn_idx])]
                        chosen_neighbors = [int(sids[j]) for j in nn_order.tolist()]

                        row = g.iloc[idx].to_dict()
                        row["gain_coverage"] = float(delta)
                        row["score_total"] = float(score)
                        row["rank_in_bin"] = int(len(selected_local))
                        per_class_rank[c] += 1
                        row["rank_in_class"] = int(per_class_rank[c])
                        row["k_bin"] = str(bin_label)
                        row["chosen_neighbors"] = chosen_neighbors
                        sel_rows.append(row)

                per_bin_selected[str(c)][str(bin_label)] = int(len(selected_local))
                shortfall = int(max(0, budget - len(selected_local)))
                spill_diag[str(c)][str(bin_label)] = {
                    "available": int(n_avail),
                    "budget": int(budget),
                    "selected": int(len(selected_local)),
                    "spillover_in": int(max(0, len(selected_local) - min(budget, n_avail))),
                    "spillover_out": shortfall,
                }

        selected_df = pd.DataFrame(sel_rows).reset_index(drop=True)

        # diagnostics
        coverage_per_class = _compute_class_coverage(df, selected_df, normalize_embeddings, sim_clip_min)
        diversity_per_bin = _compute_avg_pairwise_bin(df, selected_df, normalize_embeddings, sim_clip_min)

        # persist
        sel_path = out_dir / "selection.parquet"
        selected_df.to_parquet(sel_path, index=False)

        config_obj = {
            "objective": "facility_location + lambda_margin*margin + lambda_robust*robust_prob",
            "similarity": "cosine on L2-normalized embeddings",
            "lazy_greedy": bool(lazy_prune),
            "dtw_tie_refine": bool(dtw_tie_refine),
            "lambda_margin": float(lambda_margin),
            "lambda_robust": float(lambda_robust),
            "min_per_k": int(min_per_k),
            "per_bin_budget": per_bin_budget,
            "spillover_policy": "nearest-k bins within class",
            "random_seed": int(random_seed) if random_seed else None,
            "files": {
                "strata": str((out_dir / "binned_pool.parquet").as_posix()),
                "selection": str(sel_path.as_posix()),
            },
            "timestamp": int(time.time()),
            "coverage_per_class": coverage_per_class,
            "per_bin_selected": per_bin_selected,
            "spillover_diag": spill_diag,
            "diversity_per_bin": diversity_per_bin,
        }

        with open(out_dir / "selection_config.json", "w") as f:
            json.dump(config_obj, f, indent=2)

        print("[teach] selection complete.")
        for k, v in coverage_per_class.items():
            n_cls = int((selected_df["class_label"] == int(k)).sum())
            print(f"  class {k}: selected={n_cls} | F(S)={v:.4f}")
        print(f"  wrote → {sel_path}")
        print(f"  wrote → {(out_dir / 'selection_config.json')}")

        return selected_df, config_obj

    # ---------------- spillover helpers (moved from top-level) ---------------- #

    def spillover_by_counts(self, df: pd.DataFrame, per_bin_budget: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        """Redistributes shortfalls to nearest k-bins (within class) proportional to availability.

        Args:
            df: Stratified pool with columns `class_label` and `k_bin_label` (or `k_bin_idx`).
            per_bin_budget: Nested dict of desired counts per (class, k-bin).

        Returns:
            dict[str, dict[str, int]]: Adjusted budgets that respect availability as closely as possible.
        """
        out = copy.deepcopy(per_bin_budget)
        for c in (0, 1):
            sub = df[df["class_label"] == c]
            if sub.empty:
                continue

            if "k_bin_idx" in sub.columns:
                order = (
                    sub.groupby(["k_bin_idx", "k_bin_label"], dropna=False)["session_id"]
                    .count()
                    .reset_index()
                    .sort_values("k_bin_idx")["k_bin_label"]
                    .astype(str)
                    .tolist()
                )
            else:
                order = sorted(sub["k_bin_label"].dropna().astype(str).unique().tolist())
            if not order:
                continue

            counts = {b: int((sub["k_bin_label"].astype(str) == b).sum()) for b in order}
            want = {b: int(out.get(str(c), {}).get(b, 0)) for b in order}
            short = {b: max(0, want[b] - counts[b]) for b in order}
            extra = {b: max(0, counts[b] - want[b]) for b in order}

            for j, b in enumerate(order):
                need = short[b]
                if need <= 0:
                    continue
                left = list(range(j - 1, -1, -1))
                right = list(range(j + 1, len(order)))
                for idx in left + right:
                    donor = order[idx]
                    can = extra[donor]
                    if can <= 0:
                        continue
                    take = min(need, can)
                    want[donor] -= take
                    extra[donor] -= take
                    want[b] += take
                    need -= take
                    if need <= 0:
                        break
            out[str(c)] = {b: int(max(0, want[b])) for b in order}
        return out

    def cap_budgets_to_even_totals(self, df: pd.DataFrame, budgets: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        """Caps both class totals to the common achievable minimum without exceeding per-bin availability.

        The method clamps each (class, bin) budget to availability, then proportionally
        down-scales the over-subscribed class. Rounding drift is resolved by distributing
        +1s to bins with the largest leftover, staying ≤ availability.

        Args:
            df: Stratified pool with `class_label` and `k_bin_label`.
            budgets: Nested dict of desired counts per (class, k-bin).

        Returns:
            dict[str, dict[str, int]]: Adjusted budgets with equal class totals and availability respected.
        """
        out = copy.deepcopy(budgets)
        sub = df[["class_label", "k_bin_label"]].copy()
        sub["k_bin_label"] = sub["k_bin_label"].astype(str)
        avail = sub.groupby(["class_label", "k_bin_label"]).size()
        classes = sorted(set(int(c) for c in sub["class_label"].unique()))
        if len(classes) < 2:
            return out

        def _achievable(c: int) -> int:
            want = out.get(str(c), {})
            total = 0
            for b, w in want.items():
                total += min(int(w), int(avail.get((c, str(b)), 0)))
            return int(total)

        n_cap = min(_achievable(c) for c in classes)

        for c in classes:
            want = {str(b): int(v) for b, v in out.get(str(c), {}).items()}
            current = {b: min(v, int(avail.get((c, str(b)), 0))) for b, v in want.items()}
            tot = sum(current.values())
            if tot <= n_cap:
                out[str(c)] = current
                continue
            factor = n_cap / max(1, tot)
            shrunk = {b: int(np.floor(v * factor)) for b, v in current.items()}
            drift = n_cap - sum(shrunk.values())
            if drift > 0:
                leftovers = sorted(((b, current[b] - shrunk[b]) for b in current.keys()),
                                   key=lambda t: t[1], reverse=True)
                i = 0
                while drift > 0 and leftovers:
                    b, space = leftovers[i]
                    if space > 0:
                        shrunk[b] += 1
                        drift -= 1
                    i = (i + 1) % len(leftovers)
            out[str(c)] = shrunk
        return out

    # ---------------- existing public API below (unchanged) ---------------- #

    def build_group_iterators(self) -> dict[str, TeachIterator]:
        """
        Builds and stores group iterators on this TeachingSet.
        alternation n/a/n/a… is handled inside TeachIterator.
        """
        if self.teaching_set_df is None:
            raise ValueError("selected_df is None; run selection first.")

        dfA = self.sample_group(group="A")
        dfB = self.sample_group(group="B")
        dfC = self.sample_group(group="C")

        # compute shared y-limits: [0, global max]
        y_max = _compute_global_power_max([dfA, dfB, dfC])
        y_lim = (0.0, y_max)

        # creates iterators for each group
        # Set TeachingSet as iterators' 'parent' attr to pass config/params
        self.iter_A = TeachIterator(df=dfA.reset_index(drop=True), group="A", random_seed=self.random_seed, y_lim=y_lim, parent=self)
        self.iter_B = TeachIterator(df=dfB.reset_index(drop=True), group="B", random_seed=self.random_seed, y_lim=y_lim, parent=self)
        self.iter_C = TeachIterator(df=dfC.reset_index(drop=True), group="C", random_seed=self.random_seed, y_lim=y_lim, parent=self)
        
        return {"A": self.iter_A, "B": self.iter_B, "C": self.iter_C}

    def ids(self) -> np.ndarray:
        """Returns the unique selected session IDs."""
        if self.teaching_set_df is None:
            return np.array([], dtype=int)
        return self.teaching_set_df["session_id"].unique()

    def sample_group(self, group: Literal["A", "B", "C"]) -> pd.DataFrame:
        """Returns a group-specific view without imposing order (ordering is in `TeachIterator`)."""
        if group.upper() not in {"A", "B", "C"}:
            raise ValueError(f"unknown group '{group}'")
        df = self.teaching_set_df.copy()
        return df.assign(group=group, show_simpl=group.upper() in ["A", "B"])

    def serve_examples(self, group: Literal["A", "B", "C", "All"], *, 
                       plot_examples: bool = False, n: Optional[int] = 10, 
                       save_dir: Optional[str | Path] = None, show_meta: bool = False) -> dict[str, list[Figure]]:
        """Serves and optionally saves and/or plots a batch of examples from the teaching set.

        Args: 
            group: The teaching set / user study trial group we want to serve examples for. Can provide one of A/B/C, multiple or "All".
                   Group A receives simplified + original power & simplified SOC ordered by k (normals asc, abnormals desc), 
                   group B receives same as A but unordered, and group C same as B but only original/raw data (power + SOC).
            plot_examples: Whether to immediately plot each example when serving
            n: How many examples to serve in total across all classes (normal/abnormal).
            save_dir: If specified, where to store the teaching set
            show_mta: Whether to print metadata about the examples (charging sessions) in the set such as session ids, k values, paths 

        Raises:
            ValueError: If a group different than A/B/C is provided for `group`

        Returns: 
            dict[str, list[Figure]]: Groups are keys, values are a list of examples (figures) belonging to the group.
        """
        if self.iter_A is None or self.iter_B is None or self.iter_C is None:
            self.build_group_iterators()
        groups: list[str] = ["A", "B", "C"] if str(group).upper() == "ALL" else [str(group).upper()]
        for g in groups:
            if g not in {"A", "B", "C"}:
                raise ValueError(f"unknown group '{group}'")

        base_dir: Optional[Path] = Path(save_dir) if save_dir is not None else None
        if base_dir is not None:
            for g in groups:
                (base_dir / g).mkdir(parents=True, exist_ok=True)

        iters = {"A": self.iter_A, "B": self.iter_B, "C": self.iter_C}
        out_figs: dict[str, list[Figure]] = {g: [] for g in groups}
        out_meta: dict[str, list[dict]] = {g: [] for g in groups}

        def _save_if_needed(fig: Figure, g: str, meta: dict) -> Optional[Path]:
            if base_dir is None:
                return None
            sid = meta.get("session_id", "unknown")
            lbl = str(meta.get("label", "unknown")).lower().replace(" ", "-")
            kval = meta.get("k", None)
            k_str = f"k{kval}" if kval is not None else "kNA"
            if g == "C":
                fname = f"{g}_ex_{served+1}_{lbl}__{sid}.png"
            else:
                fname = f"{g}_ex_{served+1}_{lbl}_{k_str}__{sid}.png"
            path = (base_dir / g / fname)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            return path

        for g in groups:
            if plot_examples: 
                print("--"*8, f" Teaching Set {g} ", "--"*8)
            it = iters[g]
            served = 0
            while True:
                if n is not None and served >= int(n):
                    break
                try:
                    session_metadata = next(it)
                except StopIteration:
                    break
                fig = session_metadata.pop("fig", None)
                out_meta[g].append(session_metadata)
                if show_meta:
                    print(session_metadata)
                if isinstance(fig, Figure):
                    _ = _save_if_needed(fig, g, session_metadata)
                    if plot_examples:
                        print(f"Classification label: {session_metadata['label']}, k: {session_metadata['k']}, session ID: {session_metadata['session_id']}")
                        display(fig)
                    plt.close(fig)
                    out_figs[g].append(fig)
                served += 1

        self.last_figs = out_figs
        self.last_meta = out_meta
        return out_figs

    def save(self, *, output_dir: str | Path | None = None) -> None:
        """Writes `selection.parquet` and `selection_config.json` beside the pool files."""
        if self.teaching_set_df is None:
            raise ValueError("selected_df is None.")
        root = Path(output_dir) if output_dir is not None else self.pool.paths.get("root", Path("."))
        root.mkdir(parents=True, exist_ok=True)
        self.teaching_set_df.to_parquet(root / "selection.parquet", index=False)
        with open(root / "selection_config.json", "w") as f:
            json.dump(self.meta, f, indent=2)

    def describe(self) -> None:
        """Prints selection counts and facility-location coverage by class."""
        if self.teaching_set_df is None:
            print("teaching set: empty")
            return
        df = self.teaching_set_df
        n = len(df)
        by_class = df.groupby("class_label")["session_id"].nunique().rename("n_sessions")
        print(f"[teaching set] rows={n}, classes:\n{by_class.to_string()}")
        if "coverage_per_class" in self.meta:
            print("\n[coverage] facility-location by class:")
            for k, v in self.meta["coverage_per_class"].items():
                print(f"  class {k}: {v:.4f}")
        if "per_bin_selected" in self.meta:
            print("\n[per-bin selected] counts by class:")
            for c in ("0", "1"):
                if c in self.meta["per_bin_selected"]:
                    print(f"  class {c}: {self.meta['per_bin_selected'][c]}")


# ------------------ teaching session, e.g. serving examples from a teaching set -----------------------

@dataclass
class TeachIterator:
    """Iterator over canonical ChargingSession objects for an MT4XAI user group.

    The iterator constructs sessions from file-backed arrays produced by the ORS pool
    (raw power .npy + knots .npy or, for legacy assets, a dense simplification),
    plots immediately, and returns a compact metadata dict for logging.

    Group semantics:
      - 'A': raw power with simplification overlay, ordered
      - 'B': raw power with simplification overlay, random order
      - 'C': raw power only, random order
    """
    df: pd.DataFrame
    group: Literal["A", "B", "C"]
    parent: TeachingSet
    random_seed: Optional[int]= None
    y_lim: tuple[float, float] | None = None
    sort_key: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    _curriculum: Optional[np.ndarray] = None
    _cursor: int = 0

    def __post_init__(self):
        df = self.df.reset_index(drop=True)
        if "class_label" not in df.columns:
            raise ValueError("class_label missing from DataFrame columns")
        
        # split by class
        normals_arr = df.index[df["class_label"].astype(int) == 0].to_numpy()
        abnormals_arr = df.index[df["class_label"].astype(int) == 1].to_numpy()
        
        def order_within_class(idxs: np.ndarray) -> np.ndarray:
            """Ensures the correct drawing order, in case the TeachingSet's 
            DataFrame has been altered between construction and serving.
            """
            if idxs.size == 0:
                return idxs
            class_subset = df.loc[idxs]
            if self.group == "A":
                # normals (0): k & margin asc. abnormals (1): k & margin desc.
                if not {"k", "margin"}.issubset(class_subset.columns):
                    raise ValueError("k or margin missing from df.columns")
                asc_k = int(class_subset["class_label"].iloc[0]) == 0
                ordered_df = class_subset.sort_values(["k", "margin"], ascending=[asc_k, asc_k])
            else:
                ordered_df = class_subset.sample(frac=1.0, random_state=self.random_seed)
            
            # cap after ordering for curriculum truncation (we cut the most challenging examples)
            if getattr(self, "max_per_class", None) is not None:
                ordered_df = ordered_df.head(self.max_per_class)
            
            return ordered_df.index.to_numpy()

        normals_curriculum = order_within_class(normals_arr)
        abnormals_curriculum = order_within_class(abnormals_arr)

        # interleave: start with normals (0) and alternate. 
        merged_curriculum: list[int] = []
        i0 = i1 = 0
        while i0 < len(normals_curriculum) or i1 < len(abnormals_curriculum):
            if i0 < len(normals_curriculum):
                merged_curriculum.append(int(normals_curriculum[i0])); i0 += 1
            if i1 < len(abnormals_curriculum): # if one side runs out, append the rest
                merged_curriculum.append(int(abnormals_curriculum[i1])); i1 += 1

        self._curriculum = np.asarray(merged_curriculum, dtype=int)
        self._cursor = 0

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> dict:
        if self._cursor >= len(self._curriculum):
            raise StopIteration
        i = int(self._curriculum[self._cursor])
        self._cursor += 1
        row = self.df.iloc[i]
        meta = self._serve_row(row)
        return meta

    # build session, plot immediately, and return a small log dict
    def _serve_row(self, row: pd.Series) -> dict:
        sid = int(row.get("session_id", row.get("charging_id", -1)))
        # paths
        sts_full_path = row.get("sts_full_path", None)
        piv_path = row.get("piv_path", None)
        raw_power_path = row.get("raw_power_path", None) or ( _derive_raw_power_path(sts_full_path) if sts_full_path else None )
        if raw_power_path is None:
            raise ValueError("cannot locate raw_power_path for session")

        power = _load_power(raw_power_path)
        T = int(power.shape[0])

       # overlay simplification if we have it
        simp = None
        if self.group in ("A", "B"):
            if piv_path is not None and Path(piv_path).exists():
                idx, val = _load_knots(piv_path, base_series=power)
                simp = ChargingSessionSimplification(
                    power_knot_idx=idx, power_knot_val_kw=val, k_power=int(idx.size - 1), kind="ors"
                )
            elif sts_full_path is not None and Path(sts_full_path).exists():
                # fallback for dense curves; estimate indices, then take values from dense
                dense = _safe_load(sts_full_path).astype(float)
                idx, _ = _dense_to_knots(dense)
                idx, val = _align_knots(idx, dense[idx], T)
                simp = ChargingSessionSimplification(
                    power_knot_idx=idx, power_knot_val_kw=val, k_power=int(idx.size - 1), kind="ors"
                )
        sess = ChargingSession(session_id=sid, power_kw=power, simplification=simp)
        # immediate plot with group-specific SOC rules
        if "label" in row and pd.notna(row["label"]):
            label_str = str(row["label"])
        elif "class_label" in row and pd.notna(row["class_label"]):
            label_str = "normal" if int(row["class_label"]) == 0 else "abnormal"
        else:
            label_str = "unknown"

        # derive file paths
        sts_full_path = row.get("sts_full_path", None)
        piv_path = row.get("piv_path", None)
        raw_power_path = row.get("raw_power_path", None) or str(_derive_raw_power_path(sts_full_path)) if sts_full_path else None
        raw_soc_path = row.get("raw_soc_path", None) or (str(_derive_raw_soc_path(sts_full_path)) if sts_full_path else None)
        piv_soc_path = row.get("piv_soc_path", None)
        sts_soc_path = row.get("sts_soc_path", None)

        if self.group == "C":
            # C: raw power + raw SOC only
            power = _load_power(raw_power_path) if raw_power_path else _load_power(_derive_raw_power_path(sts_full_path))
            soc_raw = _safe_load(raw_soc_path) if raw_soc_path else _safe_load(_derive_raw_soc_path(sts_full_path))
            sess_c = ChargingSession(session_id=sid, power_kw=power, soc_pct=np.asarray(soc_raw, dtype=float))
            fig, _ = sess_c.plot(soc_mode="raw", title=None, y_lim=self.y_lim)
        else:
            # A/B: raw+simpl power + simplified SOC only
            power = _load_power(raw_power_path) if raw_power_path else _load_power(_derive_raw_power_path(sts_full_path))
            idx, val = _load_knots(piv_path, base_series=power)
            simp = ChargingSessionSimplification(
                power_knot_idx=idx, power_knot_val_kw=val, k_power=int(idx.size - 1), kind="ors"
            )
            # attach SOC simpl if present, else fallback to raw SOC densification via dense->knots
            sidx, sval = None, None

            # load a base soc series to derive values from if pivots are indices-only
            soc_base: np.ndarray | None = None
            if raw_soc_path is not None:
                try:
                    soc_base = np.asarray(_safe_load(raw_soc_path), dtype=float).reshape(-1)
                except Exception:
                    soc_base = None
            if soc_base is None and sts_soc_path is not None:
                # fallback to dense simplified soc as base
                try:
                    soc_base = np.asarray(_safe_load(sts_soc_path), dtype=float).reshape(-1)
                except Exception:
                    soc_base = None

            if piv_soc_path is not None:
                # passes the soc series so 1D index pivots can be mapped to values
                sidx, sval = _load_knots(piv_soc_path, base_series=soc_base)
            elif sts_soc_path is not None:
                # erives pivots and values from dense soc if pivots missing
                sidx, sval = _dense_to_knots(_safe_load(sts_soc_path))

            if sidx is not None and sval is not None:
                simp.soc_knot_idx = np.asarray(sidx, dtype=int)
                simp.soc_knot_val_pct = np.asarray(sval, dtype=float)

            sess = ChargingSession(session_id=sid, power_kw=power, simplification=simp)
            # subtext = f"example {self._cursor} has k={simp.k_power} segments" if (simp and simp.k_power is not None) else None
            fig, _ = sess.plot(soc_mode="simpl", title=None, y_lim=self.y_lim, 
                               anchor_endpoints=self.parent.ors_params.anchor_endpoints, 
                               t_min_eval=int(self.parent.ors_params.t_min_eval))

        result = {
            "session_id": sid,
            "group": self.group,
            "k": (int(simp.k_power) if simp and simp.k_power is not None else None),
            "label": label_str,
            "sts_full_path": row.get("sts_full_path", None),
            "sts_soc_path": sts_soc_path,
            "piv_path": row.get("piv_path", None),
            "piv_soc_path": piv_soc_path,
            "raw_power_path": str(raw_power_path),
            "raw_soc_path": raw_soc_path,
            "fig": fig,
        }
        return result


# ------------------- helpers & utils: database (CRUD), schema, embeddings ------------ #

def _compute_embedding(sts: np.ndarray, *, L: int, P: int) -> np.ndarray:
    """Builds a scale-robust, spike-aware embedding (length 2L + 2P)."""
    y = np.asarray(sts, dtype=float)
    T = y.size
    if T < 2:
        y = np.array([y[0], y[0]], dtype=float)
        T = 2

    xs = np.linspace(0.0, 1.0, num=T, dtype=float)
    grid = np.linspace(0.0, 1.0, num=L, dtype=float)
    y_rs = np.interp(grid, xs, y)

    mu = float(np.mean(y_rs))
    sd = float(np.std(y_rs)) or 1.0
    y_z = (y_rs - mu) / sd
    dy = np.diff(y_z, prepend=y_z[0])

    peaks, _ = find_peaks(y_z)
    if peaks.size > 0:
        prom = peak_prominences(y_z, peaks)[0]
        wid = peak_widths(y_z, peaks, rel_height=0.5)[0] / float(L)
        order = np.argsort(prom)[::-1][:P]
        prom_top = prom[order]
        wid_top = wid[order]
        if prom_top.size < P:
            prom_top = np.pad(prom_top, (0, P - prom_top.size))
            wid_top = np.pad(wid_top, (0, P - wid_top.size))
    else:
        prom_top = np.zeros(P, dtype=float)
        wid_top = np.zeros(P, dtype=float)

    sig = np.column_stack([prom_top, wid_top]).ravel()
    z = np.concatenate([y_z.astype(np.float32), dy.astype(np.float32), sig.astype(np.float32)], axis=0)
    return z.astype(np.float32)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces common columns to expected dtypes."""
    d = df.copy()
    if "session_id" in d.columns:
        d["session_id"] = pd.to_numeric(d["session_id"], errors="coerce").astype("Int64")
    if "label_int" in d.columns:
        d["label_int"] = pd.to_numeric(d["label_int"], errors="coerce").astype("Int64")
    if "class_label" in d.columns:
        d["class_label"] = pd.to_numeric(d["class_label"], errors="coerce").astype("Int64")
    if "k" in d.columns:
        d["k"] = pd.to_numeric(d["k"], errors="coerce")
    return d


def _normalise_embedding_column(df: pd.DataFrame, expected_dim: int) -> pd.DataFrame:
    """Ensures 'emb' exists as list[float] of expected length; repairs from alternatives if needed."""
    d = df.copy()
    if "emb" not in d.columns:
        for cand in ["embedding", "embeddings", "vec"]:
            if cand in d.columns:
                d = d.rename(columns={cand: "emb"})
                break
    if "emb" in d.columns:
        def _fix(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            if isinstance(x, list):
                return x if len(x) == expected_dim else None
            if isinstance(x, np.ndarray):
                return x.astype(float).tolist() if x.size == expected_dim else None
            return None
        d["emb"] = d["emb"].map(_fix)
    return d


def _derive_raw_power_path(sts_full_path: str | Path) -> Path:
    """Maps a saved simplification path to its mirrored raw-power path."""
    p = Path(sts_full_path)
    try:
        if "sts_full" in p.parts:
            idx = p.parts.index("sts_full")
            return Path(*p.parts[:idx], "raw_power", *p.parts[idx + 1 :])
    except Exception:
        pass
    return p

def _derive_raw_soc_path(sts_full_path: str | Path) -> Path:
    """heuristic to derive raw SOC file path from a row, mirroring pool export layout."""
    p = Path(sts_full_path)
    return p.parent.parent / "raw_soc" / p.name


def _plot_session_overlay(ax, sid: int, row: pd.Series) -> None:
    """Plots raw vs simplified overlays for a session on a given axis."""
    raw_p = _derive_raw_power_path(row["sts_full_path"])
    sts_p = Path(row["sts_full_path"])
    try:
        raw = np.load(raw_p)
    except Exception:
        raw = None
    try:
        sts = np.load(sts_p)
    except Exception:
        sts = None

    if raw is not None:
        ax.plot(raw, linewidth=1.0, label="raw")
    if sts is not None:
        ax.plot(sts, linewidth=1.5, label="simplified")
    r = row.get("robust_prob", None)
    rtxt = f"{float(r):.2f}" if r is not None and not pd.isna(r) else "NA"
    ax.set_title(f"sid={sid} k={row.get('k')} r={rtxt}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def _pick_edge_examples(kept: pd.DataFrame, cls: int, edges: list[int]) -> list[int]:
    """Picks up to two sessions at each internal edge (left/right bins) using k_bin_idx."""
    picks: list[int] = []
    if not edges or len(edges) < 2:
        return picks

    sub = kept.loc[kept["base_label"] == cls, ["session_id", "k", "k_bin_idx"]].dropna()
    if sub.empty:
        return picks

    for j in range(len(edges) - 1 - 1):
        left_idx, right_idx = j, j + 1
        boundary = edges[j + 1]

        left_rows = sub.loc[sub["k_bin_idx"] == left_idx]
        right_rows = sub.loc[sub["k_bin_idx"] == right_idx]

        if not left_rows.empty:
            sid_left = int(left_rows.iloc[(left_rows["k"] - boundary).abs().argsort().iloc[0]]["session_id"])
            if sid_left not in picks:
                picks.append(sid_left)
        if not right_rows.empty:
            sid_right = int(right_rows.iloc[(right_rows["k"] - boundary).abs().argsort().iloc[0]]["session_id"])
            if sid_right not in picks:
                picks.append(sid_right)

    return picks


def _create_main_table(conn: sqlite3.Connection) -> None:
    """Ensures the `ors_pool` table and indexes exist."""
    conn.execute(
    """
    CREATE TABLE IF NOT EXISTS ors_pool (
        session_id INTEGER PRIMARY KEY,
        label_text TEXT,
        label_int INTEGER,
        k REAL,
        err REAL,
        frag REAL,
        robust_prob REAL,
        margin REAL,
        threshold REAL,
        model_id TEXT,
        ts_unix REAL,
        sts_full_path TEXT,
        piv_path TEXT,
        emb_dim INTEGER,
        emb BLOB,
        raw_power_path TEXT,
        raw_soc_path   TEXT,
        piv_soc_path   TEXT,
        sts_soc_path   TEXT
    );
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ors_pool_label ON ors_pool(label_int);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ors_pool_k ON ors_pool(k);")
    conn.commit()


def _upsert_row(conn: sqlite3.Connection, row: dict) -> None:
    """Inserts or updates a row in `ors_pool` by session_id."""
    conn.execute(
        """
    INSERT INTO ors_pool (session_id, label_text, label_int, k, err, frag, robust_prob, margin, threshold,
                          model_id, ts_unix, sts_full_path, piv_path, emb_dim, emb,
                          raw_power_path, raw_soc_path, piv_soc_path, sts_soc_path)
    VALUES (:session_id, :label_text, :label_int, :k, :err, :frag, :robust_prob, :margin, :threshold,
            :model_id, :ts_unix, :sts_full_path, :piv_path, :emb_dim, :emb,
            :raw_power_path, :raw_soc_path, :piv_soc_path, :sts_soc_path)
    ON CONFLICT(session_id) DO UPDATE SET
        label_text=excluded.label_text,
        label_int=excluded.label_int,
        k=excluded.k,
        err=excluded.err,
        frag=excluded.frag,
        robust_prob=excluded.robust_prob,
        margin=excluded.margin,
        threshold=excluded.threshold,
        model_id=excluded.model_id,
        ts_unix=excluded.ts_unix,
        sts_full_path=excluded.sts_full_path,
        piv_path=excluded.piv_path,
        emb_dim=excluded.emb_dim,
        emb=excluded.emb,
        raw_power_path=excluded.raw_power_path,
        raw_soc_path=excluded.raw_soc_path,
        piv_soc_path=excluded.piv_soc_path,
        sts_soc_path=excluded.sts_soc_path
    """,
        row,
    )

    conn.commit()


def _rows_to_parquet(db_path: Path, out_path: Path) -> None:
    """Exports the SQLite cache to Parquet, decoding embeddings if needed."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=60.0) as rconn:
        df = pd.read_sql_query("SELECT * FROM ors_pool", rconn)

    def parse_emb(x):
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            try:
                return np.frombuffer(x, dtype=np.float32).tolist()
            except Exception:
                try:
                    return json.loads(x.decode("utf-8"))
                except Exception:
                    return None
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return None
        return x

    if "emb" in df.columns:
        df["emb"] = df["emb"].map(parse_emb)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)


def _save_json(obj: dict, path: Path) -> None:
    """Writes a JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _prepare_ors_params_for_T(params: ORSParams, T: int) -> ORSParams:
    """Clamps ORS hyperparameters to a feasible range for a sequence of length T."""
    p = copy.copy(params)
    T_safe = max(int(T), 2)
    p.dp_q = int(max(8, min(p.dp_q, T_safe - 1)))         # q must be < T
    p.max_k = int(max(1, min(p.max_k, T_safe - 1)))       # max_k < T
    p.min_k = int(max(1, min(p.min_k, p.max_k)))          # 1 <= min_k <= max_k
    return p


def _validate_ors_result(res: dict) -> tuple[bool, str]:
    """Validates that an ORS result carries required keys with finite values."""
    if not isinstance(res, dict):
        return False, "res_not_dict"
    required = ("frag", "label", "k", "err", "sts", "piv")
    missing = [k for k in required if k not in res]
    if missing:
        return False, f"missing_keys:{missing}"
    try:
        if res["frag"] is None or not np.isfinite(float(res["frag"])):
            return False, "frag_invalid"
        if res["k"] is None or not np.isfinite(float(res["k"])) or float(res["k"]) < 1:
            return False, "k_invalid"
        if res["err"] is None or not np.isfinite(float(res["err"])):
            return False, "err_invalid"
    except Exception:
        return False, "type_cast_error"
    if res["sts"] is None or res["piv"] is None:
        return False, "arrays_missing"
    return True, "ok"


def _bin_midpoint(bin_label: str) -> float:
    """Parses a bin label like '[a, b]' and returns its midpoint."""
    s = str(bin_label).strip().replace("(", "[").replace(")", "]")
    try:
        parts = s.strip("[]").split(",")
        a, b = float(parts[0]), float(parts[1])
        return 0.5 * (a + b)
    except Exception:
        return math.inf


def _export_config(cfg: TeachingPoolConfig, out_dir: Path, n_abn: int, n_norm: int) -> None:
    """Writes a compact provenance JSON for the pool build."""
    meta = {
        "model_id": Path(cfg.model_path).name,
        "threshold": cfg.threshold,
        "random_seed": cfg.random_seed,
        "abnormal_count": n_abn,
        "normal_count": n_norm,
        "timestamp": int(time.time()),
    }
    _save_json(meta, out_dir / "config.json")



def _safe_load(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing file: {p}")
    return np.load(p)

def _load_power(path: str | Path) -> np.ndarray:
    y = _safe_load(path)
    y = np.asarray(y, dtype=float).reshape(-1)
    return y

def _load_knots(
    path: str | Path,
    base_series: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads knots from a .npy file and returns (indices, values).
    The file may contain:
      - 1D int array (indices only)
      - 2D array shaped (K, 2) or (2, K) with [idx, val]
    If indices only are stored, values are derived by sampling from `base_series`.
    Args:
        path: Path to .npy file with knots.
        base_series: Optional dense series to derive values from
    Raises: ValueError if indices-only are loaded and no base series is available.
    Returns:
        idx: 1D int array of pivot indices.
        vals: 1D float array of pivot values aligned with idx.
    """
    a = _safe_load(path)
    a = np.asarray(a)
    if a.ndim == 1:
        idx = a.astype(int).reshape(-1)
        if base_series is None:
            # keep helpful error for old call sites
            raise ValueError("1D pivots require a base series to derive knot values.")
        vals = np.asarray(base_series, dtype=float)[idx]
        return idx, vals
    if a.ndim == 2:
        if a.shape[1] == 2:
            idx = a[:, 0].astype(int).reshape(-1)
            vals = a[:, 1].astype(float).reshape(-1)
            return idx, vals
        if a.shape[0] == 2:
            idx = a[0, :].astype(int).reshape(-1)
            vals = a[1, :].astype(float).reshape(-1)
            return idx, vals
    raise ValueError(f"unexpected knots array shape {a.shape} in {path}")


def _dense_to_knots(y: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct knots from a dense piecewise-linear series (best-effort).
    assumes y is already an ORS-like densification (linear between true knots).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < 2:
        return np.array([0, max(y.size - 1, 0)], dtype=int), y[[0, -1]] if y.size else np.array([0.0, 0.0])

    dy = np.diff(y)
    # change where slope changes beyond tolerance
    change = np.where(np.abs(np.diff(dy)) > tol)[0] + 1
    idx = np.concatenate(([0], change, [y.size - 1]))
    idx = np.unique(idx)
    val = y[idx]
    return idx.astype(int), val.astype(float)

def _align_knots(idx: np.ndarray, val: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """clip to [0, T-1], enforce increasing order, ensure endpoints present."""
    idx = np.asarray(idx, dtype=int)
    val = np.asarray(val, dtype=float)
    # clip and unique-increasing
    idx = np.clip(idx, 0, max(T - 1, 0))
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    val = val[order]
    # deduplicate identical indices, keep the last value
    keep = np.concatenate(([True], np.diff(idx) > 0))
    idx = idx[keep]; val = val[keep]
    # ensure endpoints
    if idx.size == 0 or idx[0] != 0:
        idx = np.insert(idx, 0, 0)
        val = np.insert(val, 0, val[0] if val.size else 0.0)
    if idx[-1] != T - 1:
        idx = np.append(idx, T - 1)
        val = np.append(val, val[-1])
    return idx, val

def _derive_raw_power_path(sts_full_path: str | Path) -> Path:
    """heuristic to derive raw power file path from a row, if not given explicitly."""
    p = Path(sts_full_path)
    return p.parent.parent / "raw_power" / p.name  # mirrors the directory naming in pool export




# --------------------- analytics, plotting ----------------- #

def _compute_global_power_max(dfs: list[pd.DataFrame]) -> float:
    vmax = 0.0
    for df in dfs:
        for _, r in df.iterrows():
            rp = r.get("raw_power_path", None)
            if rp is None and r.get("sts_full_path", None) is not None:
                rp = _derive_raw_power_path(r["sts_full_path"])
            if rp is None:
                continue
            try:
                y = _load_power(rp)
                if y.size:
                    vmax = max(vmax, float(np.nanmax(y)))
            except FileNotFoundError:
                continue
    return vmax if np.isfinite(vmax) else 0.0


def _compute_class_coverage(
    df_all: pd.DataFrame, df_sel: pd.DataFrame, normalise: bool, sim_clip_min: float | None
) -> dict[str, float]:
    """Computes facility-location coverage per class on the union of items in that class."""
    res: dict[str, float] = {"0": 0.0, "1": 0.0}
    for c in (0, 1):
        U = df_all[df_all["class_label"] == c]
        S = df_sel[df_sel["class_label"] == c]
        if U.empty or S.empty:
            res[str(c)] = 0.0
            continue
        EU = np.asarray(U["emb"].to_list(), dtype=np.float32)
        ES = np.asarray(S["emb"].to_list(), dtype=np.float32)
        if normalise:
            nu = np.linalg.norm(EU, axis=1, keepdims=True)
            nu[nu == 0.0] = 1.0
            EU = EU / nu
            ns = np.linalg.norm(ES, axis=1, keepdims=True)
            ns[ns == 0.0] = 1.0
            ES = ES / ns
        M = EU @ ES.T
        if sim_clip_min is not None and sim_clip_min > -1.0:
            M = np.clip(M, sim_clip_min, 1.0)
        g = M.max(axis=1)
        res[str(c)] = float(np.maximum(0.0, g).sum()) if sim_clip_min is not None and sim_clip_min >= 0.0 else float(g.sum())
    return res


def _compute_avg_pairwise_bin(
    df_all: pd.DataFrame, df_sel: pd.DataFrame, normalise: bool, sim_clip_min: float | None
) -> dict[str, dict[str, float]]:
    """Computes average pairwise cosine within each selected (class, bin). Lower is better diversity."""
    out: dict[str, dict[str, float]] = {"0": {}, "1": {}}
    for c in (0, 1):
        S_c = df_sel[df_sel["class_label"] == c]
        if S_c.empty:
            continue
        for b in sorted(S_c["k_bin"].astype(str).unique().tolist()):
            S_cb = S_c[S_c["k_bin"].astype(str) == b]
            if len(S_cb) < 2:
                out[str(c)][str(b)] = 0.0
                continue
            ES = np.asarray(S_cb["emb"].to_list(), dtype=np.float32)
            if normalise:
                norms = np.linalg.norm(ES, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                ES = ES / norms
            M = ES @ ES.T
            if sim_clip_min is not None and sim_clip_min > -1.0:
                M = np.clip(M, sim_clip_min, 1.0)
            n = M.shape[0]
            tri = np.triu_indices(n, k=1)
            avg = float(M[tri].mean()) if tri[0].size > 0 else 0.0
            out[str(c)][str(b)] = avg
    return out


def selection_vs_pool_report(binned_df: pd.DataFrame, selected_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Builds comparison tables of pool vs selection frequencies per class for k and k-bins.

    Computes:
      - by_k: counts, shares, selection_rate and representation lift per (class, k).
      - by_k_bin: same, but per (class, k_bin_label).
      - extremes_summary: quick view of head/tail coverage per class (min/max k and rarest k).

    Returns:
      Dict of DataFrames: {"by_k": df, "by_k_bin": df, "extremes_summary": df}.
    """
    pool = _verify_columns(binned_df)
    sel = _verify_columns(selected_df)

    req_pool = {"class_label", "k", "k_bin_label", "session_id"}
    req_sel = {"class_label", "k", "k_bin_label", "session_id"}
    missing_pool = req_pool - set(pool.columns)
    missing_sel = req_sel - set(sel.columns)
    if missing_pool:
        raise ValueError(f"binned_df is missing columns: {missing_pool}")
    if missing_sel:
        raise ValueError(f"selected_df is missing columns: {missing_sel}")

    pool_k = _counts_shares(pool, ["class_label", "k"]).rename(columns={"n": "n_pool", "share_in_class": "share_in_class_pool"})
    sel_k = _counts_shares(sel, ["class_label", "k"]).rename(columns={"n": "n_sel", "share_in_class": "share_in_class_sel"})
    by_k = pool_k.merge(sel_k, how="left", on=["class_label", "k"]).fillna({"n_sel": 0, "share_in_class_sel": 0.0})
    by_k["selection_rate"] = by_k["n_sel"] / by_k["n_pool"].replace(0, np.nan)
    by_k["lift"] = by_k["share_in_class_sel"] / by_k["share_in_class_pool"].replace(0, np.nan)
    by_k = by_k.sort_values(["class_label", "k"])

    pool_b = _counts_shares(pool, ["class_label", "k_bin_label"]).rename(columns={"n": "n_pool", "share_in_class": "share_in_class_pool"})
    sel_b = _counts_shares(sel, ["class_label", "k_bin_label"]).rename(columns={"n": "n_sel", "share_in_class": "share_in_class_sel"})
    by_bin = pool_b.merge(sel_b, how="left", on=["class_label", "k_bin_label"]).fillna({"n_sel": 0, "share_in_class_sel": 0.0})
    by_bin["selection_rate"] = by_bin["n_sel"] / by_bin["n_pool"].replace(0, np.nan)
    by_bin["lift"] = by_bin["share_in_class_sel"] / by_bin["share_in_class_pool"].replace(0, np.nan)
    by_bin = by_bin.sort_values(["class_label", "k_bin_label"])

    extremes = []
    for c in sorted(pool["class_label"].unique().tolist()):
        sub_k = by_k[by_k["class_label"] == c]
        if not sub_k.empty:
            extremes.append(
                {
                    "class_label": c,
                    "min_k": float(sub_k["k"].min()),
                    "max_k": float(sub_k["k"].max()),
                    "rare_k": float(sub_k.loc[sub_k["n_pool"].idxmin(), "k"]),
                }
            )
    extremes_df = pd.DataFrame(extremes) if extremes else pd.DataFrame(columns=["class_label", "min_k", "max_k", "rare_k"])

    return {"by_k": by_k, "by_k_bin": by_bin, "extremes_summary": extremes_df}


def _counts_shares(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Counts items per key and adds within-class share."""
    g = df.groupby(keys, dropna=False)["session_id"].count().reset_index(name="n")
    cls_tot = g.groupby(["class_label"], dropna=False)["n"].transform("sum")
    g["share_in_class"] = g["n"] / cls_tot.replace(0, np.nan)
    return g


def _verify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalises expected column names for comparison helpers."""
    d = df.copy()
    if "class_label" not in d.columns and "label_int" in d.columns:
        d = d.rename(columns={"label_int": "class_label"})
    if "k_bin_label" not in d.columns and "k_bin" in d.columns:
        d = d.rename(columns={"k_bin": "k_bin_label"})
    return d
