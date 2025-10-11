"""
mt4xai.test (minimal)
---------------------
Single-call MT4XAI user test:
1) select a session,
2) compute base label/error,
3) run ORS (DP + prefix sums),
4) plot original vs simplification,
5) return a compact result object.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any
import matplotlib.pyplot as plt

from .data import fetch_session_preds_bundle
from .plot import plot_session_with_simplification
from .ors import ORSParams, ors, base_label_from_bundle  # uses your existing helpers


@dataclass
class MT4XAIUserTestResult:
    """stores outcome for one MT4XAI test case."""
    session_id: Optional[int]
    base_label: str
    base_error: float
    simplified_label: str
    simplified_error: float
    k: int
    robust_prob: float


def run_mt4xai_user_test(
    model: Any, loader: Any, *,
    power_scaler: Any, soc_scaler: Any,
    idx_power_inp: int, idx_soc_inp: int,
    power_weight: float, decay_lambda: float,
    threshold: float,
    t_min_eval: int = 1,
    session_id: Optional[int] = None,
    batch_index: Optional[int] = None,
    sample_index: Optional[int] = None,
    figsize: Tuple[float, float] = (12.0, 5.0)
) -> MT4XAIUserTestResult:
    """
    runs a single mt4xai user test end-to-end using dp+prefix sums for stage-1.
    requires an absolute Macro-RMSE threshold.
    """

    # selection: either session_id or (batch_index, sample_index)
    if (session_id is None) == (batch_index is None or sample_index is None):
        raise ValueError("provide either session_id OR (batch_index, sample_index).")

    device = next(model.parameters()).device
    bundle = fetch_session_preds_bundle(
        model, loader,
        batch_index=batch_index, sample_index=sample_index, device=device,
        power_scaler=power_scaler, soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
        session_id=session_id
    )

    # base label/error in original units using stored predictions
    base_lbl_int, base_err, Y_abs_true = base_label_from_bundle(
        bundle,
        power_scaler=power_scaler, soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
        power_weight=power_weight, decay_lambda=decay_lambda,
        t_min_eval=t_min_eval, threshold=float(threshold)
    )
    base_lbl = "ABNORMAL" if base_lbl_int == 1 else "NORMAL"

    # ors with dp + prefix sums (your fastest exact stage-1)
    params = ORSParams(
        stage1_mode="dp_prefix",
        stage2_err_metric="l2",
        q=150, stage1_candidates=20,
        alpha=0.001, beta=3.0, gamma=0.05,
        R=1200, epsilon_mode="fraction", epsilon_value=0.3,
        t_min_eval=t_min_eval, min_k=1, max_k=15, seed=1337
    )

    res = ors(
        bundle, model, params,
        power_scaler=power_scaler, soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
        power_weight=power_weight, decay_lambda=decay_lambda,
        threshold=float(threshold)
    )

    k = int(res["k"])
    simp_err = float(res["err"])
    simp_lbl = "ABNORMAL" if res["label"] == "abnormal" else "NORMAL"
    robust_prob = 1.0 - float(res["frag"])

    # plot using your existing helper; it expects the simplified series explicitly
    plt.figure(figsize=figsize, dpi=110); plt.close()
    plot_session_with_simplification(
        bundle, power_scaler, soc_scaler, idx_power_inp, idx_soc_inp,
        simpl_power_unscaled=res["sts"],
        session_id=getattr(bundle, "session_id", None),
        k=k, threshold=float(threshold),
        simp_error=simp_err, orig_error=base_err,
        label=f"{base_lbl}, simp={simp_lbl}",
        figsize=figsize, dpi=110
    )

    return MT4XAIUserTestResult(
        session_id=getattr(bundle, "session_id", None),
        base_label=base_lbl, base_error=base_err,
        simplified_label=simp_lbl, simplified_error=simp_err,
        k=k, robust_prob=robust_prob
    )
