# src/mt4xai/ad.py
from typing import List
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from .data import SampleBundle, reconstruct_abs_from_bundle
from .inference import predict_residuals, reconstruct_abs_from_residuals_batch, inverse_targets_np, macro_rmse_per_session

def compute_session_errors(model,
                           loader,
                           device: torch.device,
                           input_features: List[str],
                           target_features: List[str],
                           power_scaler: MinMaxScaler,
                           soc_scaler: MinMaxScaler,
                           power_weight: float,
                           idx_power_inp: int,
                           idx_soc_inp: int, 
                           t_min_eval: int=0) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: session_id, error, len, and optionally per-target RMSEs.
    Only predictions for t >= t_min_eval + 1 count towards the error metric.
    """
    rows = []
    for session_ids, Xb, Yb, lengths in loader:
        P_res = predict_residuals(model, Xb, lengths, device=device)
        # residuals -> absolute in scaled space
        P_abs_scaled = reconstruct_abs_from_residuals_batch(Xb, P_res, idx_power_inp, idx_soc_inp)
        # true absolute in scaled space: base + residual Y
        base = torch.stack([Xb[..., idx_power_inp], Xb[..., idx_soc_inp]], dim=-1)  # (B, T, 2)
        base = base.unsqueeze(2)  # (B, T, 1, 2) so it broadcasts over H
        Y_abs_scaled = base + Yb  # (B, T, H, 2)

        # inverse-transform only targets to original units
        P_abs_np = inverse_targets_np(P_abs_scaled.cpu().numpy(), power_scaler, soc_scaler)
        Y_abs_np = inverse_targets_np(Y_abs_scaled.cpu().numpy(), power_scaler, soc_scaler)

        # compute macro-RMSE per session
        errs = macro_rmse_per_session(torch.from_numpy(P_abs_np),
                                      torch.from_numpy(Y_abs_np),
                                      lengths,
                                      power_weight=power_weight, 
                                      t_min_eval=t_min_eval)

        for sid, e, L in zip(session_ids, errs, lengths.tolist()):
            rows.append({"charging_id": sid, "error": float(e), "length": int(L)})
    return pd.DataFrame(rows)


def compute_bundle_error(bundle: SampleBundle,
                         power_scaler, soc_scaler,
                         power_weight: float,
                         idx_power_inp: int, idx_soc_inp: int, 
                         t_min_eval: int=1) -> float:
    """
    Macro-averaged RMSE across horizons, mixing power/SOC with `power_weight`.
    Matches the AD pipeline semantics (compute_session_errors).
    """
    T, H = bundle.length, bundle.horizon

    # scaled → absolute (scaled)
    P_abs_scaled = reconstruct_abs_from_bundle(bundle, idx_power_inp, idx_soc_inp)  # (T,H,2)
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)  # (T,1,2)
    Y_abs_scaled = base + bundle.Y_sample  # (T,H,2)

    # inverse-transform to original units
    P_abs = inverse_targets_np(P_abs_scaled.numpy(), power_scaler, soc_scaler)  # (T,H,2)
    Y_abs = inverse_targets_np(Y_abs_scaled.numpy(), power_scaler, soc_scaler)  # (T,H,2)

    per_h = []
    for h in range(H):
        end = T - (h + 1)
        if end <= t_min_eval:
            continue
        diff = P_abs[t_min_eval:end, h, :] - Y_abs[t_min_eval:end, h, :]
        rmse_c = np.sqrt(np.mean(diff**2, axis=0))  # (2,)
        val = power_weight * rmse_c[0] + (1.0 - power_weight) * rmse_c[1]
        per_h.append(val)
    return float(np.mean(per_h)) if per_h else float("nan")


def percentile_threshold(errors: np.ndarray, pct_thr: float=95.0) -> float:
    """
    Empirical threshold = pct-th percentile of error distribution.
    """
    return round(float(np.nanpercentile(errors, pct_thr)), 4)

def percentile_of_threshold(all_errs_sorted, thr: float) -> float:
    """Return percentile rank of threshold in error distribution."""
    idx = np.searchsorted(all_errs_sorted, thr, side="right")
    return 100.0 * idx / max(1, len(all_errs_sorted))


def classify_by_threshold(df_errs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Adds `label` column: 'abnormal' if error > threshold else 'normal'.
    """
    df = df_errs.copy()
    df["label"] = np.where(df["error"] > threshold, "abnormal", "normal")
    return df.sort_values("error", ascending=True).reset_index(drop=True)
