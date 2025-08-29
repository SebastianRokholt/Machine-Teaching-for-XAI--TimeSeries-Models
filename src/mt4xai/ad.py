# src/mt4xai/ad.py
from __future__ import annotations  # postpones evaluation of type hints to speed up imports
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from .data import SampleBundle, reconstruct_abs_from_bundle, make_bundle_from_session
from .inference import predict_residuals, reconstruct_abs_from_residuals_batch, inverse_targets_np, macro_rmse_per_session
from typing import Iterable

_RWSE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_EPS: float = 1e-8


def list_unique_session_ids(df_scaled: pd.DataFrame) -> np.ndarray:
    """Returns sorted unique charging_id values as a plain int ndarray."""
    return np.asarray(np.sort(np.asarray(df_scaled["charging_id"].unique())), dtype=int)

def _bundle_iter(model,
                 df_scaled: pd.DataFrame,
                 sids: Iterable[int],
                 *,
                 device,
                 input_features: list[str],
                 target_features: list[str],
                 horizon: int,
                 power_scaler,
                 soc_scaler,
                 idx_power_inp: int,
                 idx_soc_inp: int,
                 t_min_eval: int):
    """Yields (sid, residuals[T,H,C], T) for each session id, residuals in original units cropped at t>=t_min_eval."""
    for sid in sids:
        b = make_bundle_from_session(
            model=model, df_scaled=df_scaled, sid=int(sid), device=device,
            input_features=input_features, target_features=target_features, horizon=horizon,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
            sort_by="minutes_elapsed"
        )
        # b.Y_sample and b.P_sample are residual targets/preds in *scaled* units.
        # Convert to residuals in original units by inverse-transforming absolute series.
        # Reuse inverse logic via your existing helpers:
        from .inference import inverse_targets_np  # local import to avoid cycles
        base = b.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1).numpy()          # (T,1,2) scaled
        Y_abs = base + b.Y_sample.numpy()                                                # (T,H,2) scaled
        P_abs = base + b.P_sample.numpy()                                                # (T,H,2) scaled
        Y = inverse_targets_np(Y_abs, power_scaler, soc_scaler)                          # (T,H,2)
        P = inverse_targets_np(P_abs, power_scaler, soc_scaler)                          # (T,H,2)
        R = (Y - P)                                                                      # (T,H,2)
        Tlen = R.shape[0]
        t0 = min(int(t_min_eval), max(0, Tlen - 1))
        yield int(sid), R[t0:], int(Tlen)

def fit_rwse_robust_scalers(model,
                            val_scaled_df: pd.DataFrame,
                            *,
                            device,
                            input_features: list[str],
                            target_features: list[str],
                            horizon: int,
                            power_scaler,
                            soc_scaler,
                            idx_power_inp: int,
                            idx_soc_inp: int,
                            t_min_eval: int = 1,
                            cache_key: str = "default") -> tuple[np.ndarray, np.ndarray]:
    """Fits per-(h,c) median and MAD of residuals on the validation set; returns (m, mad) with shape (H,C)."""
    key = ("robust_scalers", cache_key)
    if key in _RWSE_CACHE:
        return _RWSE_CACHE[key]

    sids = list_unique_session_ids(val_scaled_df)
    all_R: list[np.ndarray] = []
    for _, R, _ in _bundle_iter(model, val_scaled_df, sids,
                                device=device, input_features=input_features, target_features=target_features,
                                horizon=horizon, power_scaler=power_scaler, soc_scaler=soc_scaler,
                                idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp, t_min_eval=t_min_eval):
        if R.size:
            all_R.append(np.asarray(R, dtype=np.float64))

    if not all_R:
        raise RuntimeError("No residuals collected from validation set for RWSE scalers.")

    R_cat = np.asarray(np.concatenate(all_R, axis=0), dtype=np.float64)    # (sum_T, H, C)
    m   = np.median(R_cat, axis=0)                                         # (H,C)
    mad = np.median(np.abs(R_cat - m), axis=0).astype(np.float64) + _EPS   # (H,C)

    _RWSE_CACHE[key] = (m, mad)
    return m, mad
def rwse_score_from_bundle(
    bundle: SampleBundle,
    m: np.ndarray,
    mad: np.ndarray,
    *,
    power_scaler: MinMaxScaler,
    soc_scaler: MinMaxScaler,
    idx_power_inp: int,
    idx_soc_inp: int,
    w_h: np.ndarray,
    w_c: np.ndarray,
    cap: float = 5.0,
    t_min_eval: int = 1,
) -> tuple[float, int]:
    """Compute RWSE for one bundle using calibrated medians and MADs; returns (score, session_length)."""
    # deltas in scaled space
    Y = np.asarray(bundle.Y_sample, dtype=np.float64)  # (T,H,C)
    P = np.asarray(bundle.P_sample, dtype=np.float64)  # (T,H,C)
    # reconstruct absolute (scaled), then inverse to original units
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1).numpy()  # (T,1,2)
    Y_abs = inverse_targets_np(base + Y, power_scaler, soc_scaler)  # (T,H,2)
    P_abs = inverse_targets_np(base + P, power_scaler, soc_scaler)  # (T,H,2)

    T = int(Y.shape[0])
    t0 = min(int(t_min_eval), max(0, T - 1))
    R = (Y_abs - P_abs)[t0:]  # (T',H,2)
    if R.size == 0:
        return 0.0, T

    m   = np.asarray(m,   dtype=np.float64)   # (H,2)
    mad = np.asarray(mad, dtype=np.float64)   # (H,2)
    w_h = np.asarray(w_h, dtype=np.float64)   # (H,)
    w_c = np.asarray(w_c, dtype=np.float64)   # (2,)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z  = np.abs(R - m[None, :, :]) / mad[None, :, :]  # (T',H,2)
        Z  = np.clip(Z, a_min=None, a_max=cap)
        Zw = Z * w_h[None, :, None] * w_c[None, None, :]
        Zw = np.nan_to_num(Zw, nan=0.0, posinf=cap, neginf=0.0)

    score = Zw.sum(axis=(1, 2)).mean()
    return float(score), T


def compute_session_MRMSE(model, loader, device: torch.device, power_scaler: MinMaxScaler,
                           soc_scaler: MinMaxScaler, power_weight: float, idx_power_inp: int, 
                           idx_soc_inp: int,  t_min_eval: int=0) -> pd.DataFrame:
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


def compute_session_RWSE(model, df_scaled: pd.DataFrame, 
                         device, input_features: list[str], target_features: list[str], 
                         power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int, m: np.ndarray, 
                         mad: np.ndarray, horizon: int, horizon_weights_decay: float=0.4, 
                         cap: float = 5.0, t_min_eval: int = 1, 
                         session_ids: Iterable[int] | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns: charging_id, length, error (RWSE)."""
    rows: list[dict] = []

    if not session_ids:
        session_ids = list(list_unique_session_ids(df_scaled))
    else: 
        session_ids = list(map(int, session_ids))

    # Calculate horizon and feature weights
    w_h = make_horizon_weights(horizon, decay=horizon_weights_decay)
    w_c = make_feature_weights(target_features)

    for sid in session_ids:
        b = make_bundle_from_session(
            model=model, df_scaled=df_scaled, sid=int(sid), device=device,
            input_features=input_features, target_features=target_features, horizon=horizon,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp, sort_by="minutes_elapsed"
        )
        s, L = rwse_score_from_bundle(
            b, m, mad,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
            w_h=w_h, w_c=w_c, cap=cap, t_min_eval=t_min_eval,
        )
        rows.append({"charging_id": int(sid), "length": int(L), "error": float(s)})
    return pd.DataFrame(rows)


def make_horizon_weights(H: int, decay: float = 0.4) -> np.ndarray:
    """Returns length-H exponential horizon weights that sum to 1."""
    w = np.exp(-decay * np.arange(int(H), dtype=float))
    return (w / w.sum()).astype(float)

def make_feature_weights(target_features: list[str]) -> np.ndarray:
    """Returns uniform feature weights over target features that sum to 1."""
    C = int(len(target_features))
    return np.ones(C, dtype=float) / float(C)


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
