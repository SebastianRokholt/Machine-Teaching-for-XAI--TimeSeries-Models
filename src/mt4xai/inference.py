# src/mt4xai/inference.py
from __future__ import annotations
import math  # postpones evaluation of type hints to speed up imports
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from .data import SessionPredsBundle, ChargingSessionDataset, reconstruct_abs_from_bundle, make_bundle_from_session_df, session_collate_fn
from .model import horizon_weights

_RWSE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_EPS: float = 1e-8


@torch.no_grad()
def predict_residuals(model, X: torch.Tensor, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Runs the sequence model and returns residual predictions with shape (B, T, H, C).
    Forward signature is (x, seq_lengths) -> (preds, out_lengths).
    We run forward on GPU but always return CPU for downstream numpy/scaler ops.
    """
    model.eval()
    X = X.to(device)

    # pack_padded_sequence requires CPU int64 lengths
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, dtype=torch.long)
    else:
        lengths = lengths.to(dtype=torch.long, device="cpu")

    P_res, _ = model(X, lengths)
    return P_res.detach().cpu()


def evaluate_model(model: nn.Module,
    dataset: ChargingSessionDataset,
    batch_size: int,
    device: torch.device,
    power_scaler: MinMaxScaler,
    horizon: int,
    idx_power: int=2,
    weight_decay: Optional[np.ndarray]=None,
 ) -> Dict[str, float | int]:
    """computes Macro-RMSE (in kW) for a single-target (power) multi-horizon forecaster.
    
    it computes a per-session RMSE that aggregates per-horizon MSE with weights w_h, then
    averages those RMSE values across sessions. predictions are reconstructed to absolute
    power and inverse-scaled to kW before error computation.
    returns: {"MacroRMSE": float, "NumSequencesEvaluated": int}
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=session_collate_fn, shuffle=False)
    model.eval()
    h_weights = horizon_weights(H=horizon, alpha=weight_decay, device=torch.device("cpu"), as_vector=True)
    seq_rmses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            _, X_batch, Y_batch, lengths = batch
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            pred_resid, _ = model(X_batch, lengths)                      # (B, T, H, 1)
            P_abs = reconstruct_abs_from_residuals_batch(pred_resid, X_batch, idx_power)  # (B,T,H,1)
            Y_abs = Y_batch + X_batch[..., [idx_power]].unsqueeze(2)     # (B, T, H, 1)

            B, _, H, _ = P_abs.shape
            for i in range(B):
                L = int(lengths[i].item())
                if L <= H + 1:
                    continue
                s, e = 1, L - H  # valid [t] for multi-horizon comparison

                # inverse-scale to kW
                pv = P_abs[i, s:e, :, 0].contiguous().view(-1, 1).cpu().numpy()
                tv = Y_abs[i, s:e, :, 0].contiguous().view(-1, 1).cpu().numpy()
                p_kw = power_scaler.inverse_transform(pv).reshape(-1, H)
                t_kw = power_scaler.inverse_transform(tv).reshape(-1, H)

                # per-horizon MSE over valid t, then weighted average across horizons
                per_h_mse = np.array([
                    mean_squared_error(t_kw[:, h], p_kw[:, h]) for h in range(H)
                ], dtype=np.float64)
                weighted_mse = float(np.dot(h_weights, per_h_mse))
                seq_rmses.append(np.sqrt(weighted_mse))
    return {
        "MacroRMSE": float(np.mean(seq_rmses)) if seq_rmses else float("nan"),
        "NumSequencesEvaluated": int(len(seq_rmses)),
    }


def reconstruct_abs_from_residuals_batch(
    P_res: Tensor, X: Tensor, idx_power: int, idx_soc: Optional[int] = None
) -> Tensor:
    """Reconstruct absolute predictions from residuals for power-only or power+SOC heads.

    Adds the residual prediction at time t to the base feature value at time t,
    writing into horizon slices that are valid (t + h < T).

    Args:
        P_res: residual predictions, shape (B, T, H, C). C ∈ {1 (power), 2 (power+SOC)}.
        X: input features, shape (B, T, F), scaled the same way as during training.
        idx_power: column index of the power feature in X.
        idx_soc: optional column index of the SOC feature in X. Required if C == 2.

    Returns:
        Tensor of absolute predictions with shape (B, T, H, C), in the same scaled
        space as X. For power-only heads (C == 1), returns power in channel 0.
    """
    assert P_res.dim() == 4 and X.dim() == 3, "unexpected Tensor dimensions"
    B, T, H, C = P_res.shape

    if X.device != P_res.device:
        X = X.to(P_res.device)

    if C == 1:
        # power-only
        base = X[..., idx_power].unsqueeze(-1)  # (B, T, 1)
    elif C == 2:
        if idx_soc is None:
            raise ValueError("idx_soc must be provided when residual head has 2 channels.")
        base = torch.stack([X[..., idx_power], X[..., idx_soc]], dim=-1)  # (B, T, 2)
    else:
        raise ValueError(f"unsupported residual head size C={C}; expected 1 or 2")

    P_abs = torch.zeros_like(P_res)

    # fill valid prefixes per horizon: t ∈ [0, T-(h+1))
    for h in range(H):
        end = T - (h + 1)
        if end <= 0:
            break
        P_abs[:, :end, h, :] = base[:, :end, :] + P_res[:, :end, h, :]

    return P_abs


def inverse_targets_np(arr_scaled: np.ndarray,
                       power_scaler: MinMaxScaler,
                       soc_scaler: MinMaxScaler | None = None) -> np.ndarray:
    """Inverse-transform target channels from scaled to original units.
    Supports one or two channels:
        C == 1 then channel 0 is power (kW)
        C >= 2 then channel 0 = power, channel 1 = SOC
    Extra channels (C > 2) are left as-is.
    """
    out = arr_scaled.copy()
    if out.size == 0:
        return out

    C = out.shape[-1]

    # power (always channel 0)
    out[..., 0] = power_scaler.inverse_transform(
        out[..., 0].reshape(-1, 1)
    ).reshape(out.shape[:-1])

    # optional SOC (channel 1)
    if C > 1 and soc_scaler is not None:
        out[..., 1] = soc_scaler.inverse_transform(
            out[..., 1].reshape(-1, 1)
        ).reshape(out.shape[:-1])

    return out


def macro_rmse_per_session(
    P_abs: Tensor,
    Y_abs: Tensor,
    lengths: Tensor,
    power_min: float | Tensor,
    power_max: float | Tensor,
    horizon_decay: float | None = None,
) -> Tensor:
    """computes per-sequence macro-RMSE (kW) for a single target across horizons.

    it inverse-transforms scaled absolute predictions and targets to kW, applies
    horizon weights, and aggregates RMSE over horizons for each sequence. it only
    evaluates residuals on time steps that have valid targets for all horizons,
    matching the training-time masking behaviour.
    
    args:
        P_abs: scaled absolute predictions with shape (B, T, H, 1) or (B, T, H).
        Y_abs: scaled absolute targets with shape (B, T, H, 1) or (B, T, H).
        lengths: 1d tensor of shape (B,) with original sequence lengths.
        power_min: min value used by the power MinMaxScaler (float or 0d tensor).
        power_max: max value used by the power MinMaxScaler (float or 0d tensor).
        horizon_decay: optional exponential decay parameter alpha for horizon weights.
                       if None or 0.0, uses uniform weights over horizons.

    returns:
        tensor of shape (B,) with per-sequence macro-RMSE in kW. sequences with no
        valid evaluation timesteps receive NaN.
    """
    # normalises power_min / power_max to plain floats
    if isinstance(power_min, torch.Tensor):
        pmin = float(power_min.item())
    else:
        pmin = float(power_min)

    if isinstance(power_max, torch.Tensor):
        pmax = float(power_max.item())
    else:
        pmax = float(power_max)

    # ensures 4d tensors with a single target channel at the end
    if P_abs.dim() == 3:
        P_abs = P_abs.unsqueeze(-1)  # (B, T, H, 1)
    if Y_abs.dim() == 3:
        Y_abs = Y_abs.unsqueeze(-1)  # (B, T, H, 1)

    # works on the same device as the predictions
    device = P_abs.device

    # inverses min–max to kW, single target in channel 0
    P_kw = P_abs[..., 0] * (pmax - pmin) + pmin  # (B, T, H)
    Y_kw = Y_abs[..., 0] * (pmax - pmin) + pmin  # (B, T, H)

    B, T, H = P_kw.shape

    # horizon weights: uniform if decay is None or 0
    if horizon_decay is None or horizon_decay == 0.0:
        w_h = torch.full((H,), 1.0 / H, dtype=P_kw.dtype, device=device)
    else:
        # use the shared horizon_weights helper, as a (H,) vector on the correct device
        w_h = horizon_weights(
            H=H,
            alpha=horizon_decay,
            device=device,
            normalise=True,
            as_vector=True,
        )  # (H,)

    # masks out timesteps without valid residual targets
    # this mirrors _vectorized_mask in train.py:
    #   t in [1, length - H)  for all horizons, so every used timestep has valid y_{t+h}
    lengths_dev = lengths.to(device=device)
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
    end = lengths_dev.unsqueeze(1) - H                                # (B, 1)
    mask_2d = (t_idx >= 1) & (t_idx < end)                            # (B, T)

    errs = torch.full((B,), float("nan"), dtype=P_kw.dtype, device=device)

    for b in range(B):
        valid_t = mask_2d[b]  # (T,)
        n_valid = int(valid_t.sum().item())
        if n_valid <= 0:
            # no valid residuals for this sequence
            continue

        # selects the valid timesteps: shapes (n_valid, H)
        y_b = Y_kw[b][valid_t, :]
        p_b = P_kw[b][valid_t, :]

        diff_sq = (p_b - y_b) ** 2  # (n_valid, H)
        # mean squared error per horizon h
        per_h_mse = diff_sq.mean(dim=0)  # (H,)
        # weighted average over horizons, then square root
        weighted_mse = torch.dot(w_h, per_h_mse)
        errs[b] = torch.sqrt(weighted_mse)

    return errs



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
    """Yields (sid, residuals[T,H,C], T) for each session, residuals in original units.

    Handles both power-only (C == 1) and power+SOC (C == 2) heads.
    """
    for sid in sids:
        b = make_bundle_from_session_df(
            model=model, df_scaled=df_scaled, sid=int(sid), device=device,
            input_features=input_features, target_features=target_features, horizon=horizon,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp
        )
        Y_scaled = b.Y_sample.numpy()  # (T,H,C)
        P_scaled = b.P_sample.numpy()  # (T,H,C)
        T, H, C = Y_scaled.shape

        # base in scaled space, matching number of channels in Y/P
        if C == 1:
            base = b.X_sample[:, [idx_power_inp]].unsqueeze(1).numpy()      # (T,1,1)
        else:
            base = b.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1).numpy()  # (T,1,2)

        Y_abs_scaled = base + Y_scaled                 # (T,H,C)
        P_abs_scaled = base + P_scaled                 # (T,H,C)

        Y = inverse_targets_np(Y_abs_scaled, power_scaler, soc_scaler)  # (T,H,C)
        P = inverse_targets_np(P_abs_scaled, power_scaler, soc_scaler)  # (T,H,C)
        R = Y - P                                                      # (T,H,C)

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
    bundle: SessionPredsBundle,
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
    """Compute RWSE for one bundle using calibrated medians and MADs.

    Works for power-only (C == 1) and power+SOC (C == 2) heads.
    """
    Y_scaled = np.asarray(bundle.Y_sample, dtype=np.float64)  # (T,H,C)
    P_scaled = np.asarray(bundle.P_sample, dtype=np.float64)  # (T,H,C)
    T, H, C = Y_scaled.shape

    # base in scaled space
    if C == 1:
        base = bundle.X_sample[:, [idx_power_inp]].unsqueeze(1).numpy() # (T,1,1)
    else:
        base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1).numpy() # (T,1,2)

    Y_abs_scaled = base + Y_scaled
    P_abs_scaled = base + P_scaled

    Y_abs = inverse_targets_np(Y_abs_scaled, power_scaler, soc_scaler)  # (T,H,C)
    P_abs = inverse_targets_np(P_abs_scaled, power_scaler, soc_scaler) # (T,H,C)

    t0 = min(int(t_min_eval), max(0, T - 1))
    R = (Y_abs - P_abs)[t0:]  # (T',H,C)
    if R.size == 0:
        return 0.0, T

    m   = np.asarray(m,   dtype=np.float64) # (H,C)
    mad = np.asarray(mad, dtype=np.float64) # (H,C)
    w_h = np.asarray(w_h, dtype=np.float64) # (H,)
    w_c = np.asarray(w_c, dtype=np.float64) # (C,)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z  = np.abs(R - m[None, :, :]) / mad[None, :, :]  # (T',H,C)
        Z  = np.clip(Z, a_min=None, a_max=cap)
        Zw = Z * w_h[None, :, None] * w_c[None, None, :]
        Zw = np.nan_to_num(Zw, nan=0.0, posinf=cap, neginf=0.0)

    score = Zw.sum(axis=(1, 2)).mean()
    return float(score), T


@torch.inference_mode()
def compute_session_MRMSE(
    model: nn.Module,
    loader: Iterable[Tuple[List[int], Tensor, Tensor, Tensor]],
    device: torch.device,
    power_scaler: MinMaxScaler,
    soc_scaler: Optional[MinMaxScaler],
    power_weight: float,
    idx_power_inp: int,
    idx_soc_inp: Optional[int],
    t_min_eval: int,
    horizon_weights_decay: Optional[float] = None,
) -> pd.DataFrame:
    """computes per-session Macro-RMSE (kW) for a *single-target* (power) residual forecaster.

    it reconstructs absolute power predictions from residuals, aligns prediction and
    target time dimensions, and then calls `macro_rmse_per_session` to obtain one
    error value per sequence. soc-related arguments are kept for backwards
    compatibility but are ignored when the model predicts power only.
    """
    model.eval()
    rows: list[dict[str, float | int]] = []

    # allow both a DataLoader and a simple list [(session_ids, X, Y, lengths)]
    for batch in loader:
        session_ids, Xb, Yb, lengths = batch  # Xb: [B, T, D], Yb: [B, T, H, 1] or [B, T, H]

        # ensure tensor types
        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, dtype=torch.long)
        else:
            lengths = lengths.to(dtype=torch.long)

        Xb = Xb.to(device=device, non_blocking=True)
        Yb = Yb.to(device=device, non_blocking=True)

        # model forward: lengths must be on cpu for packing
        lengths_cpu = lengths.to(device="cpu", dtype=torch.long)
        P_res, _ = model(Xb, lengths_cpu)  # [B, T_pred, H] or [B, T_pred, H, 1]
        if P_res.dim() == 3:
            P_res = P_res.unsqueeze(-1)    # -> [B, T_pred, H, 1]

        # reconstruct absolute *scaled* predictions; power-only when idx_soc_inp is None
        P_abs_scaled = reconstruct_abs_from_residuals_batch(P_res, Xb, idx_power_inp, idx_soc_inp)
        if P_abs_scaled.dim() == 3:
            P_abs_scaled = P_abs_scaled.unsqueeze(-1)  # safety

        # absolute targets (scaled), using the same base power feature as during training
        base_power = Xb[..., [idx_power_inp]].unsqueeze(2)  # [B, T, 1, 1]
        Y_abs_scaled = base_power + Yb                      # [B, T, H, 1]

        # align time dimensions between predictions and targets
        T_pred = P_abs_scaled.shape[1]
        T_true = Y_abs_scaled.shape[1]
        T_common = min(T_pred, T_true)
        if T_common <= 0:
            continue  # nothing to evaluate

        if T_pred != T_true:
            P_use = P_abs_scaled[:, :T_common]            # [B, T_common, H, 1]
            Y_use = Y_abs_scaled[:, :T_common]            # [B, T_common, H, 1]
            lengths_use = lengths.clamp(max=T_common)     # keep consistent with T_common
        else:
            P_use = P_abs_scaled
            Y_use = Y_abs_scaled
            lengths_use = lengths

        # scalar min/max from the power scaler
        power_min = float(power_scaler.data_min_[0])
        power_max = float(power_scaler.data_max_[0])
        decay = float(horizon_weights_decay) if horizon_weights_decay is not None else None

        # single-target power channel
        P_abs_power = P_use[..., 0:1]  # [B, T_common, H, 1]

        # compute per-sequence Macro-RMSE (kW); returns tensor of shape [B]
        errs_tensor = macro_rmse_per_session(P_abs_power, Y_use, lengths_use, power_min, power_max, decay)
        errs = errs_tensor.detach().cpu().tolist()
        lens = lengths_use.detach().cpu().tolist()

        for sid, e, L in zip(session_ids, errs, lens):
            if math.isnan(e):
                continue
            rows.append(
                {
                    "charging_id": int(sid),
                    "error": float(e),
                    "length": int(L),
                }
            )

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
        b = make_bundle_from_session_df(
            model=model, df_scaled=df_scaled, sid=int(sid), device=device,
            input_features=input_features, target_features=target_features, horizon=horizon,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp)
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


def compute_bundle_error(bundle: SessionPredsBundle,
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
    Takes a distribution of errors and returns the error value at the given percentile (e.g. the 95th percentile).
    The inverse of percentile_of_threshold
    """
    return round(float(np.nanpercentile(errors, pct_thr)), 4)

def percentile_of_threshold(all_errs_sorted, thr: float) -> float:
    """
    Takes a threshold value and returns what percentile of the error distribution it corresponds to.
    The inverse of percentile_threshold"""
    idx = np.searchsorted(all_errs_sorted, thr, side="right")
    return 100.0 * idx / max(1, len(all_errs_sorted))


def classify_by_threshold(df_errs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Adds `label` column: 'abnormal' if error > threshold else 'normal'.
    """
    df = df_errs.copy()
    df["label"] = np.where(df["error"] > threshold, "abnormal", "normal")
    return df.sort_values("error", ascending=True).reset_index(drop=True)


# TODO: Write a new function that is more user-friendly (less parameters)
# for classifying a single charging session
def classify_session(
    model,
    df_scaled: pd.DataFrame,
    sid: int,
    *,
    device: torch.device,
    input_features: list[str],
    target_features: list[str],
    horizon: int,
    power_scaler: MinMaxScaler,
    soc_scaler: MinMaxScaler,
    idx_power_inp: int,
    idx_soc_inp: int,
    power_weight: float = 1.0, # ignored in power-only case
    decay: float = 0.2,
    t_min_eval: int = 1, # currently matched by macro_rmse_per_session's behaviour
    threshold: float = 10.0,
) -> tuple[int, float]:
    """Classify a single session as normal (0) or abnormal (1) using Macro-RMSE.

    Works for power-only or power+SOC models; only the power channel contributes
    to the error. The decision rule is error > threshold.
    """
    # build a prediction bundle for this session
    bundle = make_bundle_from_session_df(
        model=model, df_scaled=df_scaled, sid=int(sid), device=device,
        input_features=input_features, target_features=target_features, horizon=horizon,
        power_scaler=power_scaler, soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp
    )

    X = bundle.X_sample  # (T,F)
    Y_res = bundle.Y_sample  # (T,H,C)
    P_res = bundle.P_sample  # (T,H,C)

    if Y_res.ndim == 2:
        Y_res = Y_res[..., None]
    if P_res.ndim == 2:
        P_res = P_res[..., None]

    T, H, C = Y_res.shape

    # absolute preds/targets in scaled space for power channel only
    base_power = X[:, idx_power_inp].unsqueeze(-1).unsqueeze(1)   # (T,1,1)
    Y_abs_scaled = base_power + Y_res[..., 0:1]   # (T,H,1)
    P_abs_scaled = base_power + P_res[..., 0:1]  

    # wrap to batch dimension for reuse of macro_rmse_per_session
    P_abs_b = P_abs_scaled.unsqueeze(0)  # (1,T,H,1)
    Y_abs_b = Y_abs_scaled.unsqueeze(0)  
    lengths = torch.tensor([T], dtype=torch.long)

    power_min = torch.tensor(float(power_scaler.data_min_[0]))
    power_max = torch.tensor(float(power_scaler.data_max_[0]))

    err_tensor = macro_rmse_per_session(
        P_abs_b, Y_abs_b, lengths, power_min, power_max, decay
    )
    error = float(err_tensor[0].item())
    label = int(error > float(threshold))
    return label, error

