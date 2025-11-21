# src/mt4xai/inference.py
from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from .model import horizon_weights
from .data import SessionPredsBundle, ChargingSessionDataset, reconstruct_abs_from_bundle, session_collate_fn

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


@torch.inference_mode()
def make_bundle_from_session_df(
    model: torch.nn.Module,
    df_scaled: pd.DataFrame,
    sid: int,
    device: torch.device,
    *,
    input_features: list[str],
    target_features: list[str],
    horizon: int,
    power_scaler,
    soc_scaler,
    idx_power_inp: int,
    idx_soc_inp: int,
) -> SessionPredsBundle:
    """
    Create a SessionPredsBundle for a single charging session.

    Args:
        model: Trained residual model.
        df_scaled: Scaled dataframe containing all sessions.
        sid: Session ID to extract.
        device: Torch device.
        input_features: Ordered list of input features used for training.
        target_features: Ordered list of targets (power or power+SOC).
        horizon: Prediction horizon.
        power_scaler: Fitted MinMaxScaler for power.
        soc_scaler: Fitted MinMaxScaler for SOC.
        idx_power_inp: Index of 'power' in the input feature vector.
        idx_soc_inp: Index of 'soc' in the input feature vector.

    Returns:
        SessionPredsBundle containing tensors, truths, and metadata.
    """
    df_session = df_scaled[df_scaled["charging_id"] == sid].copy()
    df_session = df_session.sort_values("minutes_elapsed").reset_index(drop=True)

    X_np = df_session[input_features].to_numpy(dtype=np.float32)
    Y_np = df_session[target_features].to_numpy(dtype=np.float32)

    T = X_np.shape[0]
    lengths = torch.tensor([T], dtype=torch.long)

    X = torch.from_numpy(X_np).unsqueeze(0)
    Y = torch.from_numpy(Y_np).unsqueeze(0)

    with torch.no_grad():
        pred_resid, _ = model(X.to(device), lengths)
        pred_resid = pred_resid.cpu()

    # Prepare unscaled truths (power always present; SOC optional)
    true_power_unscaled = power_scaler.inverse_transform(
        X_np[:, idx_power_inp].reshape(-1, 1)
    ).reshape(-1)

    if "soc" in input_features:
        true_soc_unscaled = soc_scaler.inverse_transform(
            X_np[:, idx_soc_inp].reshape(-1, 1)
        ).reshape(-1)
    else:
        true_soc_unscaled = None

    return SessionPredsBundle(
        batch_index=0,
        sample_index=0,
        length=T,
        horizon=horizon,
        num_targets=len(target_features),
        X_sample=X.squeeze(0),
        Y_sample=Y.squeeze(0),
        P_sample=pred_resid.squeeze(0),
        true_power_unscaled=true_power_unscaled,
        true_soc_unscaled=true_soc_unscaled,
        session_id=sid,
    )


def macro_rmse_per_session(
    P_abs: Tensor,
    Y_abs: Tensor,
    lengths: Tensor,
    power_min: float,
    power_max: float,
    decay: float,
    t_min_eval: int,
) -> Tensor:
    """computes macro-RMSE in kW per sequence for a single-target power forecaster.

    args:
        P_abs: scaled absolute predictions, shape (B, T, H) or (B, T, H, 1).
        Y_abs: scaled absolute targets, same shape as P_abs.
        lengths: 1d tensor of original sequence lengths, shape (B,).
        power_min: minimum value used by MinMaxScaler for power.
        power_max: maximum value used by MinMaxScaler for power.
        decay: horizon decay parameter lambda; if 0.0 uses uniform weights.
        t_min_eval: minimum time index from which errors contribute.

    returns:
        tensor of shape (B,) with macro-RMSE in kW per sequence.
    """
    # normalise shapes to (B, T, H, 1)
    if P_abs.dim() == 3:
        P_abs = P_abs.unsqueeze(-1)
    if Y_abs.dim() == 3:
        Y_abs = Y_abs.unsqueeze(-1)

    # choose device and move everything there
    device = P_abs.device
    Y_abs = Y_abs.to(device)
    lengths = lengths.to(device=device, dtype=torch.long)

    pmin = torch.tensor(float(power_min), device=device, dtype=P_abs.dtype)
    pmax = torch.tensor(float(power_max), device=device, dtype=P_abs.dtype)
    scale = pmax - pmin

    # inverse transform to kW; single target in channel 0
    P_kw = pmin + scale * P_abs[..., 0]  # (B, T, H)
    Y_kw = pmin + scale * Y_abs[..., 0]  # (B, T, H)

    B, T_max, H = P_kw.shape

    # horizon weights
    h = torch.arange(H, device=device, dtype=P_kw.dtype)
    if decay is None or float(decay) == 0.0:
        w_h = torch.full((H,), 1.0 / max(H, 1), device=device, dtype=P_kw.dtype)
    else:
        w_h = torch.exp(-float(decay) * h)
        w_h = w_h / w_h.sum()

    errs = torch.zeros(B, device=device, dtype=P_kw.dtype)

    for b in range(B):
        T = int(lengths[b].item())
        if T <= t_min_eval + 1:
            errs[b] = 0.0
            continue

        per_h_vals = []
        for h_idx in range(H):
            end = T - (h_idx + 1)
            if end <= t_min_eval:
                continue
            diff = P_kw[b, t_min_eval:end, h_idx] - Y_kw[b, t_min_eval:end, h_idx]
            rmse_h = torch.sqrt(torch.mean(diff * diff))
            per_h_vals.append(rmse_h)

        if per_h_vals:
            per_h = torch.stack(per_h_vals)
            errs[b] = torch.sum(per_h * w_h[: len(per_h_vals)])
        else:
            errs[b] = 0.0

    return errs



def list_unique_session_ids(df_scaled: pd.DataFrame) -> np.ndarray:
    """Returns sorted unique charging_id values as a plain int ndarray."""
    return np.asarray(np.sort(np.asarray(df_scaled["charging_id"].unique())), dtype=int)


# -------------------------------- RWSE: robust weighted squared error (power-only, single target) ---------------- #

def fit_rwse_robust_scalers(
    model: nn.Module,
    df_scaled: pd.DataFrame,
    device: torch.device,
    input_features: list[str],
    target_features: list[str],
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    horizon: int,
    t_min_eval: int = 1,
    session_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits robust location and scale parameters for RWSE on power residuals.

    It collects power residuals (true minus predicted power in kW) across the
    given sessions, for all horizons and time steps t >= t_min_eval. It then
    computes the per-channel median and median absolute deviation (MAD).

    Args:
        model: trained residual forecaster.
        df_scaled: panel of scaled charging sessions.
        device: torch device used for inference.
        input_features: list of input feature names.
        target_features: list of target feature names (includes power only).
        power_scaler: fitted MinMaxScaler for power.
        idx_power_inp: index of power in the input feature vector.
        horizon: forecast horizon H.
        t_min_eval: minimum time index t included in the residuals.
        session_ids: optional subset of charging ids to use.

    Returns:
        m: median of residuals for the power channel, shape (1,).
        mad: MAD of residuals for the power channel, shape (1,).
    """
    if not session_ids:
        session_ids = list(list_unique_session_ids(df_scaled))
    else:
        session_ids = list(map(int, session_ids))

    resid_all: list[np.ndarray] = []

    for sid in session_ids:
        bundle = make_bundle_from_session_df(
            model=model,
            df_scaled=df_scaled,
            sid=int(sid),
            device=device,
            input_features=input_features,
            target_features=target_features,
            horizon=horizon,
        )

        # single-session batch for reuse of reconstruction helper
        X = bundle.X_sample.unsqueeze(0)          # [1, T, C_in]
        P_res = bundle.P_sample.unsqueeze(0)      # [1, T, H, 1]
        Y_res = bundle.Y_sample.unsqueeze(0)      # [1, T, H, 1]
        lengths = torch.tensor([bundle.length], dtype=torch.long)

        P_abs, Y_abs = reconstruct_abs_from_residuals_batch(
            X_batch=X,
            P_batch=P_res,
            Y_batch=Y_res,
            lengths=lengths,
            idx_power_inp=idx_power_inp,
            power_min=float(power_scaler.data_min_[0]),
            power_max=float(power_scaler.data_max_[0]),
        )
        # drop batch and channel dims -> [T_common, H]
        P = P_abs[0, :, :, 0].cpu().numpy()
        Y = Y_abs[0, :, :, 0].cpu().numpy()

        if P.shape[0] <= t_min_eval:
            continue

        resid = Y[t_min_eval:, :] - P[t_min_eval:, :]   # [T_eff, H]
        resid_all.append(resid.reshape(-1, 1))          # flatten over t,h

    if not resid_all:
        # degenerate fallback to avoid division by zero
        m = np.zeros(1, dtype=float)
        mad = np.ones(1, dtype=float)
        return m, mad

    resid_all_arr = np.concatenate(resid_all, axis=0)  # [N, 1]
    m = np.median(resid_all_arr, axis=0)
    mad = np.median(np.abs(resid_all_arr - m), axis=0)

    # avoid zero MAD which would blow up scaling
    mad = np.where(mad < 1e-6, 1.0, mad)
    return m.astype(float), mad.astype(float)


def rwse_score_from_bundle(
    bundle: SessionPredsBundle,
    m: np.ndarray,
    mad: np.ndarray,
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    w_h: np.ndarray,
    cap: float = 5.0,
    t_min_eval: int = 1,
) -> tuple[float, int]:
    """Computes RWSE score for a single session bundle (power-only).

    The score is the robust weighted squared error over horizons, where
    residuals are scaled by median and MAD, capped at `cap`, and aggregated
    with exponential horizon weights w_h.

    Args:
        bundle: session prediction bundle.
        m: median residuals from fit_rwse_robust_scalers, shape (1,).
        mad: MAD residuals from fit_rwse_robust_scalers, shape (1,).
        power_scaler: fitted MinMaxScaler for power.
        idx_power_inp: index of power in input feature vector.
        w_h: horizon weights of shape (H,) that sum to 1.
        cap: cap on |scaled residual|, applied before squaring.
        t_min_eval: minimum time index t used in the score.

    Returns:
        score: RWSE value for this session.
        t_eff: number of effective time steps contributing to the score.
    """
    T = bundle.length
    H = bundle.horizon

    if T <= t_min_eval + 1:
        return 0.0, 0

    X = bundle.X_sample.unsqueeze(0)         # [1, T, C_in]
    P_res = bundle.P_sample.unsqueeze(0)     # [1, T, H, 1]
    Y_res = bundle.Y_sample.unsqueeze(0)     # [1, T, H, 1]
    lengths = torch.tensor([bundle.length], dtype=torch.long)

    P_abs, Y_abs = reconstruct_abs_from_residuals_batch(
        X_batch=X,
        P_batch=P_res,
        Y_batch=Y_res,
        lengths=lengths,
        idx_power_inp=idx_power_inp,
        power_min=float(power_scaler.data_min_[0]),
        power_max=float(power_scaler.data_max_[0]),
    )
    P = P_abs[0, :, :, 0].cpu().numpy()   # [T_common, H]
    Y = Y_abs[0, :, :, 0].cpu().numpy()

    if P.shape[0] <= t_min_eval:
        return 0.0, 0

    P_use = P[t_min_eval:, :]    # [T_eff, H]
    Y_use = Y[t_min_eval:, :]
    t_eff = P_use.shape[0]

    resid = Y_use - P_use              # [T_eff, H]
    resid_vec = resid.reshape(-1, 1)   # [T_eff * H, 1]

    m_power = float(m[0])
    mad_power = float(mad[0]) if mad[0] != 0.0 else 1e-6

    resid_scaled = (resid_vec - m_power) / mad_power
    resid_scaled_sq = np.clip(resid_scaled**2, a_min=None, a_max=cap**2)

    # reshape back to [T_eff, H, 1]; feature weights are 1 for power
    per_h_res = resid_scaled_sq.reshape(t_eff, H, 1)
    per_t_res = np.sqrt(np.sum(per_h_res, axis=2))      # [T_eff, H]

    # aggregate over horizons with weights w_h (sum to 1)
    score = float(np.sum(per_t_res * w_h[np.newaxis, :]))
    return score, t_eff


def compute_session_RWSE(
    model: nn.Module,
    df_scaled: pd.DataFrame,
    device: torch.device,
    input_features: list[str],
    target_features: list[str],
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    m: np.ndarray,
    mad: np.ndarray,
    horizon: int,
    horizon_weights_decay: float = 0.4,
    cap: float = 5.0,
    t_min_eval: int = 1,
    session_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Computes RWSE per charging session for a power-only residual model.

    Args:
        model: trained residual forecaster.
        df_scaled: panel of scaled charging sessions.
        device: torch device used for inference.
        input_features: list of input feature names.
        target_features: list of target feature names (includes power only).
        power_scaler: fitted MinMaxScaler for power.
        idx_power_inp: index of power in input feature vector.
        m: median residuals from fit_rwse_robust_scalers, shape (1,).
        mad: MAD residuals from fit_rwse_robust_scalers, shape (1,).
        horizon: forecast horizon H.
        horizon_weights_decay: decay for exponential horizon weights.
        cap: cap on |scaled residual| in RWSE.
        t_min_eval: minimum time index t used in the score.
        session_ids: optional subset of charging ids to evaluate.

    Returns:
        DataFrame with columns: charging_id, length, error (RWSE).
    """
    rows: list[dict] = []

    if not session_ids:
        session_ids = list(list_unique_session_ids(df_scaled))
    else:
        session_ids = list(map(int, session_ids))

    w_h = make_horizon_weights(horizon, decay=horizon_weights_decay)

    for sid in session_ids:
        bundle = make_bundle_from_session_df(
            model=model,
            df_scaled=df_scaled,
            sid=int(sid),
            device=device,
            input_features=input_features,
            target_features=target_features,
            horizon=horizon,
        )
        s, L = rwse_score_from_bundle(
            bundle=bundle,
            m=m,
            mad=mad,
            power_scaler=power_scaler,
            idx_power_inp=idx_power_inp,
            w_h=w_h,
            cap=cap,
            t_min_eval=t_min_eval,
        )
        rows.append(
            {
                "charging_id": int(sid),
                "length": int(L),
                "error": float(s),
            }
        )

    return pd.DataFrame(rows)


@torch.inference_mode()
def compute_session_MRMSE(
    model: nn.Module,
    loader: Iterable[Tuple[List[int], Tensor, Tensor, Tensor]],
    device: torch.device,
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    t_min_eval: int,
    horizon_weights_decay: float,
) -> pd.DataFrame:
    """computes per-session macro-RMSE (kW) for a single-target power forecaster.

    this is the power-only version used in the anomaly-detection and ORS notebooks.
    any SOC-related arguments are accepted for backwards compatibility but ignored.

    args:
        model: trained residual forecasting model.
        loader: iterable yielding (session_ids, X_batch, Y_batch, lengths).
        device: torch device used for inference.
        power_scaler: fitted MinMaxScaler for power.
        idx_power_inp: index of power in the input feature vector.
        t_min_eval: minimum time index from which errors contribute.
        horizon_weights_decay: horizon decay parameter lambda.
        soc_scaler: unused; kept for backwards compatibility.
        power_weight: unused; kept for backwards compatibility.
        idx_soc_inp: unused; kept for backwards compatibility.

    returns:
        dataframe with columns ["charging_id", "error", "length"].
    """
    model.eval()
    rows: list[dict[str, float | int]] = []

    for session_ids, Xb, Yb, lengths in loader:
        # ensure tensors and dtypes
        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, dtype=torch.long)
        else:
            lengths = lengths.to(dtype=torch.long)

        Xb = Xb.to(device=device, non_blocking=True)
        Yb = Yb.to(device=device, non_blocking=True)

        # forward pass: lengths on cpu for packing, outputs on device
        lengths_cpu = lengths.to(device="cpu", dtype=torch.long)
        P_res, _ = model(Xb, lengths_cpu)          # [B, T_pred, H] or [B, T_pred, H, 1]
        if P_res.dim() == 3:
            P_res = P_res.unsqueeze(-1)            # -> [B, T_pred, H, 1]

        # reconstruct absolute predictions in scaled space
        P_abs_scaled = reconstruct_abs_from_residuals_batch(
            P_res=P_res,
            X=Xb,
            idx_power=idx_power_inp,
            idx_soc=None,
        )
        if P_abs_scaled.dim() == 3:
            P_abs_scaled = P_abs_scaled.unsqueeze(-1)

        # absolute targets (scaled), using same base power as during training
        base_power = Xb[..., [idx_power_inp]].unsqueeze(2)  # [B, T, 1, 1]
        Y_abs_scaled = base_power + Yb                      # [B, T, H, 1]

        # align prediction / target time dims
        T_pred = P_abs_scaled.shape[1]
        T_true = Y_abs_scaled.shape[1]
        T_common = min(T_pred, T_true)
        if T_common <= 0:
            continue

        if T_pred != T_true:
            P_use = P_abs_scaled[:, :T_common]
            Y_use = Y_abs_scaled[:, :T_common]
            lengths_use = lengths.clamp(max=T_common)
        else:
            P_use = P_abs_scaled
            Y_use = Y_abs_scaled
            lengths_use = lengths

        power_min = float(power_scaler.data_min_[0])
        power_max = float(power_scaler.data_max_[0])

        errs_tensor = macro_rmse_per_session(
            P_abs=P_use,
            Y_abs=Y_use,
            lengths=lengths_use,
            power_min=power_min,
            power_max=power_max,
            decay=horizon_weights_decay,
            t_min_eval=t_min_eval,
        )

        errs = errs_tensor.detach().cpu().tolist()
        lens = lengths_use.detach().cpu().tolist()

        for sid, e, L in zip(session_ids, errs, lens):
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
    P_abs_scaled = reconstruct_abs_from_bundle(bundle, power_scaler, idx_power_inp)  # (T,H,2)
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
        P_abs_b, Y_abs_b, lengths, power_min, power_max, decay, t_min_eval
    )
    error = float(err_tensor[0].item())
    label = int(error > float(threshold))
    return label, error

