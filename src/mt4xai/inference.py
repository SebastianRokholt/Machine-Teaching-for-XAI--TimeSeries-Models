# src/mt4xai/inference.py
from typing import Tuple, Dict, List
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

@torch.no_grad()
def predict_residuals(model, X: torch.Tensor, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Runs the sequence model and returns residual predictions with shape (B, T, H, C).
    Forward signature is (x, seq_lengths) -> (preds, out_lengths).
    We run forward on GPU but ALWAYS return CPU for downstream numpy/scaler ops.
    """
    model.eval()
    X = X.to(device)

    # pack_padded_sequence requires CPU int64 lengths
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, dtype=torch.long)
    else:
        lengths = lengths.to(dtype=torch.long, device="cpu")

    P_res, _ = model(X, lengths)
    return P_res.detach().cpu()  # <-- move to CPU here


def reconstruct_abs_from_residuals_batch(X: torch.Tensor,
                                         P_res: torch.Tensor,
                                         idx_power: int,
                                         idx_soc: int) -> torch.Tensor:
    """
    Convert residual preds (delta) to absolute preds using the base value at t:
    P_abs[:, t, h, c] = base[:, t, c] + P_res[:, t, h, c], valid where t+h < T.
    X: (B, T, F), P_res: (B, T, H, C) -> P_abs: (B, T, H, C)
    """
    # Ensure both live on the same device (we standardize on CPU in this pipeline)
    if P_res.device != X.device:
        P_res = P_res.to(X.device)

    B, T, H, C = P_res.shape
    base = torch.stack([X[..., idx_power], X[..., idx_soc]], dim=-1)  # (B, T, 2)
    P_abs = torch.zeros_like(P_res)
    for h in range(H):
        end = T - (h + 1)
        if end <= 0:
            break
        P_abs[:, :end, h, :] = base[:, :end, :] + P_res[:, :end, h, :]
    return P_abs


def inverse_targets_np(arr_scaled: np.ndarray,
                       power_scaler: MinMaxScaler,
                       soc_scaler: MinMaxScaler) -> np.ndarray:
    """
    Inverse-transform only the two target channels from scaled to original units.
    arr_scaled: (..., 2) with channels [power, soc]
    """
    out = arr_scaled.copy()
    # power
    out[..., 0] = power_scaler.inverse_transform(out[..., 0].reshape(-1, 1)).reshape(out.shape[:-1])
    # soc
    out[..., 1] = soc_scaler.inverse_transform(out[..., 1].reshape(-1, 1)).reshape(out.shape[:-1])
    return out

def macro_rmse_per_session(P_abs: torch.Tensor,
                           Y_abs: torch.Tensor,
                           lengths: torch.Tensor,
                           power_weight: float = 0.5, 
                           t_min_eval: int = 0,
                           w_h: np.ndarray | None = None) -> np.ndarray:
    """
    computes one scalar error per session:
       sum_h w_h[h] * [ w*rmse_power(h) + (1-w)*rmse_soc(h) ]
    only timesteps t >= t_min_eval + 1 are included.
    if w_h is None, uses uniform weights across horizons.
    """
    B, T, H, C = P_abs.shape
    errs = np.zeros(B, dtype=np.float64)
    if w_h is None:
        w_h = np.ones(H, dtype=np.float64) / max(1, H)

    for b in range(B):
        L = int(lengths[b])
        vals, hs = [], []
        for h in range(H):
            end = L - (h + 1)
            if end <= t_min_eval:
                continue
            diff = (P_abs[b, t_min_eval:end, h, :] - Y_abs[b, t_min_eval:end, h, :]).cpu().numpy()
            rmse_c = np.sqrt(np.mean(diff**2, axis=0))
            v = power_weight * rmse_c[0] + (1.0 - power_weight) * rmse_c[1]
            vals.append(v); hs.append(h)
        if vals:
            errs[b] = float(np.sum([w_h[h] * v for v, h in zip(vals, hs)]))
        else:
            errs[b] = np.nan
    return errs

