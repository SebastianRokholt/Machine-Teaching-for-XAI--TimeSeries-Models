# src/mt4xai/plot.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from .ad import compute_bundle_error

# -----------------------------
# Shared lightweight data class
# -----------------------------
@dataclass
class SampleBundle:
    """
    Compact container for a single sample's tensors and metadata in *scaled* space.
    - X_sample: (T, F)
    - Y_sample: (T, H, C) residual targets (delta to add to base @ t)
    - P_sample: (T, H, C) residual predictions (model output)
    - true_*_unscaled: arrays in original units (for plotting ground truth lines)
    """
    batch_index: int
    sample_index: int
    length: int
    horizon: int
    num_targets: int
    X_sample: torch.Tensor
    Y_sample: torch.Tensor
    P_sample: torch.Tensor
    true_power_unscaled: np.ndarray
    true_soc_unscaled:   np.ndarray
    session_id: Optional[str] = None  # new: carry charging_id when available

# -----------------------------
# Helpers
# -----------------------------
def _get_nth_batch(loader, n: int):
    it = iter(loader)
    for _ in range(n):
        next(it)
    return next(it)

@torch.no_grad()
def fetch_sample_bundle(model: nn.Module, loader,
                        batch_index: int, sample_index: int, device: torch.device,
                        power_scaler, soc_scaler,
                        idx_power_inp: int, idx_soc_inp: int) -> SampleBundle:
    """
    Build a SampleBundle from a DataLoader batch (mirrors modelling notebook).
    - Forward pass on GPU, lengths kept on CPU int64 (pack_padded requirement).
    - Returns CPU tensors inside the bundle for plotting/post-proc.
    """
    model.eval()
    batch = _get_nth_batch(loader, batch_index)

    session_ids = None
    if len(batch) == 4:
        session_ids, Xb, Yb, Ls = batch
    else:
        Xb, Yb, Ls = batch

    if sample_index >= Xb.shape[0]:
        raise IndexError(f"sample_index {sample_index} out of range for batch {batch_index} (size={Xb.shape[0]}).")

    X_dev = Xb.to(device, non_blocking=True)
    Ls_cpu = Ls.to(dtype=torch.long, device="cpu")  # lengths must be CPU int64
    P_dev, _ = model(X_dev, Ls_cpu)

    T = Ls[sample_index].item()
    P_s = P_dev[sample_index, :T].cpu()
    Y_s = Yb[sample_index, :T].cpu()
    X_s = Xb[sample_index, :T].cpu()

    power_true = power_scaler.inverse_transform(X_s[:, [idx_power_inp]].numpy()).ravel()
    soc_true   = soc_scaler.inverse_transform(  X_s[:, [idx_soc_inp  ]].numpy()).ravel()

    H, C = P_s.shape[1], P_s.shape[2]
    sid = None if session_ids is None else session_ids[sample_index]
    return SampleBundle(
        batch_index=batch_index, sample_index=sample_index,
        length=T, horizon=H, num_targets=C,
        X_sample=X_s, Y_sample=Y_s, P_sample=P_s,
        true_power_unscaled=power_true, true_soc_unscaled=soc_true,
        session_id=sid
    )


@torch.inference_mode()
def make_bundle_from_session(model: nn.Module, df_scaled,
                             sid: str,
                             device: torch.device,
                             input_features: List[str],
                             target_features: List[str],
                             horizon: int,
                             power_scaler, soc_scaler,
                             idx_power_inp: int, idx_soc_inp: int,
                             sort_by: str = "timestamp") -> SampleBundle:
    """
    Build a SampleBundle directly from a single session DataFrame (for anomaly notebook).
    Ensures the same alignment as the modelling bundle.
    """
    g = df_scaled[df_scaled["charging_id"] == sid].sort_values(sort_by)
    X = torch.tensor(g[input_features].to_numpy(np.float32))                 # (T, F)
    y_abs = g[target_features].to_numpy(np.float32)                          # (T, C)
    T = y_abs.shape[0]
    Y = np.zeros((T, horizon, len(target_features)), dtype=np.float32)       # residuals
    for h in range(1, horizon+1):
        Y[:-h, h-1, :] = y_abs[h:, :] - y_abs[:-h, :]

    X_dev = X.unsqueeze(0).to(device, non_blocking=True)    # (1, T, F)
    L_cpu = torch.tensor([T], dtype=torch.long, device="cpu")
    P_dev, _ = model(X_dev, L_cpu)                          # (1, T, H, C)
    P = P_dev.squeeze(0).cpu()                              # (T, H, C)

    power_true = power_scaler.inverse_transform(X[:, [idx_power_inp]].numpy()).ravel()
    soc_true   = soc_scaler.inverse_transform(  X[:, [idx_soc_inp  ]].numpy()).ravel()

    return SampleBundle(
        batch_index=0, sample_index=0,
        length=T, horizon=horizon, num_targets=len(target_features),
        X_sample=X.cpu(), Y_sample=torch.from_numpy(Y), P_sample=P,
        true_power_unscaled=power_true, true_soc_unscaled=soc_true,
        session_id=sid
    )

def reconstruct_abs_from_bundle(bundle: SampleBundle, idx_power_inp: int, idx_soc_inp: int) -> torch.Tensor:
    """
    Reconstruct absolute predictions in *scaled* space:
      P_abs[t, h, c] = X[t, c_base] + P_res[t, h, c], aligned at t+h.
    Returns (T, H, C), CPU torch tensor.
    """
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)  # (T,1,2)
    return bundle.P_sample + base

def _valid_time_bounds(T: int, H: int) -> Tuple[int, int]:
    """Valid i0 indices for predicting at i0+h (< T). Returns [start, end_exclusive]."""
    return 1, max(1, T - H)


def plot_full_session(bundle: SampleBundle, power_scaler, soc_scaler,
                      idx_power_inp: int, idx_soc_inp: int,
                      target: str = "power",
                      title_suffix: str = "",
                      t_min_eval: int = 1,
                      error: Optional[float] = None,
                      threshold: Optional[float] = None,
                      label: Optional[str] = None):
    """
    Multi-horizon full-session plot with aligned horizons:
      for each horizon h: plot at absolute time indices t+h (as in modelling).
    Title optionally shows: session_id, error, threshold, label.
    """
    assert target in {"power", "soc"}
    T, H = bundle.length, bundle.horizon
    t = np.arange(T)
    idx = 0 if target == "power" else 1
    scaler = power_scaler if target == "power" else soc_scaler
    true = bundle.true_power_unscaled if idx == 0 else bundle.true_soc_unscaled

    # Title parts
    title_bits = []
    if bundle.session_id is not None:
        title_bits.append(f"Session ID: {bundle.session_id}")
    if label is not None:
        title_bits.append(label.upper())
    if error is not None:
        title_bits.append(f"error: {error:.1f}")
    if threshold is not None:
        title_bits.append(f"threshold: {threshold:.1f}")
    if title_suffix:
        title_bits.append(title_suffix)
    title_str = ", ".join(title_bits) if title_bits else title_suffix

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=t, y=true, color="black", linewidth=2.5, label=f"True {target.title()}")  # Plots true power/SOC curve
    palette = sns.color_palette("deep", n_colors=H)
    
    P_abs = bundle.P_sample + bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)

    for h0 in range(H):
        i_valid = np.arange(t_min_eval, T - (h0 + 1))
        if i_valid.size == 0:
            continue
        t_abs = i_valid + (h0 + 1)
        preds_scaled = P_abs[i_valid, h0, idx].numpy().reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).ravel()
        # Use scatter for all error-contributing points
        plt.scatter(t_abs, preds, s=10, color=palette[h0], alpha=0.5)

        # Optionally, connect points with a dashed line for visual continuity
        sns.lineplot(x=t_abs, y=preds, linestyle="--", linewidth=1.8, color=palette[h0])


        # i_valid = np.arange(1, T - (h0 + 1))
        # if i_valid.size == 0:
        #     continue
        # t_abs = i_valid + (h0 + 1)
        # preds_scaled = P_abs[i_valid, h0, idx].numpy().reshape(-1, 1)
        # preds = scaler.inverse_transform(preds_scaled).ravel()
        # sns.lineplot(x=t_abs, y=preds, linestyle="--", linewidth=1.8, color=palette[h0],
        #              label=f"Horizon={h0+1}", marker="o", markersize=3)
        # plt.scatter(t_abs, preds, s=10, color=palette[h0], alpha=0.5)

    base_title = f"{target.upper()} predictions — batch {bundle.batch_index}, sample {bundle.sample_index}"
    plt.title(base_title + (f" — {title_str}" if title_str else ""))
    plt.xlabel("Time index")
    plt.ylabel("Power (kW)" if idx == 0 else "SOC (%)")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------
# Convenience wrapper for (BATCH, SAMPLE)
# -----------------------------------------
def plot_session_from_loader_by_index(model: nn.Module, loader,
                                      batch_index: int, sample_index: int,
                                      device: torch.device,
                                      power_scaler, soc_scaler,
                                      idx_power_inp: int, idx_soc_inp: int,
                                      power_weight: float,
                                      threshold: Optional[float] = None,
                                      target_list: Optional[List[str]] = None):
    """
    Convenience: fetch bundle via (batch_index, sample_index), compute its error,
    infer label using threshold, and plot POWER/SOC (like modelling).
    """
    if target_list is None:
        target_list = ["power", "soc"]

    bundle = fetch_sample_bundle(model, loader, batch_index, sample_index, device,
                                 power_scaler, soc_scaler, idx_power_inp, idx_soc_inp)

    err = compute_bundle_error(bundle, power_scaler, soc_scaler, power_weight, idx_power_inp, idx_soc_inp)
    lbl = None
    if threshold is not None and np.isfinite(err):
        lbl = "abnormal" if err > threshold else "normal"

    for tgt in target_list:
        plot_full_session(bundle, power_scaler, soc_scaler,
                          idx_power_inp, idx_soc_inp,
                          target=tgt,
                          error=err, threshold=threshold, label=lbl)
