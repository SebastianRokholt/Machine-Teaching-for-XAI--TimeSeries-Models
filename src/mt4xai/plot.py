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
from .data import SampleBundle


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
                      label: Optional[str] = None,
                      decay_lambda: Optional[float] = None,
                      figsize: Tuple[float] = (12.0, 5.0),
                      dpi: int = 110,
                      y_lim: Optional[Tuple[float, float]] = None):
    """
    Multi-horizon full-session plot with aligned horizons (Seaborn + fixed y-axis).

    Args:
        y_lim: (ymin, ymax). If None:
               - power -> computed per-session from [0, max(true,preds)].
               - soc   -> (0, 100).
               Pass a GLOBAL y_lim to keep the same scale across sessions.
    """
    assert target in {"power", "soc"}
    T, H = bundle.length, bundle.horizon
    t = np.arange(T)
    tgt_idx = 0 if target == "power" else 1
    scaler  = power_scaler if tgt_idx == 0 else soc_scaler
    true    = bundle.true_power_unscaled if tgt_idx == 0 else bundle.true_soc_unscaled

    # Title bits
    title_bits = []
    if bundle.session_id is not None: title_bits.append(f"Session ID: {bundle.session_id}")
    if label is not None:             title_bits.append(label.upper())
    if error is not None:             title_bits.append(f"error: {error:.1f}")
    if threshold is not None:         title_bits.append(f"threshold: {threshold:.1f}")
    if decay_lambda is not None:      title_bits.append(f"λ={decay_lambda:.2f}")
    if title_suffix:                   title_bits.append(title_suffix)
    title_str = ", ".join(title_bits) if title_bits else title_suffix

    # Absolute predictions (add base inputs to residual-like P_sample)
    base_inputs = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]]     # (T, 2) scaled
    P_abs = bundle.P_sample + base_inputs.unsqueeze(1)                 # (T, H, 2) scaled

    # Precompute all unscaled predictions for plotting
    preds_all, t_abs_all = [], []
    for h0 in range(H):
        i_valid = np.arange(t_min_eval, T - (h0 + 1))
        if i_valid.size == 0: 
            continue
        t_abs = i_valid + (h0 + 1)
        preds_scaled = P_abs[i_valid, h0, tgt_idx].numpy().reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).ravel()
        preds_all.append(preds)
        t_abs_all.append(t_abs)

    # ---- Seaborn plot on a Matplotlib Axes with autoscale disabled ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Lock y-axis BEFORE drawing, and disable autoscale so Seaborn can't expand it.
    if y_lim is None:
        if tgt_idx == 0:  # power, per-session scale if global not passed
            pred_max = float(np.max(np.concatenate(preds_all))) if preds_all else 0.0
            y_max = max(float(np.max(true)), pred_max, 1.0)
            y_lim = (0.0, y_max * 1.02)
        else:             # soc
            y_lim = (0.0, 100.0)

    # True series
    sns.lineplot(x=t, y=true, color="black", linewidth=2.5, label=f"True {target.title()}", ax=ax)

    # Horizon curves with labels h=1, h=2, ...
    palette = sns.color_palette("deep", n_colors=H)
    for h0 in range(H):
        if h0 >= len(preds_all):
            continue
        t_abs = t_abs_all[h0]
        preds = preds_all[h0]
        # line with legend label
        sns.lineplot(x=t_abs, y=preds, linestyle="--", linewidth=1.8,
                     color=palette[h0], label=f"h={h0+1}", ax=ax)
        # scatter without legend entry
        ax.scatter(t_abs, preds, s=10, color=palette[h0], alpha=0.5, label="_nolegend_")

    ax.set_ylim(*y_lim)
    ax.autoscale(enable=False, axis="y")
    ax.autoscale(enable=True, axis="x")
    base_title = f"{target.upper()} predictions — batch {bundle.batch_index}, sample {bundle.sample_index}"
    ax.set_title(base_title + (f" — {title_str}" if title_str else ""))
    ax.set_xlabel("Time index")
    ax.set_ylabel("Power (kW)" if tgt_idx == 0 else "SOC (%)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.4)
    ax.legend(ncol=1, frameon=True, fontsize=9, title="Legend")

    fig.tight_layout()
    plt.show()

