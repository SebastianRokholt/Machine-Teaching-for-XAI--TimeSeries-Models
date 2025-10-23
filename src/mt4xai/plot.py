# src/mt4xai/plot.py
from __future__ import annotations
from matplotlib.figure import Figure
from matplotlib.pylab import Axes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Literal, Optional, Tuple
from .data import SessionPredsBundle, reconstruct_abs_from_bundle


# TODO: Improve visualization
def plot_raw_session(
    power_kw: np.ndarray,
    *,
    soc: np.ndarray | None = None,
    soc_mode: Literal["none", "raw", "simpl"] = "none",
    title: str | None = None,
    y_lim: tuple[float, float] | None = None,
    ) -> tuple[Figure, Axes]:
    """Plots raw power (left hand axis, kw) and optionally SOC (right hand axis, %).

    Args:
        power_kw: Dense power in kW, shape (T,).
        soc: Optional SOC series in percent, shape (T,).
        soc_mode: "none", "raw", or "simpl".
        title: Optional figure title.
        y_lim: Optional power y-limits (min, max).

    Returns:
        Matplotlib figure and left axis.
    """
    y = np.asarray(power_kw, dtype=float).reshape(-1)
    x = np.arange(y.size)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(x, y, linewidth=1.2, label="power (raw)")
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel("time (arbitrary units)")
    ax.set_ylabel("power (kW)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    lines = ax.get_lines()
    if soc_mode != "none" and soc is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, 100.0)
        ax2.set_ylabel("state of charge (%)")
        soc_color = "#0b3d91"  # dark blue
        ax2.plot(x, np.clip(soc, 0.0, 100.0), linewidth=1.2, label=f"soc ({soc_mode})", color=soc_color)
        lines = lines + ax2.get_lines()

    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best", fontsize=9)
    return fig, ax


def plot_raw_pred_session(bundle: SessionPredsBundle, power_scaler, soc_scaler,
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

    # Seaborn plot on a Matplotlib Axes with autoscale disabled
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Locks the y-axis before drawing, and disables autoscaling
    if y_lim is None:
        if tgt_idx == 0:  # power, per-session scale if global not passed
            pred_max = float(np.max(np.concatenate(preds_all))) if preds_all else 0.0
            y_max = max(float(np.max(true)), pred_max, 1.0)
            y_lim = (0.0, y_max * 1.02)
        else:  # SOC is always in range [0, 100] (%)
            y_lim = (0.0, 100.0)

    # True series
    sns.lineplot(x=t, y=true, color="black", linewidth=2.5, label=f"True {target.title()}", ax=ax)

    # Horizon curves with labels
    palette = sns.color_palette("deep", n_colors=H)
    for h0 in range(H):
        if h0 >= len(preds_all):
            continue
        t_abs = t_abs_all[h0]
        preds = preds_all[h0]
        # line with legend label
        sns.lineplot(x=t_abs, y=preds, linestyle="--", linewidth=1.8,
                     color=palette[h0], label=f"h={h0+1}", ax=ax)
        # scatterplot without legend entry
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


# TODO: Improve visualization
def plot_raw_simpl_session(
    power_kw: np.ndarray,
    idx: np.ndarray,
    val_kw: np.ndarray,
    *,
    soc: np.ndarray | None = None,
    soc_mode: Literal["none", "raw", "simpl"] = "none",
    title: str | None = None,
    y_lim: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plots raw and simplified power. Optionally overlays SOC on right-hand axis.

    Args:
        power_kw: Dense power in kW, shape (T,).
        idx: Knot indices for simplified power, shape (K,).
        val_kw: Knot values in kW, shape (K,).
        soc: Optional SOC series in percent, shape (T,).
        soc_mode: "none", "raw", or "simpl".
        title: Optional figure title.
        y_lim: Optional power y-limits (min, max).

    Returns:
        Matplotlib figure and left axis.
    """
    y = np.asarray(power_kw, dtype=float).reshape(-1)
    T = y.size
    xs = np.arange(T)
    idx = np.asarray(idx, dtype=int).reshape(-1)
    val = np.asarray(val_kw, dtype=float).reshape(-1)

    y_hat = np.interp(xs, idx.astype(float), val)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(xs, y, linewidth=1.0, alpha=0.85, label="power (raw)")
    ax.plot(xs, y_hat, linewidth=1.8, label="power (simplified)")
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel("time (arbitrary units)")
    ax.set_ylabel("power (kW)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    lines = ax.get_lines()
    if soc_mode != "none" and soc is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, 100.0)
        ax2.set_ylabel("state of charge (%)")
        soc_color = "#0b3d91"
        ax2.plot(xs, np.clip(soc, 0.0, 100.0), linewidth=1.2, label=f"soc ({soc_mode})", color=soc_color)
        lines = lines + ax2.get_lines()

    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best", fontsize=9)
    return fig, ax


def plot_raw_pred_simp_session(
    bundle: SessionPredsBundle, power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int,
    simpl_power_unscaled: np.ndarray, *, session_id: int | None = None, k: int | None = None,
    threshold: float | None = None, simp_error: float | None = None, orig_error: float | None = None,
    label: str | None = None, decay_lambda: float | None = None, noise_std_kw: float | None = None,
    robust_tau: float | None = None, figsize: tuple[float, float] = (12.0, 5.0), dpi: int = 110,
    t_min_eval: int = 1
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    plots true power, all-horizon predicted power, and an ors-simplified power curve.
    returns the matplotlib figure and axes so callers can manage rendering.
    """
    T, H = bundle.length, bundle.horizon
    t = np.arange(T)
    true = np.asarray(bundle.true_power_unscaled, dtype=float)
    sid = int(session_id) if session_id is not None else bundle.session_id

    # create figure/axes explicitly and apply all config to them
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # true power
    sns.lineplot(x=t, y=true, color="black", linewidth=2.2, label="True power", ax=ax)

    # multi-horizon predictions (power)
    palette = sns.color_palette("deep", n_colors=H)
    P_abs = reconstruct_abs_from_bundle(bundle, idx_power_inp, idx_soc_inp)
    for h0 in range(H):
        i_valid = np.arange(t_min_eval, T - (h0 + 1))
        if i_valid.size == 0:
            continue
        t_pred = i_valid + (h0 + 1)
        preds = power_scaler.inverse_transform(
            P_abs[i_valid, h0, 0].numpy().reshape(-1, 1)
        ).ravel()
        sns.lineplot(
            x=t_pred, y=preds, linestyle="--", linewidth=1.6, color=palette[h0],
            label=f"H={h0+1}", marker="o", markersize=2.5, ax=ax
        )

    # ors simplification
    sns.lineplot(
        x=t, y=simpl_power_unscaled, color="tab:red", linewidth=2.6, label="ORS simplification", ax=ax
    )

    # title
    ttl = f"ORS Simplification of Charging Session {sid}. Classification = {label}, k = {k}"
    ax.set_title(ttl)

    # caption string
    cap_parts: list[str] = []
    if label is not None:        cap_parts.append(f"Classification: {label}")
    if simp_error is not None:   cap_parts.append(f"prediction error on simplification = {simp_error:.3f}")
    if orig_error is not None:   cap_parts.append(f"prediction error on original = {orig_error:.3f}")
    if threshold is not None:    cap_parts.append(f"classification threshold = {threshold}")
    if k is not None:            cap_parts.append(f"k = {k}")
    if decay_lambda is not None: cap_parts.append(f"λ = {decay_lambda}")
    if noise_std_kw is not None: cap_parts.append(f"noise_std_kw = {noise_std_kw}")
    if robust_tau is not None:   cap_parts.append(f"robust_tau = {robust_tau}")
    fig_desc = ", ".join(cap_parts)

    # axes formatting
    ax.set_xlabel("Time index")
    ax.set_ylabel("Power (kW)")
    ax.grid(True)
    ax.legend()

    # layout and caption placement belong to the figure object
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.01, 0.02, fig_desc, fontsize=9, ha="left")

    return fig, ax
