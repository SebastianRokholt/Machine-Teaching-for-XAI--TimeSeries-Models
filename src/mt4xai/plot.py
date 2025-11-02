# src/mt4xai/plot.py
from __future__ import annotations
from matplotlib.figure import Figure
from matplotlib.pylab import Axes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Literal, Optional, Tuple, List
from .data import SessionPredsBundle, reconstruct_abs_from_bundle

COLOUR_POWER_RAW  = "#f28e2b" # raw power
COLOUR_POWER_SIMPL  = "#b22222" # dark red
COLOUR_SOC = "#0b3d91" # dark blue (for raw and simplified)

def _axis_label_with_dual_icon(ax: Axes, label: str, *, colours: list[str],
                                 side: Literal["left", "right"] = "left",
                                 fontsize: float = 12.0, labelpad: float = 8.0) -> None:
    """
    draws multiple coloured '■' squares next to the y-axis label, no mathtext.
    - keeps the actual ylabel as plain text (so layout/tight_layout work)
    - places one coloured square per entry in `colors`
    - colours ticks + matching spine using the first colour
    """
    ax.set_ylabel(label, labelpad=labelpad, fontsize=fontsize)

    # manual positioning of squares in axes coordinates after label text
    x = -0.055 if side == "left" else 1.046
    y0 = 0.75 if side == "left" else 0.87

    if len(colours) == 1 and side == "left": 
        ax.text(x+0.006, y0-0.045, "■", transform=ax.transAxes,
                    ha="center", va="center", color=colours[0], fontsize=fontsize, fontweight="bold",
                    clip_on=False)
    else:
        for i, c in enumerate(colours):
            ax.text(x, y0 - i * 0.045, "■", transform=ax.transAxes,
                    ha="center", va="center", color=c, fontsize=fontsize, fontweight="bold",
                    clip_on=False)


def plot_raw_session(
    power_kw: np.ndarray, *, soc: np.ndarray | None = None, soc_mode: Literal["none", "raw", "simpl"] = "none",
    title: str | None = None, power_y_lim: tuple[float, float] = (0, 160), 
    figsize: Tuple[int, int] = (14, 5)) -> tuple[Figure, Axes]:
    """Plots raw power (left axis, kW) and optionally SOC (right axis, %) with colour-coded axes.

    Args:
        power_kw: Dense power in kW, shape (T,).
        soc: Optional SOC series in percent, shape (T,).
        soc_mode: "none", "raw", or "simpl" — label text only.
        title: Optional figure title.
        y_lim: Optional power y-limits (min, max). If None, uses [0, max + 20].
        dt_minutes: Minutes per time step; scales x and renames the x-axis to 'Minutes elapsed'.

    Returns:
        (fig, left_axis).
    """
    y = np.asarray(power_kw, dtype=float).reshape(-1)
    T = y.size
    x = np.arange(T, dtype=float)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # main series: raw power (orange), slightly thicker
    ax.plot(x, y, linewidth=1.8, color=COLOUR_POWER_RAW, label="Power (kW)")

    # y-limits
    ax.set_ylim(power_y_lim[0], power_y_lim[1])

    # axes, title, grid
    ax.set_xlabel("Minutes elapsed", fontdict={"size": 15})
    _axis_label_with_dual_icon(ax, "Power (kW)", colours=[COLOUR_POWER_RAW], side="left", 
                               fontsize=15)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # optional SOC on right axis (blue)
    lines = ax.get_lines()
    if soc_mode != "none" and soc is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, 100.0)
        _axis_label_with_dual_icon(ax2, "Battery charge level (%)", colours=[COLOUR_SOC], 
                                   side="right", fontsize=15)
        ax2.plot(x, np.clip(np.asarray(soc, dtype=float), 0.0, 100.0), linewidth=1.6,
                 color=COLOUR_SOC, label=f"Battery charge level (%)")
        lines = lines + ax2.get_lines()

    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, bbox_to_anchor=(0.18, -0.1),
          frameon=True, framealpha=0.9, fontsize=10)
    fig.tight_layout()
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

def plot_raw_simpl_session(
    power_raw: np.ndarray, simp_idx: np.ndarray, simp_kw: np.ndarray, *,
    t_min_eval: int, anchor_endpoints: Literal["both", "last"],
    soc: np.ndarray | None = None, soc_mode: Literal["none", "raw", "simpl"] = "none",
    title: str | None = None, power_y_lim: tuple[float, float] = (0, 140),
    figsize: Tuple[int, int] = (14, 5)
) -> tuple[Figure, Axes]:
    """Plots raw and simplified power (left axis) and optionally SOC (right axis) with strong axis-series links.

    Colours:
        - simplified power: dark red
        - raw power: black
        - SOC: blue (right axis)
    """
    from mt4xai.ors import interpolate_from_pivots
    power_raw = np.asarray(power_raw, dtype=float).reshape(-1)
    T = power_raw.size
    x = np.arange(T, dtype=float)

    simp_idx = np.asarray(simp_idx, dtype=int)
    simp_kw = np.asarray(simp_kw, dtype=float)
    power_simp_interp = interpolate_from_pivots(
        T=T, pivots=simp_idx, values=simp_kw, t_min_eval=t_min_eval, anchor_endpoints=anchor_endpoints
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # raw  + simplified
    ax.plot(x, power_raw, linewidth=1.5, alpha=0.8, linestyle="dashed",
            color=COLOUR_POWER_RAW, label="power (actual)")
    ax.plot(x, power_simp_interp, linewidth=2.2, color=COLOUR_POWER_SIMPL, alpha=0.9, 
            label="power (simplified)")

    ax.set_ylim(power_y_lim[0], power_y_lim[1])

    # axes, title, grid
    ax.set_xlabel("Minutes elapsed", fontdict={"size": 15})
    _axis_label_with_dual_icon(ax, "Power (kW)", colours=[COLOUR_POWER_RAW, COLOUR_POWER_SIMPL], side="left", fontsize=14)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # build legend candidates
    lines = ax.get_lines()

    # optional SOC (right axis)
    if soc_mode != "none" and soc is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0, 100)
        _axis_label_with_dual_icon(ax2, "Battery charge level (%)", colours=[COLOUR_SOC], side="right", fontsize=14)
        ax2.plot(x, np.clip(np.asarray(soc, dtype=float), 0.0, 100.0), linewidth=1.6, alpha=0.8,
                 color=COLOUR_SOC, label=f"battery charge level")
        lines = lines + ax2.get_lines()

    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, bbox_to_anchor=(0.18, -0.1),
          frameon=True, framealpha=0.9, fontsize=10)
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
