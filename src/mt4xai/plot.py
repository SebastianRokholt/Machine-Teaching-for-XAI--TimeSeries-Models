# src/mt4xai/plot.py
from __future__ import annotations
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.pylab import Axes
from sklearn.preprocessing import MinMaxScaler
from typing import Literal, Optional, Sequence, Tuple, List
from ray.air.result import Result

HORIZON_PREDS_PALETTE = "deep"  # high-contrast separation between horizons
COLOUR_POWER_WITH_PREDS = "black"  # with multi-horizon preds, we need black for clear separation
COLOUR_POWER_RAW  = "#f28e2b" # raw power
COLOUR_POWER_SIMPL  = "#b22222" # dark red
COLOUR_SOC = "#0b3d91" # dark blue (for raw and simplified)


# ---------------------------------------- CHARGING SESSION PLOTS ---------------------------------------- #

def _axis_label_icons_raw_simp(ax: Axes, label: str, *, colours: list[str],
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


def _axis_label_icons_raw_simp_pred(ax: Axes, label: str, *, colours: list[str],
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
        if side == "left":
            x_adj = -0.0126
        else:
            x_adj = +0.02
        for i, c in enumerate(colours):
            ax.text(x+x_adj, y0 - i * 0.045, "■", transform=ax.transAxes,
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
    _axis_label_icons_raw_simp(ax, "Power (kW)", colours=[COLOUR_POWER_RAW], side="left", 
                               fontsize=15)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # optional SOC on right axis (blue)
    lines = ax.get_lines()
    if soc_mode != "none" and soc is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, 100.0)
        _axis_label_icons_raw_simp(ax2, "Battery charge level (%)", colours=[COLOUR_SOC], 
                                   side="right", fontsize=15)
        ax2.plot(x, np.clip(np.asarray(soc, dtype=float), 0.0, 100.0), linewidth=1.6,
                 color=COLOUR_SOC, label=f"Battery charge level (%)")
        lines = lines + ax2.get_lines()

    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, bbox_to_anchor=(0.18, -0.1),
          frameon=True, framealpha=0.9, fontsize=10)
    fig.tight_layout()
    return fig, ax

def plot_raw_pred_session(
    bundle: "SessionPredsBundle",
    power_scaler: MinMaxScaler, soc_scaler: MinMaxScaler,
    idx_power_inp: int, idx_soc_inp: int,
    *, t_min_eval: int = 1, title_suffix: str = "",
    figsize: Tuple[float, float] = (12.0, 5.0), dpi: int = 110,
    y_lim_power: Optional[Tuple[float, float]] = None, y_lim_soc: Tuple[float, float] = (0.0, 100.0),
    show_points: bool = True,
) -> None:
    """Plot full-session multi-horizon **power predictions** with true power and true SOC.

    Shows:
      - left y-axis: true power (kW) and per-horizon power predictions (dashed)
      - right y-axis: true SOC (%) as a single line

    Args:
        bundle: predictions bundle with X_sample (T,Cin), P_sample (T,H,1), and unscaled truths.
        power_scaler: scaler for inverse-transforming power predictions.
        soc_scaler: scaler for inverse-transforming SOC (used only for labelling; true SOC comes from bundle).
        idx_power_inp: column index of power in X_sample.
        idx_soc_inp: column index of SOC in X_sample.
        t_min_eval: first valid source index for horizon alignment (default 1).
        title_suffix: extra text for the title.
        figsize: figure size.
        dpi: figure dpi.
        y_lim_power: optional (ymin, ymax) for power. If None, compute per session.
        y_lim_soc: y-limits for SOC in percent.
        show_points: whether to scatter the horizon samples.
    """
    T, H = bundle.length, bundle.horizon
    t = np.arange(T)

    # reconstruct absolute power predictions in scaled space: base + residuals
    base_power = bundle.X_sample[:, [idx_power_inp]].unsqueeze(1)  # (T,1,1)
    P_abs_scaled = bundle.P_sample + base_power                    # (T,H,1)

    # inverse-scale truths
    true_power = bundle.true_power_unscaled
    true_soc = bundle.true_soc_unscaled
    if true_soc is None:
        raise ValueError("bundle.true_soc_unscaled is None; ensure fetch_session_preds_bundle was called with soc_scaler/idx_soc_inp.")

    # precompute per-horizon inverse-scaled predictions
    preds_all, t_abs_all = [], []
    P_np = P_abs_scaled.detach().cpu().numpy()
    for h0 in range(H):
        i_valid = np.arange(t_min_eval, T - (h0 + 1))
        if i_valid.size == 0:
            continue
        t_abs = i_valid + (h0 + 1)
        preds_scaled = P_np[i_valid, h0, 0].reshape(-1, 1)
        preds_kw = power_scaler.inverse_transform(preds_scaled).ravel()
        preds_all.append(preds_kw)
        t_abs_all.append(t_abs)

    # auto y-limits for power
    if y_lim_power is None:
        pred_max = float(np.max(np.concatenate(preds_all))) if preds_all else 0.0
        y_max = max(float(np.max(true_power)), pred_max, 1.0)
        y_lim_power = (0.0, y_max * 1.02)

    fig, ax_p = plt.subplots(figsize=figsize, dpi=dpi)
    ax_s = ax_p.twinx()

    # true series
    sns.lineplot(x=t, y=true_power, color=COLOUR_POWER_WITH_PREDS, linewidth=2.4, label="True Power", ax=ax_p)
    sns.lineplot(x=t, y=true_soc, color=COLOUR_SOC, linewidth=1.8, label="True SOC", ax=ax_s)

    # horizons
    palette = sns.color_palette(HORIZON_PREDS_PALETTE, n_colors=H)
    for h0 in range(H):
        if h0 >= len(preds_all):
            continue
        sns.lineplot(
            x=t_abs_all[h0], y=preds_all[h0], linestyle="--", linewidth=1.7,
            color=palette[h0], label=f"H={h0+1}", ax=ax_p
        )
        if show_points:
            ax_p.scatter(t_abs_all[h0], preds_all[h0], s=10, color=palette[h0], alpha=0.6, label="_nolegend_")

    # axes styling
    ax_p.set_ylim(*y_lim_power)
    ax_s.set_ylim(*y_lim_soc)
    ax_p.autoscale(enable=False, axis="y")
    ax_p.autoscale(enable=True, axis="x")

    ax_p.set_xlabel("Time index")
    ax_p.set_ylabel("Power (kW)")
    ax_s.set_ylabel("SOC (%)")

    ax_p.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_p.grid(True, alpha=0.4)

    # single combined legend
    h1, l1 = ax_p.get_legend_handles_labels()
    h2, l2 = ax_s.get_legend_handles_labels()
    ax_p.legend(h1 + h2, l1 + l2, ncol=1, frameon=True, fontsize=9, title="Legend")

    title = f"POWER predictions — batch {bundle.batch_index}, sample {bundle.sample_index}"
    if bundle.session_id is not None:
        title += f" — Session ID: {bundle.session_id}"
    if title_suffix:
        title += f" — {title_suffix}"
    ax_p.set_title(title)

    fig.tight_layout()
    plt.show()


def plot_session_power_predictions(
    session: "ChargingSession",
    *,
    t_min_eval: int = 1,
    show_points: bool = True,
    show_soc: bool = True,                 # NEW
    figsize: tuple[float, float] = (12.0, 5.0),
    dpi: int = 110,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plots true power (kW) + multi-horizon power predictions; optional SOC on a right axis (0–100%)."""
    if session.predictions is None:
        raise ValueError("session.predictions is missing.")
    if session.scaler_power is None:
        raise ValueError("session.scaler_power is missing.")

    power = session.true_power_unscaled
    soc   = getattr(session, "true_soc_unscaled", None)
    T = int(power.size)
    H = int(session.predictions.horizon)

    # base figure and axes
    fig, ax_p = plt.subplots(figsize=figsize, dpi=dpi)
    t = np.arange(T)

    # left axis: true power
    ax_p.plot(t, power, color="black", linewidth=2.5, label="True Power")

    # multi-horizon absolute predictions (residuals → kW)
    palette = sns.color_palette("deep", n_colors=H)
    Y_res = session.predictions.Y_resid_scaled
    P_res = session.predictions.P_resid_scaled
    if not isinstance(P_res, torch.Tensor) or not isinstance(Y_res, torch.Tensor):
        raise TypeError("predictions must store torch tensors (Y_resid_scaled, P_resid_scaled).")

    for h0 in range(H):
        i_valid = np.arange(t_min_eval, T - (h0 + 1))
        if i_valid.size == 0:
            continue
        t_abs = i_valid + (h0 + 1)

        # residuals (scaled) → residuals (kW) → absolute prediction
        preds_resid_scaled = P_res[i_valid, h0, 0].detach().cpu().numpy().reshape(-1, 1)
        preds_resid_kw = session.scaler_power.inverse_transform(preds_resid_scaled).ravel()
        preds_kw = power[i_valid] + preds_resid_kw

        ax_p.plot(
            t_abs, preds_kw, linestyle="--", linewidth=1.8, color=palette[h0],
            label=f"Horizon={h0+1}", marker="o" if show_points else None, markersize=3
        )

    # optional right axis: SOC (0–100%), thin light blue
    ax_s = None
    if show_soc and soc is not None:
        ax_s = ax_p.twinx()
        ax_s.set_ylim(0.0, 100.0)
        ax_s.plot(t, np.asarray(soc, float), linewidth=1.4, alpha=0.7,
                  color="magenta", label="SOC (%)")
        ax_s.set_ylabel("SOC (%)")

    # labels, grid, ticks
    ax_p.set_xlabel("Time index")
    ax_p.set_ylabel("Power (kW)")
    ax_p.grid(True, alpha=0.35)
    ax_p.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # title
    if title is None:
        title = f"Power predictions on session {session.session_id}"
        meta = getattr(session.predictions, "meta", None)
        if meta is not None and meta.batch_index is not None and meta.sample_index is not None:
            title += f"  (batch {meta.batch_index}, sample {meta.sample_index})"
    ax_p.set_title(title)

   # combined legend (moved outside to the right)
    h1, l1 = ax_p.get_legend_handles_labels()
    if ax_s is not None:
        h2, l2 = ax_s.get_legend_handles_labels()
        handles, labels = h1 + h2, l1 + l2
    else:
        handles, labels = h1, l1

    # move legend outside the right margin
    ax_p.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.10, 0.8),
        borderaxespad=0,
        frameon=True,
        fontsize=9,
        title="Legend",
    )

    fig.tight_layout()
    return fig, ax_p


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
    _axis_label_icons_raw_simp(ax, "Power (kW)", colours=[COLOUR_POWER_RAW, COLOUR_POWER_SIMPL], side="left", fontsize=14)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # build legend candidates
    lines = ax.get_lines()

    # optional SOC (right axis)
    if soc_mode != "none" and soc is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0, 100)
        _axis_label_icons_raw_simp(ax2, "Battery charge level (%)", colours=[COLOUR_SOC], side="right", fontsize=14)
        ax2.plot(x, np.clip(np.asarray(soc, dtype=float), 0.0, 100.0), linewidth=1.6, alpha=0.8,
                 color=COLOUR_SOC, label=f"battery charge level")
        lines = lines + ax2.get_lines()

    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, bbox_to_anchor=(0.18, -0.1),
          frameon=True, framealpha=0.9, fontsize=10)
    return fig, ax


def plot_raw_pred_simp_session(
    bundle: "SessionPredsBundle",
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    simpl_power_unscaled: np.ndarray,
    simpl_soc_unscaled: Optional[np.ndarray] = None,
    session_id: Optional[int | str] = None,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    simp_error: Optional[float] = None,
    orig_error: Optional[float] = None,
    label: Optional[str] = None,
    decay_lambda: Optional[float] = None,
    noise_std_kw: Optional[float] = None,
    robust_tau: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 4),
    dpi: int = 110,
    t_min_eval: int = 1,
) -> tuple[Figure, Axes]:
    """
    plots true power, multi-horizon power predictions and an ORS simplification,
    with an optional ORS-RDP SOC simplification on a right-hand axis.

    this function uses `reconstruct_abs_from_bundle` to obtain absolute
    predictions in kW. soc is only plotted in its simplified form, to avoid
    clutter and duplicate legend entries.
    """
    from .data import reconstruct_abs_from_bundle  # avoid circular import

    T, H = bundle.length, bundle.horizon
    t = np.arange(T)
    true_power = np.asarray(bundle.true_power_unscaled, dtype=float).reshape(-1)
    sid = int(session_id) if session_id is not None else bundle.session_id

    # reconstruct absolute power predictions
    P_abs, _ = reconstruct_abs_from_bundle(
        bundle=bundle,
        power_scaler=power_scaler,
        idx_power_inp=idx_power_inp,
    )
    if P_abs.ndim == 3:
        P_pow = P_abs[..., 0]  # [T, H, 1] -> [T, H]
    else:
        P_pow = P_abs  # [T, H]

    # figure and left axis (power)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # true power (kW)
    sns.lineplot(
        x=t,
        y=true_power,
        color=COLOUR_POWER_WITH_PREDS,
        linewidth=2.2,
        label="True power",
        ax=ax,
    )

    # multi-horizon predictions
    palette = sns.color_palette(HORIZON_PREDS_PALETTE, n_colors=H)
    for h0 in range(H):
        i_valid = np.arange(t_min_eval, T - (h0 + 1))
        if i_valid.size == 0:
            continue
        t_pred = i_valid + (h0 + 1)
        preds = P_pow[i_valid, h0]
        sns.lineplot(
            x=t_pred,
            y=preds,
            linestyle="--",
            linewidth=1.6,
            color=palette[h0],
            label=f"H={h0+1}",
            marker="o",
            markersize=2.5,
            ax=ax,
        )

    # ors simplification (already in kW)
    simpl_power_unscaled = np.asarray(simpl_power_unscaled, dtype=float).reshape(-1)
    if simpl_power_unscaled.size != T:
        raise ValueError("simpl_power_unscaled must have length equal to session length T.")
    sns.lineplot(
        x=t,
        y=simpl_power_unscaled,
        color=COLOUR_POWER_SIMPL,
        linewidth=2.6,
        label="ORS simplification (power)",
        ax=ax,
    )

    # left y-axis (power)
    ax.set_xlabel("Minutes elapsed")
    _axis_label_icons_raw_simp_pred(
        ax,
        "Power (kW)",
        colours=[COLOUR_POWER_WITH_PREDS, COLOUR_POWER_SIMPL],
        side="left",
        fontsize=12,
    )

    # optional SOC ORS-RDP overlay on right axis
    ax2: Optional[Axes] = None
    if simpl_soc_unscaled is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, 100.0)
        _axis_label_icons_raw_simp_pred(
            ax2,
            "Battery charge level (%)",
            colours=[COLOUR_SOC],
            side="right",
            fontsize=12,
        )

        simpl_soc_unscaled = np.asarray(simpl_soc_unscaled, dtype=float).reshape(-1)
        if simpl_soc_unscaled.size != T:
            raise ValueError("simpl_soc_unscaled must have length equal to session length T.")

        ax2.plot(
            t,
            np.clip(simpl_soc_unscaled, 0.0, 100.0),
            linewidth=1.8,
            color=COLOUR_SOC,
            label="SOC (ORS-RDP)",
        )

    # title
    ttl = f"ORS simplification of charging session {sid}"
    if label is not None:
        ttl += f" — classification: {label}"
    if k is not None:
        ttl += f" (k={k})"
    ax.set_title(ttl)

    # caption (kept short so layout is stable)
    cap_parts: list[str] = []
    if orig_error is not None:
        cap_parts.append(f"orig macro-RMSE={orig_error:.3f}")
    if simp_error is not None:
        cap_parts.append(f"simpl macro-RMSE={simp_error:.3f}")
    if threshold is not None:
        cap_parts.append(f"threshold={float(threshold):.3f}")
    if decay_lambda is not None:
        cap_parts.append(f"λ={float(decay_lambda):.3f}")
    if noise_std_kw is not None:
        cap_parts.append(f"noise σ={float(noise_std_kw):.3f} kW")
    if robust_tau is not None:
        cap_parts.append(f"robust τ={float(robust_tau):.3f}")
    if cap_parts:
        ax.text(
            0.01,
            -0.16,
            " | ".join(cap_parts),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    # collect all lines (including SOC on ax2)
    lines = list(ax.get_lines())
    labels = [ln.get_label() for ln in lines]

    if ax2 is not None:
        ax2_lines = list(ax2.get_lines())
        ax2_labels = [ln.get_label() for ln in ax2_lines]
        lines.extend(ax2_lines)
        labels.extend(ax2_labels)

    # place legend outside the plot area
    fig.legend(
        lines,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.07, 0.94),   # near top-right corner outside of the the axes
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    # make room on the right so legend never squashes the plot
    fig.subplots_adjust(right=0.80)

    return fig, ax


def plot_inputs_to_single_output_grid(
    session: "ChargingSession",
    i_list: list[int],
    horizon: int = 1,                      # 1..H
    features_to_show: Optional[list[str]] = None,   # ["power"] by default
    window_len: Optional[int] = 30,        # last N steps up to i0; None = from start
    ncols: int = 3,
    right_pad_steps: float = 1.8,          # extra x padding to the right
    annotate: bool = True,
    *, batch_index: Optional[int] = None, sample_index: Optional[int] = None
) -> Tuple[Figure, np.ndarray]:
    """
    plots unscaled power inputs vs a single unscaled absolute power output at horizon h.

    computes y_pred(kW) = power_kw[i0] + inverse_transform(residual_pred_scaled).
    the function expects `session.predictions` with scaled residual tensors on cpu and
    uses `session.scaler_power` to invert residuals.

    args:
        session: charging session with predictions attached by `fetch_session_preds_bundle`.
        i_list: source indices i0 where we display a point at t = i0 + h.
        horizon: 1-indexed prediction horizon h to show.
        features_to_show: context inputs to draw as lines; defaults to ["power"].
        window_len: window ending at i0 (inclusive); None = from start.
        ncols: grid columns.
        right_pad_steps: x-axis padding.
        annotate: add numeric labels next to markers.
        batch_index, sample_index: optional metadata for titles.

    returns:
        (fig, axes)
    """
    if session.predictions is None:
        raise ValueError("session.predictions is None; attach model outputs before plotting.")

    H = int(session.predictions.horizon)
    if H < 1:
        raise ValueError("invalid session.horizon; expected ≥ 1.")
    h = max(1, min(int(horizon), H))

    # base series in original units
    power = session.true_power_unscaled
    T = int(power.size)
    power_scaler = getattr(session, "scaler_power", None)
    if power_scaler is None:
        raise ValueError("session.scaler_power is missing.")

    # valid i0 range [start_valid, end_valid_excl) so i0 + h in [0, T-1]
    start_valid, end_valid_excl = 1, max(0, T - H)
    if not i_list:
        raise ValueError("i_list cannot be empty.")
    clamped = []
    for i0 in i_list:
        i1 = min(max(int(i0), start_valid), end_valid_excl - 1)
        if start_valid <= i1 < end_valid_excl:
            clamped.append(i1)
    if not clamped:
        raise ValueError(f"no valid i0 in i_list. valid range is 1 .. {T - H - 1}.")

    # choose context features (power only now)
    if features_to_show is None:
        features_to_show = ["power"]
    features_to_show = [f for f in features_to_show if f == "power"]

    Y_resid = session.predictions.Y_resid_scaled   # (T,H,C) on cpu
    P_resid = session.predictions.P_resid_scaled   # (T,H,C) on cpu
    if not isinstance(P_resid, torch.Tensor) or not isinstance(Y_resid, torch.Tensor):
        raise TypeError("predictions must hold torch tensors.")

    # layout
    n = len(clamped)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.2 * nrows), squeeze=False)

    for ax, i0 in zip(axes.ravel(), clamped):
        # window [w0, w1)
        if window_len is None:
            w0, w1 = 0, i0 + 1
        else:
            w0 = max(0, i0 - int(window_len) + 1)
            w1 = i0 + 1
        tt = np.arange(w0, w1)

        # context line(s)
        ymins, ymaxs = [], []
        if "power" in features_to_show:
            seg = power[w0:w1]
            ax.plot(tt, seg, label="power", linewidth=2)
            ymins.append(float(np.min(seg))); ymaxs.append(float(np.max(seg)))

        # single-horizon point at t_pred = i0 + h
        t_pred = i0 + h
        if 0 <= t_pred < T:
            # residuals (scaled) → residuals (unscaled)
            resid_pred_scaled = (
                P_resid[i0, h - 1, 0].detach().unsqueeze(0).cpu().numpy().reshape(-1, 1)
            )
            resid_true_scaled = (
                Y_resid[i0, h - 1, 0].detach().unsqueeze(0).cpu().numpy().reshape(-1, 1)
            )
            resid_pred_kw = power_scaler.inverse_transform(resid_pred_scaled).ravel()[0]
            resid_true_kw = power_scaler.inverse_transform(resid_true_scaled).ravel()[0]
            base_kw = float(power[i0])

            y_pred = base_kw + resid_pred_kw
            y_true = base_kw + resid_true_kw

            ax.scatter([t_pred], [y_pred], s=120, marker="x", linewidths=2.5,
                       color="tab:red", zorder=5, label="Pred")
            ax.scatter([t_pred], [y_true], s=120, marker="x", linewidths=2.5,
                       color="tab:blue", zorder=5, label="True")

            if annotate:
                ax.annotate(f"{y_pred:.2f}", (t_pred, y_pred),
                            textcoords="offset points", xytext=(6, 8),
                            fontsize=8, color="tab:red")
                ax.annotate(f"{y_true:.2f}", (t_pred, y_true),
                            textcoords="offset points", xytext=(6, -14),
                            fontsize=8, color="tab:blue")

            ymins.append(min(y_pred, y_true)); ymaxs.append(max(y_pred, y_true))

        # bounds + cosmetics
        if ymins and ymaxs:
            y_lo, y_hi = min(ymins), max(ymaxs)
            span = max(1e-9, y_hi - y_lo)
            pad = 0.08 * span if span > 0 else 0.5
            ax.set_ylim(y_lo - pad, y_hi + pad)

        ax.axvline(i0, color="black", linestyle=":", linewidth=1.8)
        ax.set_xlim(min(w0, t_pred), max(w1 - 1, t_pred) + right_pad_steps)
        title_suffix = ""
        if batch_index is not None and sample_index is not None:
            title_suffix = f" | batch {batch_index}, sample {sample_index}"
        ax.set_title(f"i₀={i0}, H={h} → t={t_pred}{title_suffix}")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Power (kW)")
        ax.grid(True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="upper left")

    # hide any unused axes
    for k in range(len(clamped), nrows * ncols):
        axes.ravel()[k].set_axis_off()

    fig.suptitle(f"Inputs → Single POWER output @ horizon H={h} (shared axis)", y=1.02, fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_grid_power_predictions(
    sessions: Sequence["ChargingSession"],
    *,
    t_min_eval: int = 1,
    show_points: bool = True,
    show_soc: bool = True,
    figsize_per_cell: tuple[float, float] = (5.6, 3.8),
) -> Tuple[Figure, np.ndarray]:
    """Plot true power, multi-horizon predictions, and optional SOC for multiple sessions.
    
    Each column is a session; each row is a horizon h=1..H.
    Shows true power (black), multi-horizon predictions (dashed colours), and SOC (magenta, right y-axis).
    """
    if not sessions:
        raise ValueError("sessions cannot be empty")
    if any(s.predictions is None for s in sessions):
        raise ValueError("all sessions must have predictions attached")
    if any(s.scaler_power is None for s in sessions):
        raise ValueError("all sessions must define scaler_power")

    # ensure all sessions share same horizon
    H = int(sessions[0].predictions.horizon)
    if any(int(s.predictions.horizon) != H for s in sessions):
        raise ValueError("all sessions must share the same horizon H")

    ncols = len(sessions)
    fig_w = max(1, ncols) * figsize_per_cell[0]
    fig_h = max(1, H) * figsize_per_cell[1]
    fig, axes = plt.subplots(
        nrows=H, ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    # spacing tuned for readable y-labels and top-right legend
    fig.subplots_adjust(
        hspace=0.35,
        wspace=0.45,
        left=0.08,
        right=0.88,
        top=0.93,
        bottom=0.08,
    )

    palette = sns.color_palette("deep", n_colors=H)

    # accumulate legend handles
    legend_handles = {}
    for col, sess in enumerate(sessions):
        power = sess.true_power_unscaled
        soc   = sess.true_soc_unscaled
        T     = int(power.size)
        t_all = np.arange(T)

        for h0 in range(H):
            ax: Axes = axes[h0, col]
            h1 = h0 + 1

            # true power
            h_true, = ax.plot(t_all, power, color="black", linewidth=2, label="True Power")
            legend_handles["True Power"] = h_true

            ax.set_xlabel("Time index")
            ax.set_ylabel("Power (kW)")
            ax.grid(True, alpha=0.35)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # valid emission indices
            i_valid = np.arange(t_min_eval, T - h1)
            if i_valid.size > 0:
                t_pred = i_valid + h1
                P_res = sess.predictions.P_resid_scaled
                if not isinstance(P_res, torch.Tensor):
                    raise TypeError("predictions.P_resid_scaled must be a torch tensor")

                resid_scaled = P_res[i_valid, h0, 0].detach().cpu().numpy().reshape(-1, 1)
                resid_kw = sess.scaler_power.inverse_transform(resid_scaled).ravel()
                preds_kw = power[i_valid] + resid_kw

                h_pred, = ax.plot(
                    t_pred, preds_kw,
                    linestyle="--",
                    linewidth=1.8,
                    color=palette[h0],
                    marker="o" if show_points else None,
                    markersize=3,
                    label=f"H={h1}",
                )
                legend_handles[f"H={h1}"] = h_pred

            # SOC overlay (magenta curve only)
            if show_soc and soc is not None:
                ax_r = ax.twinx()
                ax_r.set_ylim(0.0, 100.0)
                h_soc, = ax_r.plot(
                    t_all, np.asarray(soc, float),
                    linewidth=1.2,
                    color="magenta",
                    label="SOC (%)",
                )
                legend_handles["SOC (%)"] = h_soc
                ax_r.set_ylabel("SOC (%)")
                if h0 != 0:
                    ax_r.set_yticklabels([])

            # titles per column on the top row
            if h0 == 0:
                meta = getattr(sess.predictions, "meta", None)
                if meta is not None and meta.batch_index is not None and meta.sample_index is not None:
                    title = f"batch {meta.batch_index}, sample {meta.sample_index}"
                else:
                    title = f"session {sess.session_id}"
                ax.set_title(title)

    # unified legend: all horizons + True Power + SOC
    handles = list(legend_handles.values())
    labels  = list(legend_handles.keys())
    fig.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.3,
        frameon=True,
        fontsize=10,
        title="Legend",
    )

    fig.suptitle("Power predictions across multiple sessions (rows=horizons, cols=sessions)", y=0.995, fontsize=20)
    return fig, axes


# ------------------------------------------------- LOSS PLOTS ---------------------------------------------------- #

def plot_losses_from_result(trial_result: Result, title_prefix="") -> None:
    """Plots the training and validation loss curves from a trial result"""
    if trial_result is None:
        print("[plot] No best_result to plot."); return
    hist = trial_result.metrics_dataframe.copy()
    time_col = "time_total_s" if "time_total_s" in hist.columns else (
        "timestamp" if "timestamp" in hist.columns else "epoch")
    hist = (hist.sort_values(["epoch", time_col])
                .drop_duplicates(subset="epoch", keep="last")
                .sort_values("epoch"))
    hist["epoch"] = hist["epoch"].astype(int)

    plt.figure(figsize=(9,5))
    plt.plot(hist["epoch"], hist["train_loss"], label="Train Loss (Huber, orig units)", marker="o")
    plt.plot(hist["epoch"], hist["val_loss"],   label="Val Loss (Huber, orig units)",   marker="o")
    plt.ylabel("SmoothL1 (Huber)"); plt.xlabel("Epoch")
    plt.title(f"{title_prefix} Train/Val Loss"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9,5))
    plt.plot(hist["epoch"], hist["val_metric"], label="val_metric (macro-RMSE)", marker="o")
    if "val_rmse_power" in hist.columns: plt.plot(hist["epoch"], hist["val_rmse_power"], label="RMSE Power (kW)", marker="o")
    if "val_rmse_soc"   in hist.columns: plt.plot(hist["epoch"], hist["val_rmse_soc"],   label="RMSE SOC (%)",   marker="o")
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    plt.title(f"{title_prefix} Validation RMSE"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()