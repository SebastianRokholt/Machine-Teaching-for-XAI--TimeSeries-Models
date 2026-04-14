# mt4xai/ors.py
# Portions adapted from: https://github.com/BrigtHaavardstun/kSimplification (MIT License)
# by Brigt Håvardstun. The ORS algorithm was proposed in the 2024 research paper 
# "Optimal Robust Simplifications for Explaining Time Series Classifications" (2024) 
# by Brigt Håvardstun, Cèsar Ferri and Jan Arne Telle.

# ----------------------------------------- imports, config and outputs ----------------------------------------- #
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal, Optional, List, Tuple, Dict, Iterable
import heapq
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from rdp import rdp

from .data import reconstruct_abs_from_bundle, SessionPredsBundle
from .inference import predict_residuals



Stage1Mode = Literal["dp", "dp_prefix", "dp_prefix_v3", "rdp"]


@dataclass
class ORSParams:
    """Stores configuration for Optimal Robust Simplification (ORS).

    The stage-1 generator supports three DP variants:

    - ``"dp"`` runs vanilla DP table construction and legacy ranking.
    - ``"dp_prefix"`` runs the legacy prefix implementation kept for benchmarking.
    - ``"dp_prefix_v3"`` runs the improved prefix implementation and is the default.
    - ``"rdp"`` runs the heuristic RDP candidate generator.

    Attributes:
        stage1_mode: Stage-1 generator mode.
        stage2_err_metric: Stage-2 error metric name.
        dp_q: Number of top-ranked DP candidates kept in the heap.
        dp_alpha: Stage-1 weight for L2 error in ``alpha * err + beta * k``.
        rdp_stage1_candidates: Number of epsilon probes for Stage-1 RDP.
        beta: Weight of segment count in objective.
        gamma: Weight of fragility in objective.
        R: Number of robustness perturbation samples.
        epsilon_mode: Epsilon mode for robustness perturbation.
        epsilon_value: Epsilon value in the selected mode.
        random_seed: Random seed used in robustness sampling.
        min_k: Lower segment-count bound for valid candidates.
        max_k: Upper segment-count bound for valid candidates.
        t_min_eval: First timestep included in classification evaluation.
        anchor_endpoints: Endpoint policy for interpolation.
        model_id: Optional model identifier used in metadata.
        soc_stage1_mode: SOC simplification Stage-1 mode.
        soc_rdp_epsilon: SOC RDP epsilon value.
        soc_rdp_candidates: SOC RDP candidate count.
        soc_rdp_eps_min: SOC RDP minimum epsilon.
        soc_rdp_eps_max: SOC RDP maximum epsilon.
    """
    stage1_mode: Stage1Mode = "dp_prefix_v3"
    stage2_err_metric: str = "l2"
    dp_q: int = 500
    dp_alpha: float = 0.01
    rdp_stage1_candidates: int = 50
    beta: float = 3.0
    gamma: float = 0.05
    R: int = 10000
    epsilon_mode: str = "fraction"
    epsilon_value: float = 0.3
    random_seed: Optional[int] = 42
    min_k: int = 1
    max_k: int = 100
    t_min_eval: int = 0
    anchor_endpoints: Literal["both", "last"] = "last"
    model_id: Optional[str] = None
    # SOC with RDP specific params
    soc_stage1_mode: str | None = "rdp"
    soc_rdp_epsilon: float | None = 0.75
    soc_rdp_candidates: int = 5       # parsed, not used by fixed epsilon path yet
    soc_rdp_eps_min: float = 1.0e-6   # parsed, future-proof
    soc_rdp_eps_max: float = 100.0


@dataclass
class ORSOutcome:
    """Diagnostics for an evaluated candidate."""
    k_opt: int
    keep_mask: np.ndarray
    simplified_power: np.ndarray
    robust_prob: float
    simplified_label: str
    simplified_error: float
    l2_err: float
    objective: float


# ----------------------------------------- macro-rmse / predictions -------------------------------------- #
# TODO: Move these to inference.py 

def build_true_abs_from_series(power_true: np.ndarray, H: int) -> np.ndarray:
    """build absolute targets Y_abs[t, h] aligned with horizon h in original units.

    for each horizon h (1-based), Y_abs[t, h-1] corresponds to the true power
    at time t + h. entries beyond the available range are left at zero and
    do not contribute to macro-RMSE once t_min_eval is applied.

    args:
        power_true: [T] array of true power values in kW.
        H: prediction horizon (number of steps ahead).

    returns:
        Y_abs: [T, H] array of absolute targets.
    """
    T = power_true.size
    Y = np.zeros((T, H), dtype=float)
    for h0 in range(H):
        h1 = h0 + 1
        end = T - h1
        if end <= 0:
            break
        Y[:end, h0] = power_true[h1 : h1 + end]
    return Y


def macro_rmse_from_abs(
    P_abs: np.ndarray,
    Y_abs: np.ndarray,
    decay_lambda: float,
    t_min_eval: int,
) -> float:
    """computes macro-RMSE across horizons with exponential decay, in original units.

    supports both single-channel and multi-channel targets:

      - if P_abs has shape [T, H], it is treated as a single channel.
      - if P_abs has shape [T, H, C], errors are aggregated over channels.

    args:
        P_abs: absolute predictions [T, H] or [T, H, C].
        Y_abs: absolute targets with the same shape.
        decay_lambda: non-negative decay parameter λ for horizon weights.
        t_min_eval: minimum time index (inclusive) from which errors contribute.

    returns:
        scalar macro-RMSE value aggregated over horizons.
    """
    if P_abs.ndim == 2:
        P = P_abs[..., np.newaxis]
        Y = Y_abs[..., np.newaxis]
    elif P_abs.ndim == 3:
        P, Y = P_abs, Y_abs
    else:
        raise ValueError(
            f"P_abs and Y_abs must have shape (T, H) or (T, H, C), got {P_abs.shape}"
        )

    T, H, _ = P.shape

    # horizon weights w_h ∝ exp(-λ h)
    w_h = np.exp(-float(decay_lambda) * np.arange(H, dtype=float))
    w_h = w_h / np.sum(w_h)

    vals: list[float] = []
    for h in range(H):
        end = T - (h + 1)
        if end <= t_min_eval:
            continue

        diff = P[t_min_eval:end, h, :] - Y[t_min_eval:end, h, :]
        rmse_h = float(np.sqrt(np.mean(diff**2)))
        vals.append(rmse_h)

    if not vals:
        return 0.0

    vals_arr = np.asarray(vals, dtype=float)
    return float(np.sum(vals_arr * w_h[: len(vals_arr)]))


@torch.inference_mode()
def macro_rmse_for_power_batch(
    bundle: SessionPredsBundle,
    model: torch.nn.Module,
    power_batch_kw: np.ndarray,
    *,
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    decay_lambda: float,
    t_min_eval: int,
    Y_abs_true: np.ndarray,
) -> np.ndarray:
    """run a batched forward pass for perturbed power series and return macro-RMSE per perturbation.

    args:
        bundle: reference SessionPredsBundle (provides X_sample, length, horizon).
        model: trained residual forecaster.
        power_batch_kw: [R, T] batch of perturbed absolute power curves in kW.
        power_scaler: scaler for power (for inverse transform).
        idx_power_inp: index of power feature in X_sample.
        decay_lambda: macro-RMSE horizon decay parameter.
        t_min_eval: minimum time index from which errors contribute.
        Y_abs_true: [T, H] absolute target path built from the *original* power series.

    returns:
        errs: [R] macro-RMSE values for each perturbed curve.
    """
    T = bundle.length
    R, T_in = power_batch_kw.shape
    if T_in != T:
        raise ValueError(f"power_batch_kw has T={T_in}, expected {T}")

    # build scaled input batch with perturbed power
    X_base = bundle.X_sample.clone()  # [T, F] scaled
    p_scaled = power_scaler.transform(power_batch_kw.reshape(-1, 1)).reshape(R, T)
    Xs = X_base.unsqueeze(0).repeat(R, 1, 1).contiguous()  # [R, T, F]
    Xs[:, :, idx_power_inp] = torch.from_numpy(p_scaled.astype(np.float32)).to(Xs)

    lengths = torch.full((R,), T, dtype=torch.long)
    device = next(model.parameters()).device
    P_res = predict_residuals(model, Xs.to(device), lengths, device=device)  # [R, T, H, 1]

    # reconstructs the absolute kW predictions
    # we reuse the bundle logic but now in batched form
    P_abs = []
    for r in range(R):
        b_tmp = SessionPredsBundle(
            X_sample=Xs[r].detach().cpu(),
            P_sample=P_res[r].detach().cpu(),
            Y_sample=bundle.Y_sample, # targets do not depend on perturbation
            length=bundle.length,
            horizon=bundle.horizon,
            session_id=bundle.session_id,
            true_power_unscaled=bundle.true_power_unscaled,
            true_soc_unscaled=bundle.true_soc_unscaled,
            batch_index=bundle.batch_index,
            sample_index=bundle.sample_index,
            num_targets=bundle.num_targets,
        )
        P_r, _ = reconstruct_abs_from_bundle(
            bundle=b_tmp,
            power_scaler=power_scaler,
            idx_power_inp=idx_power_inp,
        )
        # P_r is [T, H] or [T, H, C], so take power channel if needed
        P_abs.append(P_r[..., 0] if P_r.ndim == 3 else P_r)

    P_abs_arr = np.stack(P_abs, axis=0)  # [R, T, H]

    errs = np.zeros(R, dtype=float)
    for r in range(R):
        errs[r] = macro_rmse_from_abs(
            P_abs_arr[r], Y_abs_true, decay_lambda=decay_lambda, t_min_eval=t_min_eval
        )
    return errs


def classify_macro_rmse_from_power(
    bundle: SessionPredsBundle,
    model: torch.nn.Module,
    power_kw: np.ndarray,
    *,
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    decay_lambda: float,
    t_min_eval: int,
    Y_abs_true: np.ndarray,
    threshold: float,
) -> Tuple[int, float]:
    """classify a power curve by macro-RMSE threshold; return (label_int, err).
    label_int is 1 for "abnormal" and 0 for "normal".
    """
    errs = macro_rmse_for_power_batch(
        bundle,
        model,
        power_kw[None, :],
        power_scaler=power_scaler,
        idx_power_inp=idx_power_inp,
        decay_lambda=decay_lambda,
        t_min_eval=t_min_eval,
        Y_abs_true=Y_abs_true,
    )
    err = float(errs[0])
    return (1 if err > float(threshold) else 0), err


def base_label_from_bundle(
    bundle: SessionPredsBundle,
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    decay_lambda: float,
    t_min_eval: int,
    threshold: float,
) -> tuple[str, float, np.ndarray]:
    """computes baseline macro-RMSE and label for an unperturbed session bundle.

    it reconstructs absolute predictions and targets from the residual forecasts,
    computes macro-RMSE with exponential horizon decay, then classifies the session
    as normal or abnormal by comparing the error to a fixed threshold.

    returns:
        label: "normal" or "abnormal" based on the threshold.
        err: macro-RMSE for the original (unsimplified) session.
        Y_abs_true: [T, H] absolute target path for reuse in ORS.
    """
    # reconstruct absolute predictions and targets
    P_abs, Y_abs = reconstruct_abs_from_bundle(
        bundle=bundle,
        power_scaler=power_scaler,
        idx_power_inp=idx_power_inp,
    )

    # flatten to [T, H] for power-only targets
    P_pow = P_abs[..., 0]
    Y_pow = Y_abs[..., 0]

    err = macro_rmse_from_abs(P_pow, Y_pow, decay_lambda=decay_lambda, t_min_eval=t_min_eval)
    label = "abnormal" if err >= float(threshold) else "normal"

    # reuse targets as ground truth for perturbation-based evaluation
    return label, err, Y_pow


# ----------------------------------------- stage-1: dp and rdp ---------------------------------------- #

# internal stage-1 candidate tuple:
# (stage1_cost_es, pivot_indices, optional_pivot_values)
# optional_pivot_values is used for candidates that cannot be represented
# faithfully by y[pivots], such as both-ends single-line candidates.
Stage1Candidate = Tuple[float, np.ndarray, Optional[np.ndarray]]

def precompute_prefix_sums(y: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Precompute prefix sums for O(1) segment SSE queries (DP-prefix)."""
    t = np.arange(len(y), dtype=float)
    S1 = np.cumsum(np.ones_like(t))
    Sy = np.cumsum(y)
    St = np.cumsum(t)
    St2 = np.cumsum(t * t)
    Sty = np.cumsum(t * y)
    Sy2 = np.cumsum(y * y)
    return S1, Sy, St, St2, Sty, Sy2


def seg_error_for_line(a: float, b: float, L: int, R: int, S: Tuple[np.ndarray, ...]) -> float:
    """Return SSE of segment [L..R] for line a*t+b using prefix sums."""
    S1, Sy, St, St2, Sty, Sy2 = S
    n = S1[R] - (S1[L - 1] if L > 0 else 0.0)
    sy = Sy[R] - (Sy[L - 1] if L > 0 else 0.0)
    st = St[R] - (St[L - 1] if L > 0 else 0.0)
    st2 = St2[R] - (St2[L - 1] if L > 0 else 0.0)
    sty = Sty[R] - (Sty[L - 1] if L > 0 else 0.0)
    sy2 = Sy2[R] - (Sy2[L - 1] if L > 0 else 0.0)
    return sy2 - 2 * a * sty - 2 * b * sy + (a * a) * st2 + 2 * a * b * st + n * (b * b)


def line_through(i: int, yi: float, j: int, yj: float) -> Tuple[float, float]:
    """Return slope, intercept of line through points (i,yi) and (j,yj)."""
    a = (yj - yi) / (j - i)
    b = yi - a * i
    return a, b


def build_error_tables_with_prefix(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build err, err_left, err_right, err_all via prefix sums (DP-prefix)."""
    n = len(y)
    S = precompute_prefix_sums(y)

    err = np.zeros((n, n), dtype=float)
    errL = np.zeros((n, n), dtype=float)
    errR = np.zeros((n, n), dtype=float)
    errA = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = line_through(i, y[i], j, y[j])
            err[i, j] = seg_error_for_line(a, b, i, j, S)
            # extended left [0..j]
            errL[i, j] = seg_error_for_line(a, b, 0, j, S)
            # extended right [i..n-1]
            errR[i, j] = seg_error_for_line(a, b, i, n - 1, S)
            # extended both ends [0..n-1]
            errA[i, j] = seg_error_for_line(a, b, 0, n - 1, S)
    return err, errL, errR, errA


def build_error_tables_with_prefix_legacy(
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build err tables with the legacy prefix implementation.

    This function keeps the historical v2 behaviour for benchmarking:
    ``err`` uses prefix sums, while ``errL``, ``errR``, and ``errA`` use direct
    summation over dense windows.

    Args:
        y: One-dimensional signal of length ``n``.

    Returns:
        Tuple of ``(err, errL, errR, errA)`` tables with shape ``(n, n)``.
    """
    n = len(y)
    t = np.arange(n, dtype=float)
    S = precompute_prefix_sums(y)

    err = np.zeros((n, n), dtype=float)
    errL = np.zeros((n, n), dtype=float)
    errR = np.zeros((n, n), dtype=float)
    errA = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = line_through(i, y[i], j, y[j])
            err[i, j] = seg_error_for_line(a, b, i, j, S)
            yhat = a * t[: j + 1] + b
            errL[i, j] = float(np.sum((y[: j + 1] - yhat) ** 2))
            yhat = a * t[i:] + b
            errR[i, j] = float(np.sum((y[i:] - yhat) ** 2))
            yhat = a * t + b
            errA[i, j] = float(np.sum((y - yhat) ** 2))
    return err, errL, errR, errA


def build_error_tables_no_prefix(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build err tables by direct summation (vanilla DP O(n^3) build)."""
    n = len(y)
    t = np.arange(n, dtype=float)

    err  = np.full((n, n), 0.0, dtype=float)
    errL = np.full((n, n), 0.0, dtype=float)
    errR = np.full((n, n), 0.0, dtype=float)
    errA = np.full((n, n), 0.0, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            a = (y[j] - y[i]) / (j - i)
            b = y[i] - a * i
            # segment [i..j]
            yhat = a * t[i:j+1] + b
            err[i, j] = float(np.sum((y[i:j+1] - yhat)**2))
            # extend to left
            yhatL = a * t[:j+1] + b
            errL[i, j] = float(np.sum((y[:j+1] - yhatL)**2))
            # extend to right
            yhatR = a * t[i:] + b
            errR[i, j] = float(np.sum((y[i:] - yhatR)**2))
            # extend both
            yhatA = a * t + b
            errA[i, j] = float(np.sum((y - yhatA)**2))
    return err, errL, errR, errA


def _dp_with_heaps_and_ranks_legacy(
    err: np.ndarray,
    errL: np.ndarray,
    errR: np.ndarray,
    errA: np.ndarray,
    alpha: float,
    beta: float,
    q: int,
) -> List[Tuple[float, np.ndarray]]:
    """Run the historical Stage-1 DP heap ranking.

    This implementation keeps the original v1 and v2 behaviour, including
    pseudo-pivot both-ends candidates represented as ``[0, k, n-1]``.
    """
    n = err.shape[0]
    if n < 2:
        return [(0.0, np.array([0], dtype=int))]

    D: List[List[Tuple[float, np.ndarray]]] = [[] for _ in range(n)]
    uid = 0

    def push_Hj(Hj, cost: float, i: int, r: int, piv: np.ndarray) -> None:
        nonlocal uid
        uid += 1
        heapq.heappush(Hj, (float(cost), uid, int(i), int(r), tuple(int(x) for x in piv)))

    def push_Hn(Hn, cost: float, i: int, r: int, piv: np.ndarray) -> None:
        nonlocal uid
        uid += 1
        heapq.heappush(Hn, (float(cost), uid, int(i), int(r), tuple(int(x) for x in piv)))

    for j in range(1, n):
        Hj: List[Tuple[float, int, int, int, Tuple[int, ...]]] = []
        cost0 = alpha * errL[0, j] + beta
        push_Hj(Hj, cost0, 0, 0, np.array([0, j], dtype=int))

        for i in range(1, j):
            if not D[i]:
                continue
            prev_cost, prev_piv = D[i][0]
            cost = prev_cost + alpha * err[i, j] + beta
            push_Hj(Hj, cost, i, 1, np.append(prev_piv, j))

        while Hj and len(D[j]) < q:
            val, _, i, r, piv_t = heapq.heappop(Hj)
            piv = np.array(piv_t, dtype=int)
            D[j].append((float(val), piv))

            if r >= 1 and r < len(D[i]):
                next_prev_cost, next_prev_piv = D[i][r]
                nxt_cost = next_prev_cost + alpha * err[i, j] + beta
                push_Hj(Hj, nxt_cost, i, r + 1, np.append(next_prev_piv, j))

    Hn: List[Tuple[float, int, int, int, Tuple[int, ...]]] = []
    for i in range(1, n - 1):
        if not D[i]:
            continue
        prev_cost, prev_piv = D[i][0]
        cost = prev_cost + alpha * errR[i, n - 1] + beta
        push_Hn(Hn, cost, i, 1, np.append(prev_piv, n - 1))

    for i in range(0, n - 1):
        for k in range(i + 1, n):
            cost = alpha * errA[i, k] + beta
            push_Hn(Hn, cost, -1, 0, np.array([0, k, n - 1], dtype=int))

    finals: List[Tuple[float, np.ndarray]] = []
    seen: set[Tuple[int, ...]] = set()
    while Hn and len(finals) < q:
        val, _, i, r, piv_t = heapq.heappop(Hn)
        piv = np.array(piv_t, dtype=int)

        piv = np.unique(np.clip(piv, 0, n - 1))
        if piv[0] != 0:
            piv = np.insert(piv, 0, 0)
        if piv[-1] != n - 1:
            piv = np.append(piv, n - 1)

        key = tuple(piv.tolist())
        if key not in seen:
            finals.append((float(val), piv))
            seen.add(key)

        if i >= 1 and r < len(D[i]):
            next_prev_cost, next_prev_piv = D[i][r]
            nxt_cost = next_prev_cost + alpha * errR[i, n - 1] + beta
            push_Hn(Hn, nxt_cost, i, r + 1, np.append(next_prev_piv, n - 1))

    finals.sort(key=lambda x: x[0])
    return finals[:q]


def _dp_with_heaps_and_ranks_v3(
    y: np.ndarray,
    err: np.ndarray,
    errL: np.ndarray,
    errR: np.ndarray,
    errA: np.ndarray,
    alpha: float,
    beta: float,
    q: int,
) -> List[Stage1Candidate]:
    """Run Stage-1 DP heap ranking with cost-consistent both-ends candidates."""
    n = err.shape[0]
    if n < 2:
        return [(0.0, np.array([0], dtype=int), None)]

    D: List[List[Tuple[float, np.ndarray]]] = [[] for _ in range(n)]
    uid = 0

    def push_Hj(Hj, cost: float, i: int, r: int, piv: np.ndarray) -> None:
        nonlocal uid
        uid += 1
        heapq.heappush(Hj, (float(cost), uid, int(i), int(r), tuple(int(x) for x in piv)))

    def push_Hn(Hn, cost: float, i: int, r: int, piv: np.ndarray, vals: Optional[np.ndarray]) -> None:
        nonlocal uid
        uid += 1
        vals_t = None if vals is None else tuple(float(v) for v in vals)
        heapq.heappush(Hn, (float(cost), uid, int(i), int(r), tuple(int(x) for x in piv), vals_t))

    # D[j] build for j = 1..n-1
    for j in range(1, n):
        Hj: List[Tuple[float, int, int, int, Tuple[int, ...]]] = []

        # seed from pseudo-start (i==0) -> left-extended first segment to j
        cost0 = alpha * errL[0, j] + beta
        push_Hj(Hj, cost0, 0, 0, np.array([0, j], dtype=int))

        # seed from DP predecessors: combine D[i][1] with segment i->j
        for i in range(1, j):
            if not D[i]:
                continue
            prev_cost, prev_piv = D[i][0]
            cost = prev_cost + alpha * err[i, j] + beta
            push_Hj(Hj, cost, i, 1, np.append(prev_piv, j))

        while Hj and len(D[j]) < q:
            val, _, i, r, piv_t = heapq.heappop(Hj)
            piv = np.array(piv_t, dtype=int)
            D[j].append((float(val), piv))

            if r >= 1 and r < len(D[i]):
                next_prev_cost, next_prev_piv = D[i][r]
                nxt_cost = next_prev_cost + alpha * err[i, j] + beta
                push_Hj(Hj, nxt_cost, i, r + 1, np.append(next_prev_piv, j))

    # final heap H_n
    Hn: List[Tuple[float, int, int, int, Tuple[int, ...]]] = []
    for i in range(1, n - 1):
        if not D[i]:
            continue
        prev_cost, prev_piv = D[i][0]
        cost = prev_cost + alpha * errR[i, n - 1] + beta
        push_Hn(Hn, cost, i, 1, np.append(prev_piv, n - 1), None)

    # both-ends single-line candidates with explicit endpoint values.
    # this keeps stage-1 cost consistent with stage-2 reconstruction.
    for i in range(0, n - 1):
        for k in range(i + 1, n):
            a, b = line_through(i, y[i], k, y[k])
            cost = alpha * errA[i, k] + beta
            vals = np.array([b, a * (n - 1) + b], dtype=float)
            push_Hn(Hn, cost, -1, 0, np.array([0, n - 1], dtype=int), vals)

    finals: List[Stage1Candidate] = []
    seen: set[Tuple] = set()
    while Hn and len(finals) < q:
        val, _, i, r, piv_t, vals_t = heapq.heappop(Hn)
        piv = np.array(piv_t, dtype=int)
        vals = None if vals_t is None else np.array(vals_t, dtype=float)

        piv = np.unique(np.clip(piv, 0, n - 1))
        if vals is not None:
            if piv.size != 2:
                continue
        if piv[0] != 0:
            piv = np.insert(piv, 0, 0)
        if piv[-1] != n - 1:
            piv = np.append(piv, n - 1)

        if vals is None:
            key = ("piv", tuple(piv.tolist()))
        else:
            key = ("piv_vals", tuple(piv.tolist()), tuple(np.round(vals, 12).tolist()))
        if key not in seen:
            finals.append((float(val), piv, vals))
            seen.add(key)

        if i >= 1 and r < len(D[i]):
            next_prev_cost, next_prev_piv = D[i][r]
            nxt_cost = next_prev_cost + alpha * errR[i, n - 1] + beta
            push_Hn(Hn, nxt_cost, i, r + 1, np.append(next_prev_piv, n - 1), None)

    finals.sort(key=lambda x: x[0])
    return finals[:q]


def stage1_dp(y: np.ndarray, q: int, alpha: float, beta: float) -> List[Stage1Candidate]:
    """Run vanilla DP Stage-1 with legacy candidate semantics.

    Args:
        y: One-dimensional signal.
        q: Number of top candidates to return.
        alpha: Error weight in ``alpha * err + beta * k``.
        beta: Segment-count weight in ``alpha * err + beta * k``.

    Returns:
        Ranked stage-1 candidates as ``(cost, pivots, values_or_none)``.
    """
    if len(y) < 2:
        return [(0.0, np.array([0], dtype=int), None)]
    err, errL, errR, errA = build_error_tables_no_prefix(y)
    cands = _dp_with_heaps_and_ranks_legacy(
        err,
        errL,
        errR,
        errA,
        alpha=alpha,
        beta=beta,
        q=q,
    )
    return [(float(cost), np.asarray(piv, dtype=int), None) for cost, piv in cands]


def stage1_dp_prefix(y: np.ndarray, q: int, alpha: float, beta: float) -> List[Stage1Candidate]:
    """Run legacy DP-prefix Stage-1 kept for benchmarking.

    This mode keeps the historical v2 behaviour and remains available so that
    benchmark runs can compare v1, v2, and v3 in one module.

    Args:
        y: One-dimensional signal.
        q: Number of top candidates to return.
        alpha: Error weight in ``alpha * err + beta * k``.
        beta: Segment-count weight in ``alpha * err + beta * k``.

    Returns:
        Ranked stage-1 candidates as ``(cost, pivots, values_or_none)``.
    """
    if len(y) < 2:
        return [(0.0, np.array([0], dtype=int), None)]
    err, errL, errR, errA = build_error_tables_with_prefix_legacy(y)
    cands = _dp_with_heaps_and_ranks_legacy(
        err,
        errL,
        errR,
        errA,
        alpha=alpha,
        beta=beta,
        q=q,
    )
    return [(float(cost), np.asarray(piv, dtype=int), None) for cost, piv in cands]


def stage1_dp_prefix_v3(y: np.ndarray, q: int, alpha: float, beta: float) -> List[Stage1Candidate]:
    """Run v3 DP-prefix Stage-1 with full prefix-sum table acceleration.

    Args:
        y: One-dimensional signal.
        q: Number of top candidates to return.
        alpha: Error weight in ``alpha * err + beta * k``.
        beta: Segment-count weight in ``alpha * err + beta * k``.

    Returns:
        Ranked stage-1 candidates as ``(cost, pivots, values_or_none)``.
    """
    if len(y) < 2:
        return [(0.0, np.array([0], dtype=int), None)]
    err, errL, errR, errA = build_error_tables_with_prefix(y)
    return _dp_with_heaps_and_ranks_v3(
        y,
        err,
        errL,
        errR,
        errA,
        alpha=alpha,
        beta=beta,
        q=q,
    )


def stage1_rdp(y: np.ndarray, stage1_candidates: int, beta: float) -> List[Tuple[float, np.ndarray]]:
    """Heuristic Stage-1 via RDP; sweep epsilons, keep best candidate per k under l2 + beta*k."""
    n = len(y)
    x = np.arange(n)
    out: List[Tuple[float, np.ndarray]] = []
    if n < 2:
        return [(0.0, np.array([0], dtype=int))]

    epsilons = np.geomspace(1e-6, max(1e-6, float(np.ptp(y))), num=stage1_candidates)
    candidates: List[Tuple[float, np.ndarray]] = []
    for eps in epsilons:
        keep = rdp(np.column_stack([x, y]), epsilon=float(eps), return_mask=True)
        piv = np.flatnonzero(keep)
        if piv.size < 2:
            piv = np.array([0, n - 1], dtype=int)
        y_hat = np.interp(x, piv, y[piv])
        err = float(np.sum((y - y_hat) ** 2))
        k = int(piv.size - 1)
        cost = err + beta * k
        candidates.append((cost, piv))

    best_per_k: Dict[int, Tuple[float, np.ndarray]] = {}
    for cost, piv in candidates:
        k = int(piv.size - 1)
        if (k not in best_per_k) or (cost < best_per_k[k][0]):
            best_per_k[k] = (cost, piv)
    out = list(best_per_k.values())
    out.sort(key=lambda x: x[0])
    return out


def _eval_piecewise_with_linear_extrap(x: int, t: np.ndarray, y: np.ndarray) -> float:
    """Evaluate piecewise linear curve with linear edge extrapolation at index x."""
    xf = float(x)
    if xf <= float(t[0]):
        if t.size == 1 or int(t[1]) == int(t[0]):
            return float(y[0])
        m = (float(y[1]) - float(y[0])) / float(int(t[1]) - int(t[0]))
        return float(y[0]) + m * (xf - float(t[0]))
    if xf >= float(t[-1]):
        if t.size == 1 or int(t[-1]) == int(t[-2]):
            return float(y[-1])
        m = (float(y[-1]) - float(y[-2])) / float(int(t[-1]) - int(t[-2]))
        return float(y[-1]) + m * (xf - float(t[-1]))
    return float(np.interp([xf], t.astype(float), y.astype(float))[0])


def _postprocess_candidate(
    piv_rel: np.ndarray,
    values_rel: Optional[np.ndarray],
    T: int,
    offset: int,
    anchor_endpoints: Literal["both", "last"],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Shift candidate to absolute time and enforce endpoint policy with value consistency."""
    p = np.asarray(piv_rel, dtype=int).reshape(-1) + int(offset)
    v = None if values_rel is None else np.asarray(values_rel, dtype=float).reshape(-1)

    if v is not None and p.size != v.size:
        raise ValueError("pivot values must match pivot indices")

    order = np.argsort(p, kind="stable")
    p = p[order]
    if v is not None:
        v = v[order]

    if v is None:
        return _postprocess_pivots(p, T, 0, anchor_endpoints), None

    dedup: Dict[int, float] = {}
    for pi, vi in zip(p.tolist(), v.tolist()):
        dedup[int(pi)] = float(vi)
    p = np.array(sorted(dedup.keys()), dtype=int)
    v = np.array([dedup[int(pi)] for pi in p], dtype=float)

    if anchor_endpoints == "both" and (p.size == 0 or int(p[0]) != 0):
        v0 = _eval_piecewise_with_linear_extrap(0, p, v)
        p = np.r_[0, p]
        v = np.r_[v0, v]

    if p.size == 0 or int(p[-1]) != T - 1:
        v_end = _eval_piecewise_with_linear_extrap(T - 1, p, v)
        p = np.r_[p, T - 1]
        v = np.r_[v, v_end]

    p = np.clip(p, 0, T - 1).astype(int)
    dedup2: Dict[int, float] = {}
    for pi, vi in zip(p.tolist(), v.tolist()):
        dedup2[int(pi)] = float(vi)
    p = np.array(sorted(dedup2.keys()), dtype=int)
    v = np.array([dedup2[int(pi)] for pi in p], dtype=float)
    return p, v


def _ors_candidates_with_values(y: np.ndarray, params: ORSParams) -> Optional[List[Stage1Candidate]]:
    """Dispatch stage-1 candidate generation and preserve optional pivot values."""
    T = int(y.shape[0])

    # optionally slices original series to window where model predictions are evaluated
    if params.anchor_endpoints == "last" and params.t_min_eval > 0:
        piv_rel, offset = y[int(params.t_min_eval):], int(params.t_min_eval)
    else:
        piv_rel, offset = y, 0

    # stage-1 returns a list of relative pivot arrays (sorted and with slice ends included)
    mode = params.stage1_mode.lower()
    if mode == "dp":
        cand_rel = stage1_dp(piv_rel, q=params.dp_q, alpha=params.dp_alpha, beta=params.beta)
    elif mode == "dp_prefix":
        cand_rel = stage1_dp_prefix(piv_rel, q=params.dp_q, alpha=params.dp_alpha, beta=params.beta)
    elif mode == "dp_prefix_v3":
        cand_rel = stage1_dp_prefix_v3(piv_rel, q=params.dp_q, alpha=params.dp_alpha, beta=params.beta)
    elif mode == "rdp":
        cands_plain = stage1_rdp(piv_rel, stage1_candidates=params.rdp_stage1_candidates, beta=params.beta)
        cand_rel = [(float(cost), np.asarray(piv, dtype=int), None) for cost, piv in cands_plain]
    else:
        print(
            f"[ors] Warning: unknown stage1_mode: {params.stage1_mode}. "
            "Valid modes are 'dp', 'dp_prefix', 'dp_prefix_v3', and 'rdp'. Reverting to mode 'rdp'."
        )
        cands_plain = stage1_rdp(piv_rel, stage1_candidates=params.rdp_stage1_candidates, beta=params.beta)
        cand_rel = [(float(cost), np.asarray(piv, dtype=int), None) for cost, piv in cands_plain]

    # shifts relative pivots back to absolute time and enforces endpoint policy
    cands_abs: List[Stage1Candidate] = []
    for cost, piv, vals in cand_rel:
        p_abs, v_abs = _postprocess_candidate(
            piv_rel=piv,
            values_rel=vals,
            T=T,
            offset=offset,
            anchor_endpoints=params.anchor_endpoints,
        )
        cands_abs.append((float(cost), p_abs, v_abs))
    return cands_abs


def ors_candidates(y: np.ndarray, params: ORSParams) -> Optional[List[Tuple[float, np.ndarray]]]:
    """Dispatch stage-1 candidate generation according to mode.

    This public helper preserves the legacy return shape ``(cost, pivots)``.
    """
    cands = _ors_candidates_with_values(y, params)
    if cands is None:
        return None
    return [(float(cost), piv) for (cost, piv, _vals) in cands]


def _postprocess_pivots(piv_rel: np.ndarray, T: int, offset: int, anchor_endpoints: Literal["both","last"]) -> np.ndarray:
    """Shift relative pivots back to absolute time and enforce endpoint policy."""
    P = np.asarray(piv_rel, dtype=int).reshape(-1) + int(offset)
    if anchor_endpoints == "both":
        P = np.r_[0, P, T - 1]
    else:  # "last"
        P = np.r_[P, T - 1]
    return np.unique(np.clip(P, 0, T - 1))

# ----------------------------------------- ORS stage-2: robustness --------------------------------------- #

def interpolate_from_pivots(
    T: int, pivots: np.ndarray, values: np.ndarray, t_min_eval,
    anchor_endpoints: Literal["both", "last"] = "last"
) -> np.ndarray:
    """Return a dense length-T series from pivots with optional left extrapolation.

    If anchor_endpoints == "last", the curve linearly extends the first interior segment
    over [0, t_min_eval) so the value and slope at t_min_eval match, no breakpoint is added.
    If anchor_endpoints == "both", behaviour matches the legacy 'anchor 0 and T-1'.

    Args:
        T: Output length.
        pivots: Knot indices in absolute time (sorted or unsorted), shape (K,).
        values: Knot values (kW), shape (K,).
        t_min_eval: First timestep used by the classifier.
        anchor_endpoints: Whether to anchor both ends or only the last one.

    Returns:
        Dense piecewise-linear series of shape (T,).
    """
    t = np.asarray(pivots, dtype=int).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]

    if anchor_endpoints in ("last", "both") and t[-1] != T - 1:
        t = np.concatenate([t, [T - 1]])
        y = np.concatenate([y, [y[-1]]])

    x = np.arange(T, dtype=float)

    if anchor_endpoints == "both" or t_min_eval <= 0:
        return np.interp(x, t.astype(float), y.astype(float))

    # legacy 'both' is handled above; below is slope-preserving left tail for 'last'
    out = np.interp(x, t.astype(float), y.astype(float))

    # find the segment that governs t_min_eval to get its slope m
    j = int(np.searchsorted(t, t_min_eval, side="right"))
    i0 = max(0, j - 1)
    i1 = min(j, t.size - 1)

    # ensure two distinct knots to define a slope; fall back to flat if degenerate
    denom = int(t[i1]) - int(t[i0])
    if denom == 0:
        m = 0.0
        y_eval = float(y[i0])
    else:
        m = (float(y[i1]) - float(y[i0])) / float(denom)
        y_eval = float(y[i0]) + m * float(t_min_eval - int(t[i0]))

    if t_min_eval > 0:
        left_idx = np.arange(t_min_eval, dtype=float)
        out[:t_min_eval] = y_eval - m * (t_min_eval - left_idx)

    return out


def interpolate_batch_from_pivots(
    T: int, pivots: np.ndarray, value_batch: np.ndarray, t_min_eval,
    anchor_endpoints: Literal["both", "last"] = "last"
) -> np.ndarray:
    """Vectorised interpolation with evaluation-aware left extrapolation."""
    t = np.asarray(pivots, dtype=int).reshape(-1)
    V = np.asarray(value_batch, dtype=float)          # shape (R, K)
    R, K = V.shape
    order = np.argsort(t, kind="stable")
    t, V = t[order], V[:, order]

    if anchor_endpoints in ("last", "both") and t[-1] != T - 1:
        t = np.concatenate([t, [T - 1]])
        V = np.concatenate([V, V[:, [-1]]], axis=1)
        K += 1

    x = np.arange(T, dtype=float)

    if anchor_endpoints == "both" or t_min_eval <= 0:
        # piecewise-linear interpolation per row
        out = np.empty((R, T), dtype=float)
        for r in range(R):
            out[r] = np.interp(x, t.astype(float), V[r].astype(float))
        return out

    # build interior by segments (faster than per-row np.interp) and then fix left tail
    out = np.empty((R, T), dtype=float)
    for s in range(K - 1):
        a, b = int(t[s]), int(t[s + 1])
        if b <= a:
            continue
        span = b - a
        m = (V[:, s + 1] - V[:, s]) / float(span)
        xs = np.arange(span + 1, dtype=float)
        seg = V[:, [s]] + m[:, None] * xs[None, :]
        out[:, a:b + 1] = seg
    out[:, -1] = V[:, -1]

    # slope-preserving left tail for all rows
    j = int(np.searchsorted(t, t_min_eval, side="right"))
    i0 = max(0, j - 1)
    i1 = min(j, K - 1)
    denom = float(max(1, int(t[i1]) - int(t[i0])))
    m = (V[:, i1] - V[:, i0]) / denom
    y_eval = V[:, i0] + m * float(t_min_eval - int(t[i0]))
    left = np.arange(t_min_eval, dtype=float)
    out[:, :t_min_eval] = y_eval[:, None] - m[:, None] * (t_min_eval - left)[None, :]

    return out


def fragility_uniform_band_batched(
    bundle: SessionPredsBundle,
    model: torch.nn.Module,
    pivots: np.ndarray,
    sts_y: np.ndarray,
    *,
    R: int,
    eps: float,
    base_label: int,
    power_scaler: MinMaxScaler,
    idx_power_inp: int,
    decay_lambda: float,
    t_min_eval: int,
    anchor_endpoints: Literal["both", "last"] = "last",
    Y_abs_true: np.ndarray,
    threshold: float,
    seed: Optional[int],
) -> float:
    """estimate fragility by sampling uniform ±epsilon at pivot values and reclassifying."""
    rng = np.random.default_rng(seed)
    k = len(pivots) - 1
    T = sts_y.size

    pv = sts_y[pivots].astype(float)                                # (k+1,)
    noise = rng.uniform(-eps, +eps, size=(R, k + 1)).astype(float)  # (R, k+1)
    pv_batch = pv[None, :] + noise                                  # (R, k+1)
    power_batch = interpolate_batch_from_pivots(
        T,
        pivots,
        pv_batch,
        t_min_eval=t_min_eval,
        anchor_endpoints=anchor_endpoints,
    )  # (R, T)

    errs = macro_rmse_for_power_batch(
        bundle,
        model,
        power_batch,
        power_scaler=power_scaler,
        idx_power_inp=idx_power_inp,
        decay_lambda=decay_lambda,
        t_min_eval=t_min_eval,
        Y_abs_true=Y_abs_true,
    )
    labels = (errs > float(threshold)).astype(int)
    flips = np.count_nonzero(labels != int(base_label))
    return float(flips) / float(R)


# ----------------------------------------- helpers ---------------------------------------------------- #

def _epsilon_from_range(y_min: float, y_max: float, params: ORSParams) -> float:
    """Compute epsilon for robustness sampling based on params."""
    if params.epsilon_mode == "fraction":
        return float(params.epsilon_value) * max(1e-9, (y_max - y_min))
    elif params.epsilon_mode == "kw":
        return float(params.epsilon_value)
    else:
        raise ValueError(f"unknown epsilon_mode: {params.epsilon_mode}")


def _k_span(cands: Iterable[Tuple]) -> Tuple[Optional[int], Optional[int]]:
    """Return (k_min, k_max) over a candidate iterable."""
    ks = [int(len(cand[1]) - 1) for cand in (cands or [])]
    return (min(ks), max(ks)) if ks else (None, None)


def _session_id(bundle) -> Optional[int]:
    """Best-effort session id fetch for logging."""
    try:
        sid = getattr(bundle, "session_id", None)
        return int(sid) if sid is not None else None
    except Exception:
        return None


# ----------------------------------------- main driver ------------------------------------------------ #

def ors(bundle: SessionPredsBundle,
        model: torch.nn.Module,
        params: ORSParams,
        *,
        power_scaler,
        idx_power_inp: int,
        decay_lambda: float,
        threshold: float) -> Optional[Dict]:
    """Run ORS end-to-end with fallbacks, keeping the label-consistency constraint.

    The pipeline:
      1) Stage-1 candidate generation (DP, legacy DP-prefix, v3 DP-prefix, or RDP).
      2) Base label on the original series.
      3) Filter candidates that do NOT keep the base label (constraint).
      4) Robustness estimation (uniform ±epsilon at pivots).
      5) Stage-2 selection with objective: alpha*err + beta*k + gamma*frag, where
         - alpha = dp_alpha for DP modes. For RDP, alpha is implicit in stage-1
           because its cost is l2 + beta*k (final selection still uses gamma*frag).

    Fallbacks when no valid candidate survives the label/k filters:
      - Fallback #1: rerun Stage-1 with dp_q = dp_q * 2 (if DP mode) and beta = beta * 4.
      - Fallback #2: switch to RDP with rdp_stage1_candidates = max(200, 3*(max_k-min_k+1)).
      - If still none: log and return None (caller may skip this session).

    Args:
      bundle: Session bundle with scaled inputs, predictions, and unscaled series.
      model: PyTorch model to evaluate perturbations.
      params: ORSParams configuration.
      power_scaler, soc_scaler: scalers for inverse transforms.
      idx_power_inp, idx_soc_inp: column indices in X for power/SOC.

    Returns:
      dict with keys {obj,k,piv,sts,frag,err,label} on success; None if no valid candidate. 
      Additionally includes piv_soc and sts_soc when params.soc_stage1_mode == "rdp" and SOC is present.
    """
    T = bundle.length
    y = np.asarray(bundle.true_power_unscaled, dtype=float)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    sid = _session_id(bundle)
    
    # attach SOC simplification if configured, doesn't impact power simplification
    def _with_soc(result: Dict | None) -> Dict | None:
        if result is None:
            return None
        try:
            if getattr(params, "soc_stage1_mode", None) == "rdp":
                y_soc = np.asarray(bundle.true_soc_unscaled, dtype=float).reshape(-1)
                if y_soc.size != T:
                    raise ValueError("power and SOC must have equal length for teaching overlays")
                piv_soc, sts_soc = _ors_soc_rdp_select(y_soc, params)
                if piv_soc is not None and sts_soc is not None:
                    result["piv_soc"] = piv_soc
                    result["sts_soc"] = sts_soc
        except Exception as e:
            # non-fatal: keep power result intact
            # reason: soc overlays are auxiliary for teaching; never block power
            pass
        return result


    # ---- Stage-1 candidates (initial) ----
    cands = _ors_candidates_with_values(y, params)
    if cands is None:
        # explicit None: log + return degenerate k=1 for safety as requested
        print(f"[ORS][warn] sid={sid} got cands=None (mode={params.stage1_mode}, dp_q={params.dp_q}, "
              f"rdp_stage1_candidates={params.rdp_stage1_candidates}). Returning k=1 fallback.")
        piv = np.array([0, T - 1], dtype=int)
        sts = interpolate_from_pivots(T, piv, y[piv], anchor_endpoints=params.anchor_endpoints)
        return _with_soc(dict(obj=0.0, k=1, piv=piv, sts=sts, frag=0.0, err=0.0, label="normal"))

    # base label on original series
    base_lbl_str, base_err, Y_abs_true = base_label_from_bundle(
        bundle,
        power_scaler=power_scaler,
        idx_power_inp=idx_power_inp,
        decay_lambda=decay_lambda,
        t_min_eval=params.t_min_eval,
        threshold=threshold,
    )
    # internal numeric label: 1 = abnormal, 0 = normal
    base_lbl = 1 if base_lbl_str == "abnormal" else 0

    # epsilon for robustness sampling
    eps = _epsilon_from_range(y_min, y_max, params)

    # theorem-2 gap diagnostic (DP only)
    if params.stage1_mode == "dp" and len(cands) >= 2:
        q_eff = min(params.dp_q, len(cands))
        d = float(cands[q_eff - 1][0] - cands[0][0]) if q_eff >= 2 else float("inf")
        if params.gamma > d:
            print(f"[ORS] warning: gamma={params.gamma:.4g} > gap d={d:.4g}; optimality not guaranteed. "
                  f"consider increasing dp_q or reducing gamma.")

    def evaluate_candidates(cands_list: List[Stage1Candidate], p: ORSParams) -> Optional[Dict]:
        """evaluate stage-1 candidates under constraints; return best dict or None."""
        best: Optional[Dict] = None
        for cost_es, piv, piv_vals in cands_list:
            k = int(len(piv) - 1)
            if k < int(p.min_k) or k > int(p.max_k):
                continue

            knot_values = np.asarray(piv_vals, dtype=float) if piv_vals is not None else y[piv]
            sts = interpolate_from_pivots(
                T, piv, knot_values, t_min_eval=p.t_min_eval, anchor_endpoints=p.anchor_endpoints
            )
            l2_err = float(np.sum((y - sts) ** 2))

            lbl_sts, err_sts = classify_macro_rmse_from_power(
                bundle,
                model,
                sts,
                power_scaler=power_scaler,
                idx_power_inp=idx_power_inp,
                decay_lambda=decay_lambda,
                t_min_eval=p.t_min_eval,
                Y_abs_true=Y_abs_true,
                threshold=threshold,
            )

            # enforces label-consistency. keep only simplifications that preserve the base label
            if int(lbl_sts) != base_lbl:
                continue

            frag = fragility_uniform_band_batched(
                bundle,
                model,
                pivots=piv,
                sts_y=sts,
                R=p.R,
                eps=eps,
                base_label=base_lbl,  # int: 0 or 1
                power_scaler=power_scaler,
                idx_power_inp=idx_power_inp,
                decay_lambda=decay_lambda,
                t_min_eval=p.t_min_eval,
                anchor_endpoints=p.anchor_endpoints,
                Y_abs_true=Y_abs_true,
                threshold=threshold,
                seed=p.random_seed,
            )

            # stage-1 cost_es is:
            #  - dp: dp_alpha*l2_err + beta*k
            #  - rdp: l2_err + beta*k
            # stage-2 adds gamma*frag
            total = float(cost_es + p.gamma * frag)
            robust_prob=1.0 - frag
            cand = dict(
                obj=total,
                k=k,
                piv=piv,
                sts=sts,
                frag=float(frag),
                robust_prob=robust_prob,
                err=float(err_sts),
                label=("abnormal" if lbl_sts == 1 else "normal"),
                l2_err=l2_err,
            )
            if (best is None) or (cand["obj"] < best["obj"]):
                best = cand
        return best


    # ------ evaluate initial candidates ------ #
    best = evaluate_candidates(cands, params)
    if best is not None:
        return _with_soc(best)

    # ----- Fallback #1: log + rerun with dp_q*2 (if DP) and beta*4 ----- #
    k_min, k_max = _k_span(cands)
    print(f"[ORS][warn] sid={sid} no valid candidates after constraints "
          f"(mode={params.stage1_mode}, k_span={k_min}..{k_max}, dp_q={params.dp_q}, beta={params.beta}). "
          f"Trying fallback #1.")

    p_fb1 = replace(params, beta=params.beta * 4.0)
    if params.stage1_mode in {"dp", "dp_prefix", "dp_prefix_v3"}:
        p_fb1 = replace(p_fb1, dp_q=max(1, params.dp_q * 2))

    cands1 = _ors_candidates_with_values(y, p_fb1)
    if cands1 is None:
        print(f"[ORS][warn] sid={sid} fallback #1 produced cands=None; proceeding to fallback #2.")
    else:
        best = evaluate_candidates(cands1, p_fb1)
        if best is not None:
            return _with_soc(best)

    # ----- Fallback #2: switch to RDP with large stage1_candidates ----- #
    k_min1, k_max1 = _k_span(cands1 or [])
    print(f"[ORS][warn] sid={sid} still no valid candidates "
          f"(fb#1 k_span={k_min1}..{k_max1}, dp_q={getattr(p_fb1, 'dp_q', None)}, beta={p_fb1.beta}). "
          f"Trying fallback #2 (rdp).")

    rdp_count = max(200, 3 * (params.max_k - params.min_k + 1))
    p_fb2 = replace(params, stage1_mode="rdp", rdp_stage1_candidates=rdp_count)

    cands2 = _ors_candidates_with_values(y, p_fb2)
    if cands2 is None:
        print(f"[ORS][warn] sid={sid} fallback #2 (rdp) produced cands=None; giving up for this session.")
        return None

    best = evaluate_candidates(cands2, p_fb2)
    if best is not None:
        return _with_soc(best)

    k_min2, k_max2 = _k_span(cands2)
    print(f"[ORS][warn] sid={sid} fallbacks exhausted (rdp k_span={k_min2}..{k_max2}); skipping session.")
    return None


def _ors_soc_rdp_select(y_soc: np.ndarray, params: ORSParams) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Selects an SOC simplification using ORS stage-1 with RDP candidates.

    This uses the same candidate generator `ors_candidates` with `stage1_mode="rdp"`,
    and selects the best candidate by the stage-1 cost_es returned for RDP, which is
    l2 + beta*k. Robustness and label constraints are intentionally not applied for
    SOC, since classification depends on power, not SOC.

    Args:
        y_soc: Dense SOC series in percent, shape (T,).
        params: ORS parameters; only beta, min_k, max_k and rdp-related knobs matter.

    Returns:
        (piv, sts) where `piv` are pivot indices (int32) and `sts` is the dense
        piecewise-linear reconstruction (float64). Returns (None, None) if no
        candidate can be produced.
    """
    y_soc = np.asarray(y_soc, dtype=float).reshape(-1)
    T = y_soc.size
    if T < 2:
        return None, None

    # force stage-1 RDP for SOC, preserve k-range and beta
    p_soc = replace(params, stage1_mode="rdp")
    cands = ors_candidates(y_soc, p_soc)
    if not cands:
        return None, None

    best_cost = float("inf")
    best_piv = None
    for cost_es, piv in cands:
        k = int(len(piv) - 1)
        if k < int(p_soc.min_k) or k > int(p_soc.max_k):
            continue
        if float(cost_es) < best_cost:
            best_cost = float(cost_es)
            best_piv = piv

    if best_piv is None:
        return None, None

    piv = np.asarray(best_piv, dtype=int).reshape(-1)
    xs = np.arange(T, dtype=float)
    sts = np.interp(xs, piv.astype(float), y_soc[piv])
    return piv.astype(np.int32), sts


# ----------------------------------------- diagnostics ------------------------------------------------ #

def summarize_candidates_table(outcomes: List[ORSOutcome]) -> pd.DataFrame:
    """Build a summary DataFrame over evaluated candidates for debugging/reporting."""
    rows = []
    for o in outcomes:
        rows.append({
            "k": o.k_opt,
            "robust_prob": o.robust_prob,
            "frag": 1.0 - o.robust_prob,
            "simpl_err(MRMSE)": o.simplified_error,
            "L2_err": o.l2_err,
            "objective": o.objective,
            "label": o.simplified_label
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["objective", "k", "L2_err"], ascending=[True, True, True])
    return df
