# mt4xai/ors.py
# Portions adapted from: https://github.com/BrigtHaavardstun/kSimplification (MIT License)
# Copyright (c) 2024 Brigt Håvardstun, Cèsar Ferri, Jan Arne Telle

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import heapq
import numpy as np
import pandas as pd
import torch
from rdp import rdp
from .plot import reconstruct_abs_from_bundle           # (T,H,2) scaled
from .inference import inverse_targets_np, predict_residuals, reconstruct_abs_from_residuals_batch


# ----------------------------------------- config and outputs ----------------------------------------- #

@dataclass
class ORSParams:
    """
    holds configuration for optimal robust simplification.

    the stage-1 generator is chosen via stage1_mode: 
    - "dp": vanilla dynamic programming with heaps (paper-exact; no prefix sums), 
    - "dp_prefix": same dp + heaps but with prefix-sum error tables (faster), 
    - "rdp": ramer-douglas-peucker heuristic.

    in stage-2, stage2_err_metric selects the error used in the final objective:
    - "l2": squared euclidean distance,
    - "mrmse": macro-rmse from the forecasting/classification pipeline.

    all computations operate in original units (kW / %soc).
    """

    # which stage-1 generator to use: "dp" (paper-exact, no prefix sums), "dp_prefix" (faster DP), or "rdp" (RDP-generated candidates)
    stage1_mode: str = "dp"   # default is paper-exact/vanilla: DP with no prefix sums

    # which error to use in the *final objective* (stage-2 selection): "l2" or "mrmse"
    stage2_err_metric: str = "l2"  # default to squared euclidean (paper-exact)

    q: int = 100
    stage1_candidates: int = 40

    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0

    R: int = 10_000
    epsilon_mode: str = "fraction"
    epsilon_value: float = 0.1

    seed: Optional[int] = 1337
    min_k: int = 1
    max_k: int = 60
    t_min_eval: int = 0


@dataclass
class ORSOutcome:
    """
    stores one evaluated candidate for diagnostics.
    includes pivot mask, simplified series, l2 error, macro-rmse error, robustness and objective value.
    """
    k_opt: int
    keep_mask: np.ndarray
    simplified_power: np.ndarray
    robust_prob: float
    simplified_label: str
    simplified_error: float
    l2_err: float
    objective: float


# ----------------------------------------- macro-rmse classifier -------------------------------------- #

def build_true_abs_from_series(power_true: np.ndarray, soc_true: np.ndarray, H: int) -> np.ndarray:
    """
    builds absolute targets Y_abs[t,h,c] aligned with horizon h in original units.
    """
    T = power_true.size
    Y = np.zeros((T, H, 2), dtype=float)
    for h0 in range(H):
        h1 = h0 + 1
        end = T - h1
        if end <= 0:
            break
        Y[:end, h0, 0] = power_true[h1:h1+end]
        Y[:end, h0, 1] = soc_true[h1:h1+end]
    return Y


def macro_rmse_from_abs(P_abs: np.ndarray, Y_abs: np.ndarray, power_weight: float,
                        decay_lambda: float, t_min_eval: int) -> float:
    """
    computes macro-rmse across horizons with an exponential decay, in original units.
    """
    T, H, _ = P_abs.shape
    w_h = np.exp(-float(decay_lambda) * np.arange(H, dtype=float))
    w_h = w_h / np.sum(w_h)
    vals = []
    for h in range(H):
        end = T - (h + 1)
        if end <= t_min_eval:
            continue
        diff = P_abs[t_min_eval:end, h, :] - Y_abs[t_min_eval:end, h, :]
        rmse_c = np.sqrt(np.mean(diff**2, axis=0))
        vals.append(power_weight * rmse_c[0] + (1.0 - power_weight) * rmse_c[1])
    if not vals:
        return 0.0
    return float(np.sum(np.asarray(vals) * w_h[:len(vals)]))


@torch.inference_mode()
def macro_rmse_for_power_batch(bundle, model: torch.nn.Module, power_batch_kw: np.ndarray, *,
                               power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int,
                               power_weight: float, decay_lambda: float, t_min_eval: int,
                               Y_abs_true: np.ndarray) -> np.ndarray:
    """
    runs a batched forward pass for perturbed power series and returns macro-rmse per perturbation.
    uses reconstruction + inverse-transform helpers to stay in original units.
    """
    T = bundle.length
    R = power_batch_kw.shape[0]
    X_base = bundle.X_sample.clone()                              # (T, F) scaled
    p_scaled = power_scaler.transform(power_batch_kw.reshape(-1, 1)).reshape(R, T)
    Xs = X_base.unsqueeze(0).repeat(R, 1, 1).contiguous()         # (R, T, F)
    Xs[:, :, idx_power_inp] = torch.from_numpy(p_scaled.astype(np.float32)).to(Xs)

    lengths = torch.tensor([T] * R, dtype=torch.long)
    device = next(model.parameters()).device
    P_res = predict_residuals(model, Xs.to(device), lengths, device=device)               # (R, T, H, C)
    P_abs_scaled = reconstruct_abs_from_residuals_batch(Xs, P_res, idx_power_inp, idx_soc_inp)  # (R,T,H,2)
    P_abs = inverse_targets_np(P_abs_scaled.cpu().numpy(), power_scaler, soc_scaler)      # (R, T, H, 2)

    errs = np.zeros(R, dtype=float)
    for r in range(R):
        errs[r] = macro_rmse_from_abs(P_abs[r], Y_abs_true, power_weight, decay_lambda, t_min_eval)
    return errs


def classify_macro_rmse_from_power(bundle, model, power_kw: np.ndarray, *,
                                   power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int,
                                   power_weight: float, decay_lambda: float, t_min_eval: int,
                                   Y_abs_true: np.ndarray, threshold: float) -> tuple[int, float]:
    """
    classifies a single power curve by macro-rmse threshold.
    returns (label_int, err).
    """
    errs = macro_rmse_for_power_batch(bundle, model, power_kw[None, :], power_scaler=power_scaler,
                                      soc_scaler=soc_scaler, idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
                                      power_weight=power_weight, decay_lambda=decay_lambda, t_min_eval=t_min_eval,
                                      Y_abs_true=Y_abs_true)
    err = float(errs[0])
    return (1 if err > float(threshold) else 0), err


def base_label_from_bundle(bundle, *, power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int,
                           power_weight: float, decay_lambda: float, t_min_eval: int,
                           threshold: float) -> tuple[int, float, np.ndarray]:
    """
    obtains the base label and error of the unmodified session using stored predictions.
    """
    power_true = np.asarray(bundle.true_power_unscaled, dtype=float)
    soc_true = np.asarray(bundle.true_soc_unscaled, dtype=float)
    Y_abs_true = build_true_abs_from_series(power_true, soc_true, bundle.horizon)

    # predictions for the unmodified input (no extra forward needed)
    P_abs_scaled = reconstruct_abs_from_bundle(bundle, idx_power_inp, idx_soc_inp).numpy()
    P_abs = inverse_targets_np(P_abs_scaled, power_scaler, soc_scaler)

    err = macro_rmse_from_abs(P_abs, Y_abs_true, power_weight, decay_lambda, t_min_eval)
    lbl = 1 if err > float(threshold) else 0
    return lbl, err, Y_abs_true


# ----------------------------------------- stage-1: dp and rdp ---------------------------------------- #

def precompute_prefix_sums(y: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    precomputes segment errors for lines through endpoints using prefix sums for O(1) queries.
    identical in spirit to the paper's error tables; we only accelerate the segment sse query.
    """
    t = np.arange(len(y), dtype=float)
    S1 = np.cumsum(np.ones_like(t))
    Sy = np.cumsum(y)
    St = np.cumsum(t)
    St2 = np.cumsum(t * t)
    Sty = np.cumsum(t * y)
    Sy2 = np.cumsum(y * y)
    return S1, Sy, St, St2, Sty, Sy2


def seg_error_for_line(a: float, b: float, L: int, R: int, S: tuple[np.ndarray, ...]) -> float:
    S1, Sy, St, St2, Sty, Sy2 = S
    n = S1[R] - (S1[L - 1] if L > 0 else 0.0)
    sy = Sy[R] - (Sy[L - 1] if L > 0 else 0.0)
    st = St[R] - (St[L - 1] if L > 0 else 0.0)
    st2 = St2[R] - (St2[L - 1] if L > 0 else 0.0)
    sty = Sty[R] - (Sty[L - 1] if L > 0 else 0.0)
    sy2 = Sy2[R] - (Sy2[L - 1] if L > 0 else 0.0)
    return sy2 - 2 * a * sty - 2 * b * sy + (a * a) * st2 + 2 * a * b * st + n * (b * b)


def line_through(i: int, yi: float, j: int, yj: float) -> tuple[float, float]:
    a = (yj - yi) / (j - i)
    b = yi - a * i
    return a, b


def build_error_tables_with_prefix(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    precomputes segment errors err, err_left, err_right, err_all for lines through endpoints.
    source: authors' kSimplification repo and paper appendix. Implemented with prefix sums for O(1) segment queries.
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
            # extended left
            yhat = a * t[: j + 1] + b
            errL[i, j] = float(np.sum((y[: j + 1] - yhat) ** 2))
            # extended right
            yhat = a * t[i:] + b
            errR[i, j] = float(np.sum((y[i:] - yhat) ** 2))
            # extended both ends
            yhat = a * t + b
            errA[i, j] = float(np.sum((y - yhat) ** 2))
    return err, errL, errR, errA


def build_error_tables_no_prefix(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    precomputes segment errors for lines through endpoints by direct summation (O(n^3) build).
    matches the paper's baseline arithmetic.
    """
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

            # extend to left [0..j]
            yhatL = a * t[:j+1] + b
            errL[i, j] = float(np.sum((y[:j+1] - yhatL)**2))

            # extend to right [i..n-1]
            yhatR = a * t[i:] + b
            errR[i, j] = float(np.sum((y[i:] - yhatR)**2))

            # extend both [0..n-1]
            yhatA = a * t + b
            errA[i, j] = float(np.sum((y - yhatA)**2))

    return err, errL, errR, errA


def _dp_with_heaps_and_ranks(err: np.ndarray, errL: np.ndarray, errR: np.ndarray, errA: np.ndarray,
                             alpha: float, beta: float, q: int) -> list[tuple[float, np.ndarray]]:
    """
    stage-1 dp with heaps and ranks (paper-exact). for each end index j, maintains a heap H_j seeded
    from the left-extended first segment and from D[i][1] transitions; on each pop, pushes the 'next rank'
    successor from D[i][r+1]. final heap H_n extends to n-1 and adds single-line both-ends candidates.
    Note: Added "uid" as a strict tiebreaker so no two entries are identical for comparison. 
    source: Optimal Robust Simplifications for Explaining Time Series Classifications (Appendix 7.1)
    and authors' repo: https://github.com/BrigtHaavardstun/kSimplification
    """
    n = err.shape[0]
    if n < 2:
        return [(0.0, np.array([0], dtype=int))]

    # D[j]: list of (cost_es, pivots_ndarray) for the j-th endpoint (1-indexed ranks)
    D: list[list[tuple[float, np.ndarray]]] = [[] for _ in range(n)]

    uid = 0  # strict tiebreaker for heap entries

    # helper: push into Hj with heap-safe payload
    def push_Hj(Hj, cost: float, i: int, r: int, piv: np.ndarray):
        nonlocal uid
        uid += 1
        heapq.heappush(Hj, (float(cost), uid, int(i), int(r), tuple(int(x) for x in piv)))

    # helper: push into Hn with heap-safe payload
    def push_Hn(Hn, cost: float, i: int, r: int, piv: np.ndarray):
        nonlocal uid
        uid += 1
        heapq.heappush(Hn, (float(cost), uid, int(i), int(r), tuple(int(x) for x in piv)))

    # build D[j] for j = 1..n-1
    for j in range(1, n):
        Hj: list[tuple[float, int, int, int, tuple[int, ...]]] = []

        # seed from pseudo-start (i == 0) → left-extended first segment to j
        cost0 = alpha * errL[0, j] + beta
        push_Hj(Hj, cost0, 0, 0, np.array([0, j], dtype=int))  # r==0 => no DP successor

        # seed from DP predecessors: for each i<j, combine D[i][1] with segment i→j
        for i in range(1, j):
            if not D[i]:
                continue
            prev_cost, prev_piv = D[i][0]  # rank-1 (index 0)
            cost = prev_cost + alpha * err[i, j] + beta
            push_Hj(Hj, cost, i, 1, np.append(prev_piv, j))

        # pop best q items for D[j]; when an item came from D[i][r], push successor (r+1)
        while Hj and len(D[j]) < q:
            val, _, i, r, piv_t = heapq.heappop(Hj)
            piv = np.array(piv_t, dtype=int)
            D[j].append((float(val), piv))

            if r >= 1 and r < len(D[i]):
                next_prev_cost, next_prev_piv = D[i][r]
                nxt_cost = next_prev_cost + alpha * err[i, j] + beta
                push_Hj(Hj, nxt_cost, i, r + 1, np.append(next_prev_piv, j))

    # final heap H_n: extend to the end (n-1) and add both-ends single-segment options
    Hn: list[tuple[float, int, int, int, tuple[int, ...]]] = []

    # extend each D[i][1] to n-1 with a right extension; again push successors r+1 on pop
    for i in range(1, n - 1):
        if not D[i]:
            continue
        prev_cost, prev_piv = D[i][0]
        cost = prev_cost + alpha * errR[i, n - 1] + beta
        push_Hn(Hn, cost, i, 1, np.append(prev_piv, n - 1))

    # also consider all single-line "both-ends" candidates (no DP successor)
    for i in range(0, n - 1):
        for k in range(i + 1, n):
            cost = alpha * errA[i, k] + beta
            push_Hn(Hn, cost, -1, 0, np.array([0, k, n - 1], dtype=int))

    # extract top-q final candidates, pushing successors for DP-derived items
    finals: list[tuple[float, np.ndarray]] = []
    seen_keys: set[tuple[int, ...]] = set()
    while Hn and len(finals) < q:
        val, _, i, r, piv_t = heapq.heappop(Hn)
        piv = np.array(piv_t, dtype=int)

        # normalise pivots: ensure [0, ..., n-1], strictly increasing, unique
        piv = np.unique(np.clip(piv, 0, n - 1))
        if piv[0] != 0:
            piv = np.insert(piv, 0, 0)
        if piv[-1] != n - 1:
            piv = np.append(piv, n - 1)

        key = tuple(piv.tolist())
        if key not in seen_keys:
            finals.append((float(val), piv))
            seen_keys.add(key)

        # push successor when derived from D[i][r] (r>=1) and next rank exists
        if i >= 1 and r < len(D[i]):
            next_prev_cost, next_prev_piv = D[i][r]
            nxt_cost = next_prev_cost + alpha * errR[i, n - 1] + beta
            push_Hn(Hn, nxt_cost, i, r + 1, np.append(next_prev_piv, n - 1))

    finals.sort(key=lambda x: x[0])
    return finals[:q]


def stage1_dp(y: np.ndarray, q: int, alpha: float, beta: float) -> list[tuple[float, np.ndarray]]:
    """
    stage-1 "vanilla" DP exactly as in the paper/repo: uses O(n^3) error-table build (no prefix sums)
    and the heap+rank loop to produce the top-q candidates under alpha*err + beta*k.
    """
    if len(y) < 2:
        return [(0.0, np.array([0], dtype=int))]
    err, errL, errR, errA = build_error_tables_no_prefix(y)
    return _dp_with_heaps_and_ranks(err, errL, errR, errA, alpha=alpha, beta=beta, q=q)



def stage1_dp_prefix(y: np.ndarray, q: int, alpha: float, beta: float) -> list[tuple[float, np.ndarray]]:
    """
    fast stage-1: same dp+heaps but with prefix-sum error tables for O(1) segment sse.
    """
    if len(y) < 2:
        return [(0.0, np.array([0], dtype=int))]
    err, errL, errR, errA = build_error_tables_with_prefix(y)
    return _dp_with_heaps_and_ranks(err, errL, errR, errA, alpha=alpha, beta=beta, q=q)



def stage1_rdp(y: np.ndarray, stage1_candidates: int, beta: float) -> list[tuple[float, np.ndarray]]:
    """
    heuristic stage-1 using ramer-douglas-peucker to propose pivot sets; scored by l2+beta*k.
    Generates candidates more efficiently than DP, however, optimal candidate not guaranteed. 
    Yields faster results but candidates tend to have higher k than with DP. 
    """
    n = len(y)
    x = np.arange(n)
    candidates: list[tuple[float, np.ndarray]] = []
    if n < 2:
        return [(0.0, np.array([0], dtype=int))]

    # sample epsilons geometrically to obtain a spread of ks
    epsilons = np.geomspace(1e-6, max(1e-6, float(np.ptp(y))), num=stage1_candidates)
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

    # keep the best per k
    best_per_k: dict[int, tuple[float, np.ndarray]] = {}
    for cost, piv in candidates:
        k = int(piv.size - 1)
        if (k not in best_per_k) or (cost < best_per_k[k][0]):
            best_per_k[k] = (cost, piv)
    out = list(best_per_k.values())
    out.sort(key=lambda x: x[0])
    return out


def ors_candidates(y: np.ndarray, params: ORSParams) -> list[tuple[float, np.ndarray]]:
    """
    stage-1 dispatcher that returns a list of (cost_es, pivots). dp modes use l2 by construction.
    """
    mode = params.stage1_mode.lower()
    if mode == "dp":
        return stage1_dp(y, q=params.q, alpha=params.alpha, beta=params.beta)
    elif mode == "dp_prefix":
        return stage1_dp_prefix(y, q=params.q, alpha=params.alpha, beta=params.beta)   # your existing fast DP
    elif mode == "rdp":
        return stage1_rdp(y, stage1_candidates=params.stage1_candidates, beta=params.beta)
    else:
        raise ValueError(f"unknown stage1_mode: {params.stage1_mode}")


# ----------------------------------------- stage-2: robustness --------------------------------------- #

def interpolate_from_pivots(T: int, pivots: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    interpolates a length-T series linearly between pivot values.
    """
    x = np.arange(T, dtype=float)
    return np.interp(x, pivots, pv)


def interpolate_batch_from_pivots(T: int, pivots: np.ndarray, pv_batch: np.ndarray) -> np.ndarray:
    """
    vectorised linear interpolation for a batch of pivot-value sets.
    """
    R = pv_batch.shape[0]
    out = np.empty((R, T), dtype=float)
    # piecewise fill per segment
    for s in range(len(pivots) - 1):
        i0, i1 = int(pivots[s]), int(pivots[s + 1])
        if i1 <= i0:
            continue
        span = i1 - i0
        xs = np.arange(span + 1, dtype=float)
        y0 = pv_batch[:, [s]]
        y1 = pv_batch[:, [s + 1]]
        m = (y1 - y0) / span
        seg = y0 + m * xs  # (R, span+1)
        out[:, i0:i1 + 1] = seg
    # fix potential rounding at the tail
    out[:, -1] = pv_batch[:, -1]
    return out


def fragility_uniform_band_batched(bundle, model, pivots: np.ndarray, sts_y: np.ndarray, *,
                                   R: int, eps: float, base_label: int,
                                   power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int,
                                   power_weight: float, decay_lambda: float, t_min_eval: int,
                                   Y_abs_true: np.ndarray, threshold: float, seed: Optional[int]) -> float:
    """
    estimates fragility by sampling uniform ±epsilon noise at pivot values and classifying each perturbation.
    identical to the paper's "uniform band at pivots", batched for efficiency."""
    rng = np.random.default_rng(seed)
    k = len(pivots) - 1
    T = sts_y.size
    pv = sts_y[pivots].astype(float)                                # (k+1,)
    noise = rng.uniform(-eps, +eps, size=(R, k + 1)).astype(float)  # (R, k+1)
    pv_batch = pv[None, :] + noise                                  # (R, k+1)
    power_batch = interpolate_batch_from_pivots(T, pivots, pv_batch)  # (R, T)

    errs = macro_rmse_for_power_batch(bundle, model, power_batch,
                                      power_scaler=power_scaler, soc_scaler=soc_scaler,
                                      idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
                                      power_weight=power_weight, decay_lambda=decay_lambda,
                                      t_min_eval=t_min_eval, Y_abs_true=Y_abs_true)
    labels = (errs > float(threshold)).astype(int)
    flips = np.count_nonzero(labels != int(base_label))
    return float(flips) / float(R)


# ----------------------------------------- main driver ------------------------------------------------ #

def ors(bundle, model: torch.nn.Module, params: ORSParams, *,
                      power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int,
                      power_weight: float, decay_lambda: float, threshold: float) -> dict:
    """
    runs ors end-to-end: stage-1 candidate generation, same-label filtering, robustness estimation,
    and final selection with objective alpha*err + beta*k + gamma*frag in original units.
    """
    T = bundle.length
    y = np.asarray(bundle.true_power_unscaled, dtype=float)
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # stage-1 candidates
    cands = ors_candidates(y, params)
    if not cands:
        # degenerate fallback
        piv = np.array([0, T - 1], dtype=int)
        sts = interpolate_from_pivots(T, piv, y[piv])
        return dict(obj=0.0, k=1, piv=piv, sts=sts, frag=0.0, err=0.0, label="normal")

    # base label on original series
    base_lbl, base_err, Y_abs_true = base_label_from_bundle(
        bundle, power_scaler=power_scaler, soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
        power_weight=power_weight, decay_lambda=decay_lambda,
        t_min_eval=params.t_min_eval, threshold=threshold
    )

    # epsilon selection
    if params.epsilon_mode == "fraction":
        eps = float(params.epsilon_value) * max(1e-9, (y_max - y_min))
    elif params.epsilon_mode == "kw":
        eps = float(params.epsilon_value)
    else:
        raise ValueError(f"unknown epsilon_mode: {params.epsilon_mode}")

    # paper-exact theorem 2 gap diagnostic (only meaningful for dp)
    if params.stage1_mode == "dp" and len(cands) >= 2:
        q_eff = min(params.q, len(cands))
        d = float(cands[q_eff - 1][0] - cands[0][0]) if q_eff >= 2 else float("inf")
        if params.gamma > d:
            print(f"[ORS] warning: gamma={params.gamma:.4g} > gap d={d:.4g}; optimality not guaranteed. "
                  f"consider increasing q or reducing gamma.")

    # evaluate all candidates
    best: dict | None = None
    outcomes: list[ORSOutcome] = []
    for cost_es, piv in cands:
        k = int(len(piv) - 1)
        if k < int(params.min_k) or k > int(params.max_k):
            continue

        sts = interpolate_from_pivots(T, piv, y[piv])
        l2_err = float(np.sum((y - sts) ** 2))

        # label and error for the unperturbed simplification
        lbl_sts, err_sts = classify_macro_rmse_from_power(
            bundle, model, sts, power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
            power_weight=power_weight, decay_lambda=decay_lambda,
            t_min_eval=params.t_min_eval, Y_abs_true=Y_abs_true, threshold=threshold
        )
        if lbl_sts != base_lbl:
            # skip candidates that do not keep the original classification
            continue

        frag = fragility_uniform_band_batched(
            bundle, model, pivots=piv, sts_y=sts, R=params.R, eps=eps, base_label=base_lbl,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
            power_weight=power_weight, decay_lambda=decay_lambda,
            t_min_eval=params.t_min_eval, Y_abs_true=Y_abs_true, threshold=threshold,
            seed=params.seed
        )
        # final objective; note cost_es == alpha*err + beta*k from stage-1
        total = float(cost_es + params.gamma * frag)

        outcome = ORSOutcome(
            k_opt=k, keep_mask=np.isin(np.arange(T), piv), simplified_power=sts,
            robust_prob=1.0 - frag, simplified_label=("abnormal" if lbl_sts == 1 else "normal"),
            simplified_error=err_sts, l2_err=l2_err, objective=total
        )
        outcomes.append(outcome)

        if (best is None) or (total < best["obj"]):
            best = dict(obj=total, k=k, piv=piv, sts=sts, frag=float(frag),
                        err=float(err_sts), label=("abnormal" if lbl_sts == 1 else "normal"))

    # if everything was filtered out by the label constraint, fall back to the lowest objective ignoring the constraint
    if best is None:
        # pick the minimal total without the label constraint
        for cost_es, piv in cands:
            k = int(len(piv) - 1)
            if k < int(params.min_k) or k > int(params.max_k):
                continue
            sts = interpolate_from_pivots(T, piv, y[piv])
            l2_err = float(np.sum((y - sts) ** 2)) 
            lbl_sts, err_sts = classify_macro_rmse_from_power(
                bundle, model, sts, power_scaler=power_scaler, soc_scaler=soc_scaler,
                idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
                power_weight=power_weight, decay_lambda=decay_lambda,
                t_min_eval=params.t_min_eval, Y_abs_true=Y_abs_true, threshold=threshold
            )
            frag = fragility_uniform_band_batched(
                bundle, model, pivots=piv, sts_y=sts, R=params.R, eps=eps, base_label=base_lbl,
                power_scaler=power_scaler, soc_scaler=soc_scaler,
                idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
                power_weight=power_weight, decay_lambda=decay_lambda,
                t_min_eval=params.t_min_eval, Y_abs_true=Y_abs_true, threshold=threshold,
                seed=params.seed
            )

             # compute the error metric for the unperturbed simplification
            if params.stage2_err_metric == "l2":
                err_for_obj = l2_err
            elif params.stage2_err_metric == "mrmse":
                err_for_obj = err_sts
            else:
                raise ValueError(f"unknown stage2_err_metric: {params.stage2_err_metric}")

            # final objective recomputed with chosen metric
            total = float(params.alpha * err_for_obj + params.beta * k + params.gamma * frag)
            cand = dict(obj=total, k=int(len(piv) - 1), piv=piv, sts=sts, frag=float(frag),
                        err=float(err_sts), label=("abnormal" if lbl_sts == 1 else "normal"))
            if (best is None) or (cand["obj"] < best["obj"]):
                best = cand

    return best


# ----------------------------------------- diagnostics ------------------------------------------------ #

def summarize_candidates_table(outcomes: list[ORSOutcome]) -> pd.DataFrame:
    """
    builds a summary dataframe over evaluated candidates for debugging and reporting.
    """
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

