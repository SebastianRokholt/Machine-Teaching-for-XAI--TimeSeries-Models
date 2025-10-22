# mt4xai/ors.py
# Portions adapted from: https://github.com/BrigtHaavardstun/kSimplification (MIT License)
# by Brigt Håvardstun. The ORS algorithm was proposed in the 2024 research paper 
# "Optimal Robust Simplifications for Explaining Time Series Classifications" (2024) 
# by Brigt Håvardstun, Cèsar Ferri and Jan Arne Telle.

# ----------------------------------------- imports, config and outputs ----------------------------------------- #
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, List, Tuple, Dict, Iterable
import heapq
import numpy as np
import pandas as pd
import torch
from rdp import rdp
from .plot import reconstruct_abs_from_bundle
from .inference import inverse_targets_np, predict_residuals, reconstruct_abs_from_residuals_batch



@dataclass
class ORSParams:
    """Configuration parameters for Optimal Robust Simplification (ORS).

    This dataclass separates parameters according to which Stage-1 generator uses them:
    DP / DP-prefix only:
      - dp_q: number of top-ranked pivot sets to keep in the heap (controls search breadth)
      - dp_alpha: stage-1 weight on L2 error in α·err + β·k
    RDP only:
      - rdp_stage1_candidates: number of ε-probes (RDP runs) to try
    Shared across all three modes:
      - stage1_mode: which generator to use ("dp", "dp_prefix", "rdp")
      - stage2_err_metric: metric for final selection in stage 2 of ORP ("l2" or "mrmse")
      - beta, gamma: weights for k and fragility in the final objective
      - R, epsilon_mode, epsilon_value: robustness sampling configuration
      - seed, min_k, max_k, t_min_eval, model_id: misc. control parameters
    """
    stage1_mode: str = "dp_prefix"
    stage2_err_metric: str = "l2"
    dp_q: int = 500
    dp_alpha: float = 0.01
    rdp_stage1_candidates: int = 50
    beta: float = 3.0
    gamma: float = 0.05
    R: int = 10000
    epsilon_mode: str = "fraction"
    epsilon_value: float = 0.3
    seed: Optional[int] = 42
    min_k: int = 1
    max_k: int = 100
    t_min_eval: int = 1
    model_id: Optional[str] = None


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


# ----------------------------------------- macro-rmse classifier -------------------------------------- #

def build_true_abs_from_series(power_true: np.ndarray, soc_true: np.ndarray, H: int) -> np.ndarray:
    """Build absolute targets Y_abs[t,h,c] aligned with horizon h in original units."""
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
    """Compute macro-RMSE across horizons with exponential decay, in original units."""
    T, H, _ = P_abs.shape
    w_h = np.exp(-float(decay_lambda) * np.arange(H, dtype=float))
    w_h = w_h / np.sum(w_h)
    vals: List[float] = []
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
def macro_rmse_for_power_batch(bundle,
                               model: torch.nn.Module,
                               power_batch_kw: np.ndarray,
                               *,
                               power_scaler,
                               soc_scaler,
                               idx_power_inp: int,
                               idx_soc_inp: int,
                               power_weight: float,
                               decay_lambda: float,
                               t_min_eval: int,
                               Y_abs_true: np.ndarray) -> np.ndarray:
    """Run a batched forward pass for perturbed power series and return macro-RMSE per perturbation."""
    T = bundle.length
    R = power_batch_kw.shape[0]
    X_base = bundle.X_sample.clone()                              # (T, F) scaled
    p_scaled = power_scaler.transform(power_batch_kw.reshape(-1, 1)).reshape(R, T)
    Xs = X_base.unsqueeze(0).repeat(R, 1, 1).contiguous()         # (R, T, F)
    Xs[:, :, idx_power_inp] = torch.from_numpy(p_scaled.astype(np.float32)).to(Xs)

    lengths = torch.tensor([T] * R, dtype=torch.long)
    device = next(model.parameters()).device
    P_res = predict_residuals(model, Xs.to(device), lengths, device=device)                     # (R, T, H, C)
    P_abs_scaled = reconstruct_abs_from_residuals_batch(Xs, P_res, idx_power_inp, idx_soc_inp)  # (R,T,H,2)
    P_abs = inverse_targets_np(P_abs_scaled.cpu().numpy(), power_scaler, soc_scaler)            # (R, T, H, 2)

    errs = np.zeros(R, dtype=float)
    for r in range(R):
        errs[r] = macro_rmse_from_abs(P_abs[r], Y_abs_true, power_weight, decay_lambda, t_min_eval)
    return errs


def classify_macro_rmse_from_power(bundle,
                                   model: torch.nn.Module,
                                   power_kw: np.ndarray,
                                   *,
                                   power_scaler,
                                   soc_scaler,
                                   idx_power_inp: int,
                                   idx_soc_inp: int,
                                   power_weight: float,
                                   decay_lambda: float,
                                   t_min_eval: int,
                                   Y_abs_true: np.ndarray,
                                   threshold: float) -> Tuple[int, float]:
    """Classify a power curve by macro-RMSE threshold; return (label_int, err)."""
    errs = macro_rmse_for_power_batch(bundle, model, power_kw[None, :],
                                      power_scaler=power_scaler, soc_scaler=soc_scaler,
                                      idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
                                      power_weight=power_weight, decay_lambda=decay_lambda,
                                      t_min_eval=t_min_eval, Y_abs_true=Y_abs_true)
    err = float(errs[0])
    return (1 if err > float(threshold) else 0), err


def base_label_from_bundle(bundle,
                           *,
                           power_scaler,
                           soc_scaler,
                           idx_power_inp: int,
                           idx_soc_inp: int,
                           power_weight: float,
                           decay_lambda: float,
                           t_min_eval: int,
                           threshold: float) -> Tuple[int, float, np.ndarray]:
    """Obtain base label+error of the unmodified session using stored predictions."""
    power_true = np.asarray(bundle.true_power_unscaled, dtype=float)
    soc_true = np.asarray(bundle.true_soc_unscaled, dtype=float)
    Y_abs_true = build_true_abs_from_series(power_true, soc_true, bundle.horizon)

    P_abs_scaled = reconstruct_abs_from_bundle(bundle, idx_power_inp, idx_soc_inp).numpy()
    P_abs = inverse_targets_np(P_abs_scaled, power_scaler, soc_scaler)

    err = macro_rmse_from_abs(P_abs, Y_abs_true, power_weight, decay_lambda, t_min_eval)
    lbl = 1 if err > float(threshold) else 0
    return lbl, err, Y_abs_true


# ----------------------------------------- stage-1: dp and rdp ---------------------------------------- #

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


def _dp_with_heaps_and_ranks(err: np.ndarray, errL: np.ndarray, errR: np.ndarray, errA: np.ndarray,
                             alpha: float, beta: float, q: int) -> List[Tuple[float, np.ndarray]]:
    """Stage-1 DP with heaps+ranks; returns top-q (cost_es, pivots)."""
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
        push_Hn(Hn, cost, i, 1, np.append(prev_piv, n - 1))

    # also consider single-line both-ends candidates
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


def stage1_dp(y: np.ndarray, q: int, alpha: float, beta: float) -> List[Tuple[float, np.ndarray]]:
    """Vanilla DP: build O(n^3) error tables; return top-q candidates under alpha*err + beta*k."""
    if len(y) < 2:
        return [(0.0, np.array([0], dtype=int))]
    err, errL, errR, errA = build_error_tables_no_prefix(y)
    return _dp_with_heaps_and_ranks(err, errL, errR, errA, alpha=alpha, beta=beta, q=q)


def stage1_dp_prefix(y: np.ndarray, q: int, alpha: float, beta: float) -> List[Tuple[float, np.ndarray]]:
    """Fast DP-prefix: same DP+heaps with prefix sums for O(1) segment SSE queries."""
    if len(y) < 2:
        return [(0.0, np.array([0], dtype=int))]
    err, errL, errR, errA = build_error_tables_with_prefix(y)
    return _dp_with_heaps_and_ranks(err, errL, errR, errA, alpha=alpha, beta=beta, q=q)


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


def ors_candidates(y: np.ndarray, params: ORSParams) -> Optional[List[Tuple[float, np.ndarray]]]:
    """Dispatch Stage-1 candidate generation according to mode."""
    mode = params.stage1_mode.lower()
    if mode == "dp":
        return stage1_dp(y, q=params.dp_q, alpha=params.dp_alpha, beta=params.beta)
    elif mode == "dp_prefix":
        return stage1_dp_prefix(y, q=params.dp_q, alpha=params.dp_alpha, beta=params.beta)
    elif mode == "rdp":
        return stage1_rdp(y, stage1_candidates=params.rdp_stage1_candidates, beta=params.beta)
    else:
        raise ValueError(f"unknown stage1_mode: {params.stage1_mode}")


# ----------------------------------------- stage-2: robustness --------------------------------------- #

def interpolate_from_pivots(T: int, pivots: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """Interpolate a length-T series linearly between pivot values."""
    x = np.arange(T, dtype=float)
    return np.interp(x, pivots, pv)


def interpolate_batch_from_pivots(T: int, pivots: np.ndarray, pv_batch: np.ndarray) -> np.ndarray:
    """Vectorized linear interpolation for a batch of pivot-value sets."""
    R = pv_batch.shape[0]
    out = np.empty((R, T), dtype=float)
    for s in range(len(pivots) - 1):
        i0, i1 = int(pivots[s]), int(pivots[s + 1])
        if i1 <= i0:
            continue
        span = i1 - i0
        xs = np.arange(span + 1, dtype=float)
        y0 = pv_batch[:, [s]]
        y1 = pv_batch[:, [s + 1]]
        m = (y1 - y0) / span
        seg = y0 + m * xs
        out[:, i0:i1 + 1] = seg
    out[:, -1] = pv_batch[:, -1]
    return out


def fragility_uniform_band_batched(bundle,
                                   model: torch.nn.Module,
                                   pivots: np.ndarray,
                                   sts_y: np.ndarray,
                                   *,
                                   R: int,
                                   eps: float,
                                   base_label: int,
                                   power_scaler,
                                   soc_scaler,
                                   idx_power_inp: int,
                                   idx_soc_inp: int,
                                   power_weight: float,
                                   decay_lambda: float,
                                   t_min_eval: int,
                                   Y_abs_true: np.ndarray,
                                   threshold: float,
                                   seed: Optional[int]) -> float:
    """Estimate fragility by sampling uniform ±epsilon at pivot values and reclassifying."""
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


# ----------------------------------------- helpers ---------------------------------------------------- #

def _epsilon_from_range(y_min: float, y_max: float, params: ORSParams) -> float:
    """Compute epsilon for robustness sampling based on params."""
    if params.epsilon_mode == "fraction":
        return float(params.epsilon_value) * max(1e-9, (y_max - y_min))
    elif params.epsilon_mode == "kw":
        return float(params.epsilon_value)
    else:
        raise ValueError(f"unknown epsilon_mode: {params.epsilon_mode}")


def _k_span(cands: Iterable[Tuple[float, np.ndarray]]) -> Tuple[Optional[int], Optional[int]]:
    """Return (k_min, k_max) over a candidate iterable."""
    ks = [int(len(p) - 1) for _, p in (cands or [])]
    return (min(ks), max(ks)) if ks else (None, None)


def _session_id(bundle) -> Optional[int]:
    """Best-effort session id fetch for logging."""
    try:
        sid = getattr(bundle, "session_id", None)
        return int(sid) if sid is not None else None
    except Exception:
        return None


# ----------------------------------------- main driver ------------------------------------------------ #

def ors(bundle,
        model: torch.nn.Module,
        params: ORSParams,
        *,
        power_scaler,
        soc_scaler,
        idx_power_inp: int,
        idx_soc_inp: int,
        power_weight: float,
        decay_lambda: float,
        threshold: float) -> Optional[Dict]:
    """Run ORS end-to-end with fallbacks, keeping the label-consistency constraint.

    The pipeline:
      1) Stage-1 candidate generation (DP/DP-prefix/RDP).
      2) Base label on the original series.
      3) Filter candidates that do NOT keep the base label (constraint).
      4) Robustness estimation (uniform ±epsilon at pivots).
      5) Stage-2 selection with objective: alpha*err + beta*k + gamma*frag, where
         - alpha = dp_alpha for DP modes; for RDP, alpha is implicit in stage-1
           since its cost is l2 + beta*k (final selection still uses gamma*frag).

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
      power_weight, decay_lambda, threshold: macro-RMSE configuration.

    Returns:
      dict with keys {obj,k,piv,sts,frag,err,label} on success; None if no valid candidate.
    """
    T = bundle.length
    y = np.asarray(bundle.true_power_unscaled, dtype=float)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    sid = _session_id(bundle)

    # ---- Stage-1 candidates (initial) ----
    cands = ors_candidates(y, params)
    if cands is None:
        # explicit None: log + return degenerate k=1 for safety as requested
        print(f"[ORS][warn] sid={sid} got cands=None (mode={params.stage1_mode}, dp_q={params.dp_q}, "
              f"rdp_stage1_candidates={params.rdp_stage1_candidates}). Returning k=1 fallback.")
        piv = np.array([0, T - 1], dtype=int)
        sts = interpolate_from_pivots(T, piv, y[piv])
        return dict(obj=0.0, k=1, piv=piv, sts=sts, frag=0.0, err=0.0, label="normal")

    # base label on original series
    base_lbl, base_err, Y_abs_true = base_label_from_bundle(
        bundle,
        power_scaler=power_scaler, soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
        power_weight=power_weight, decay_lambda=decay_lambda,
        t_min_eval=params.t_min_eval, threshold=threshold
    )

    # epsilon for robustness sampling
    eps = _epsilon_from_range(y_min, y_max, params)

    # theorem-2 gap diagnostic (DP only)
    if params.stage1_mode == "dp" and len(cands) >= 2:
        q_eff = min(params.dp_q, len(cands))
        d = float(cands[q_eff - 1][0] - cands[0][0]) if q_eff >= 2 else float("inf")
        if params.gamma > d:
            print(f"[ORS] warning: gamma={params.gamma:.4g} > gap d={d:.4g}; optimality not guaranteed. "
                  f"consider increasing dp_q or reducing gamma.")

    def evaluate_candidates(cands_list: List[Tuple[float, np.ndarray]], p: ORSParams) -> Optional[Dict]:
        """Evaluate Stage-1 candidates under constraints; return best dict or None."""
        best: Optional[Dict] = None
        for cost_es, piv in cands_list:
            k = int(len(piv) - 1)
            if k < int(p.min_k) or k > int(p.max_k):
                continue

            sts = interpolate_from_pivots(T, piv, y[piv])
            l2_err = float(np.sum((y - sts) ** 2))

            lbl_sts, err_sts = classify_macro_rmse_from_power(
                bundle, model, sts,
                power_scaler=power_scaler, soc_scaler=soc_scaler,
                idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
                power_weight=power_weight, decay_lambda=decay_lambda,
                t_min_eval=p.t_min_eval, Y_abs_true=Y_abs_true, threshold=threshold
            )
            if lbl_sts != base_lbl:
                continue  # KEEP the label-consistency constraint (no relaxation)

            frag = fragility_uniform_band_batched(
                bundle, model, pivots=piv, sts_y=sts, R=p.R, eps=eps, base_label=base_lbl,
                power_scaler=power_scaler, soc_scaler=soc_scaler,
                idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
                power_weight=power_weight, decay_lambda=decay_lambda,
                t_min_eval=p.t_min_eval, Y_abs_true=Y_abs_true, threshold=threshold,
                seed=p.seed
            )

            # Stage-1 cost_es is:
            #  - DP: dp_alpha*l2 + beta*k
            #  - RDP: l2 + beta*k (dp_alpha not in its stage-1)
            # Stage-2 adds gamma*frag
            total = float(cost_es + p.gamma * frag)

            cand = dict(
                obj=total, k=k, piv=piv, sts=sts, frag=float(frag),
                err=float(err_sts), label=("abnormal" if lbl_sts == 1 else "normal")
            )
            if (best is None) or (cand["obj"] < best["obj"]):
                best = cand
        return best

    # ----- evaluate initial candidates -----
    best = evaluate_candidates(cands, params)
    if best is not None:
        return best

    # ----- Fallback #1: log + rerun with dp_q*2 (if DP) and beta*4 -----
    k_min, k_max = _k_span(cands)
    print(f"[ORS][warn] sid={sid} no valid candidates after constraints "
          f"(mode={params.stage1_mode}, k_span={k_min}..{k_max}, dp_q={params.dp_q}, beta={params.beta}). "
          f"Trying fallback #1.")

    p_fb1 = replace(params, beta=params.beta * 4.0)
    if params.stage1_mode in {"dp", "dp_prefix"}:
        p_fb1 = replace(p_fb1, dp_q=max(1, params.dp_q * 2))

    cands1 = ors_candidates(y, p_fb1)
    if cands1 is None:
        print(f"[ORS][warn] sid={sid} fallback #1 produced cands=None; proceeding to fallback #2.")
    else:
        best = evaluate_candidates(cands1, p_fb1)
        if best is not None:
            return best

    # ----- Fallback #2: switch to RDP with large stage1_candidates -----
    k_min1, k_max1 = _k_span(cands1 or [])
    print(f"[ORS][warn] sid={sid} still no valid candidates "
          f"(fb#1 k_span={k_min1}..{k_max1}, dp_q={getattr(p_fb1, 'dp_q', None)}, beta={p_fb1.beta}). "
          f"Trying fallback #2 (rdp).")

    rdp_count = max(200, 3 * (params.max_k - params.min_k + 1))
    p_fb2 = replace(params, stage1_mode="rdp", rdp_stage1_candidates=rdp_count)

    cands2 = ors_candidates(y, p_fb2)
    if cands2 is None:
        print(f"[ORS][warn] sid={sid} fallback #2 (rdp) produced cands=None; giving up for this session.")
        return None

    best = evaluate_candidates(cands2, p_fb2)
    if best is not None:
        return best

    k_min2, k_max2 = _k_span(cands2)
    print(f"[ORS][warn] sid={sid} fallbacks exhausted (rdp k_span={k_min2}..{k_max2}); skipping session.")
    return None


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
