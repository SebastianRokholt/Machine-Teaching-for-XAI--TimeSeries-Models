# mt4xai/ors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
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
    holds tunable parameters for the ors procedure.

    stage1_mode: "dp" uses the exact dynamic-programming + heaps from the paper; "rdp" uses a fast heuristic.
    q: number of least-cost candidates to keep in stage-1 (paper: top-q).
    stage1_candidates: number of rdp candidates to sample if stage1_mode="rdp".
    alpha, beta, gamma: objective weights on error, simplicity (k), and fragility respectively.
    R: number of perturbations for fragility estimation (paper default uses large R, e.g. 10_000).
    epsilon_mode: "fraction" uses epsilon_value * (ymax - ymin); "kw" uses epsilon in kW directly.
    epsilon_value: epsilon value as described by epsilon_mode.
    enforce_same_label: require h(sts) = h(ts) (matches the paper’s constraint).
    seed: rng seed.
    min_k, max_k: candidate k range to consider in stage-2.
    t_min_eval: number of initial time steps to ignore when computing macro-rmse.
    """
    stage1_mode: str = "dp"
    q: int = 100
    stage1_candidates: int = 40

    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0

    R: int = 10_000
    epsilon_mode: str = "fraction"
    epsilon_value: float = 0.1
    enforce_same_label: bool = True

    seed: Optional[int] = 1337
    min_k: int = 1
    max_k: int = 60
    t_min_eval: int = 0


@dataclass
class ORSOutcome:
    """stores a single candidate evaluation for inspection and debugging."""
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
    builds absolute ground-truth targets Y_abs[t,h,c] aligned with horizon h (no scaling).
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
    computes the Macro-RMSE with horizon decay on absolute targets/predictions.
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
    runs a single batched forward pass for a batch of perturbed power series in kW and
    returns a vector of macro-rmse values (length R). uses the modelling helpers to
    reconstruct absolute predictions and inverse-transform to original units.
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
    classifies a power curve (kW) by Macro-RMSE threshold. returns (label_int, err), where
    label_int is 1 for abnormal and 0 for normal.
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
    computes the base label h(ts) and its error using the predictions already stored in the bundle.
    returns (label_int, err, Y_abs_true).
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


def build_error_tables(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    precomputes segment errors err, err_left, err_right, err_all for lines through endpoints.
    source: authors’ kSimplification repo and paper appendix; implemented with prefix sums for O(1) segment queries.
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


def stage1_dp(y: np.ndarray, q: int, alpha: float, beta: float) -> list[tuple[float, np.ndarray]]:
    """
    returns the q least-cost candidates under alpha*err + beta*k using dp + heaps.
    source: Optimal Robust Simplifications for Explaining Time Series Classifications (paper) and
    the kSimplification repository; adapted to python.
    """
    n = len(y)
    if n < 2:
        return [(0.0, np.array([0], dtype=int))]
    err, errL, errR, errA = build_error_tables(y)

    # per-end heaps with up to q candidates each; store as max-heaps via (-cost, (cost, pivots))
    heaps: List[List[tuple[float, tuple[float, List[int]]]]] = [[] for _ in range(n)]
    global_heap: List[tuple[float, tuple[float, List[int]]]] = []

    # initialize with first segment extended left
    for j in range(1, n):
        for i in range(j):
            cost = alpha * errL[i, j] + beta
            piv = [0, j] if i == 0 else [0, i, j]
            cand = (cost, piv)
            heapq.heappush(heaps[j], (-cost, cand))
            if len(heaps[j]) > q:
                heapq.heappop(heaps[j])

    # dp transitions
    for j in range(1, n):
        for i in range(j):
            for _, (c_prev, piv_prev) in heaps[i]:
                cost = c_prev + alpha * err[i, j] + beta
                piv = piv_prev + [j]
                cand = (cost, piv)
                heapq.heappush(heaps[j], (-cost, cand))
                if len(heaps[j]) > q:
                    heapq.heappop(heaps[j])

    # finalize by extending to the end
    for i in range(n - 1):
        for _, (c_prev, piv_prev) in heaps[i]:
            cost = c_prev + alpha * errR[i, n - 1] + beta
            piv = piv_prev + [n - 1]
            cand = (cost, piv)
            heapq.heappush(global_heap, (-cost, cand))
            if len(global_heap) > q:
                heapq.heappop(global_heap)

    # also consider a single segment through (i,k) extended both ways
    for i in range(n - 1):
        for k in range(i + 1, n):
            cost = alpha * errA[i, k] + beta
            piv = [0, k, n - 1] if (i > 0 and k < n - 1) else [0, k, n - 1]
            cand = (cost, piv)
            heapq.heappush(global_heap, (-cost, cand))
            if len(global_heap) > q:
                heapq.heappop(global_heap)

    results = [cand for _, cand in global_heap]
    # ensure pivots are strictly increasing and within [0, n-1]
    cleaned: list[tuple[float, np.ndarray]] = []
    for cost, piv in results:
        piv = np.unique(np.clip(np.asarray(piv, dtype=int), 0, n - 1))
        if piv[0] != 0:
            piv = np.insert(piv, 0, 0)
        if piv[-1] != n - 1:
            piv = np.append(piv, n - 1)
        cleaned.append((float(cost), piv))
    cleaned.sort(key=lambda x: x[0])
    # deduplicate by identical pivot sets
    uniq = []
    seen = set()
    for cost, piv in cleaned:
        key = tuple(piv.tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((cost, piv))
    return uniq[:q]


def stage1_rdp(y: np.ndarray, stage1_candidates: int, beta: float) -> list[tuple[float, np.ndarray]]:
    """
    returns a pool of candidates by varying the rdp epsilon and computing cost_es = err + beta*k.
    source: RDP; used here as a speed-oriented heuristic (not part of the original paper).
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
    dispatches stage-1 according to params.stage1_mode and returns a list of (cost_es, pivots).
    """
    if params.stage1_mode == "dp":
        return stage1_dp(y, q=params.q, alpha=params.alpha, beta=params.beta)
    if params.stage1_mode == "rdp":
        return stage1_rdp(y, stage1_candidates=params.stage1_candidates, beta=params.beta)
    raise ValueError(f"unknown stage1_mode: {params.stage1_mode}")


# ----------------------------------------- stage-2: robustness --------------------------------------- #

def interpolate_from_pivots(T: int, pivots: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    builds a series of length T by linear interpolation between pivot values pv at indices pivots.
    """
    x = np.arange(T, dtype=float)
    return np.interp(x, pivots, pv)


def interpolate_batch_from_pivots(T: int, pivots: np.ndarray, pv_batch: np.ndarray) -> np.ndarray:
    """
    builds a batch (R, T) by linear interpolation between row-wise pivot values.
    implements a compact loop over segments for efficiency.
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
    estimates fragility as in the paper: sample R perturbations by adding iid Unif[-eps, +eps]
    to the k+1 pivot values and classify each perturbed series. batches model inference for speed.
    source: kSimplification repo / paper section on local robustness (uniform band around pivots).
    """
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

def ors_optimal_mrmse(bundle, model: torch.nn.Module, params: ORSParams, *,
                      power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int,
                      power_weight: float, decay_lambda: float, threshold: float) -> dict:
    """
    performs the ors optimization exactly as in the paper:
    stage-1 generates candidates minimizing alpha*err + beta*k (dp top-q or rdp pool);
    stage-2 adds gamma*frag with perturbations in a uniform band at pivots and selects the minimum.
    the classifier h is the macro-rmse threshold rule used in your anomaly detection.

    returns a dict with keys: 'obj', 'k', 'piv', 'sts', 'frag', 'err', 'label'.
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

    # paper theorem 2 gap diagnostic (only meaningful for dp)
    if params.stage1_mode == "dp" and len(cands) >= 2:
        q_eff = min(params.q, len(cands))
        d = float(cands[q_eff - 1][0] - cands[0][0]) if q_eff >= 2 else float("inf")
        if params.gamma > d:
            print(f"[ORS] warning: gamma={params.gamma:.4g} > gap d={d:.4g}; optimality may not be guaranteed. "
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
        if params.enforce_same_label and (lbl_sts != base_lbl):
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
            total = float(cost_es + params.gamma * frag)
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

