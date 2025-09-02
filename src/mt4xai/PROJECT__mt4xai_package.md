# src/mt4xai/data.py
```python
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math, random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

RANDOM_SEED = 42  # same as modelling notebook


@dataclass
class Session:
    """stores one session in scaled space for fast perturbation inference.
    fields
    - X_scaled: (T,F) scaled inputs
    - T, F, H: length, input size, horizon
    - idx_power, idx_soc: input indices in X
    - power_scaler, soc_scaler: fitted scalers for inverse transforms
    """
    X_scaled: torch.Tensor
    T: int
    F: int
    H: int
    idx_power: int
    idx_soc: int
    power_scaler: MinMaxScaler
    soc_scaler: MinMaxScaler


@dataclass
class SessionPredsBundle:
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
    true_soc_unscaled: np.ndarray
    session_id: Optional[str] = None 


def make_session_template_df(df_scaled,
                             sid: int,
                             *,
                             input_features: list[str],
                             horizon: int,
                             power_scaler: MinMaxScaler,
                             soc_scaler: MinMaxScaler,
                             idx_power_inp: int,
                             idx_soc_inp: int,
                             sort_by: str = "minutes_elapsed") -> Session:
    """
    builds a Session from a scaled DataFrame.
    """
    s = df_scaled[df_scaled["charging_id"] == sid].sort_values(sort_by)
    X = torch.tensor(s[input_features].to_numpy(np.float32))  # (T,F) scaled
    T, F = int(X.shape[0]), int(X.shape[1])
    return Session(
        X_scaled=X.contiguous(), T=T, F=F, H=int(horizon),
        idx_power=int(idx_power_inp), idx_soc=int(idx_soc_inp),
        power_scaler=power_scaler, soc_scaler=soc_scaler,
    )


def get_nth_batch(loader, n: int):
    it = iter(loader)
    for _ in range(n):
        next(it)
    return next(it)


@torch.no_grad()
def fetch_session_preds_bundle(model: nn.Module, loader,
                               batch_index: int | None, sample_index: int | None,
                               device: torch.device,
                               power_scaler, soc_scaler,
                               idx_power_inp: int, idx_soc_inp: int,
                               session_id: int | None = None) -> SessionPredsBundle:
    """
    builds a SessionPredsBundle from a loader by either:
      - selecting a specific (batch_index, sample_index), or
      - selecting a specific charging session by 'session_id'
    the two modes are mutually exclusive.

    params
    - model, loader, device, scalers, idx_power_inp/idx_soc_inp: as before
    - batch_index, sample_index: choose a sample by indices
    - session_id: choose a sample by charging_id

    returns
    - SessionPredsBundle with CPU tensors for plotting/post-proc
    """
    use_id = session_id is not None
    use_idx = (batch_index is not None) and (sample_index is not None)
    if (use_id and use_idx) or (not use_id and not use_idx):
        raise ValueError("provide either session_id OR (batch_index, sample_index), not both.")

    model.eval()

    if use_id:
        sid, Xb, Yb, Ls = get_session_from_loader(loader, session_id=session_id)
        # shapes from helper are batched (1, T, F) etc.
        X_dev = Xb.to(device, non_blocking=True)
        L_cpu = Ls.to(dtype=torch.long, device="cpu")
        P_dev, _ = model(X_dev, L_cpu)  # (1, T, H, C)

        T = int(Ls[0].item())
        P_s = P_dev[0, :T].cpu()
        Y_s = Yb[0, :T].cpu()
        X_s = Xb[0, :T].cpu()

        power_true = power_scaler.inverse_transform(X_s[:, [idx_power_inp]].numpy()).ravel()
        soc_true   = soc_scaler.inverse_transform(  X_s[:, [idx_soc_inp  ]].numpy()).ravel()

        H, C = P_s.shape[1], P_s.shape[2]
        return SessionPredsBundle(
            batch_index=0, sample_index=0,
            length=T, horizon=H, num_targets=C,
            X_sample=X_s, Y_sample=Y_s, P_sample=P_s,
            true_power_unscaled=power_true, true_soc_unscaled=soc_true,
            session_id=int(sid) if isinstance(sid, (int, np.integer)) else sid
        )

    batch = get_nth_batch(loader, int(batch_index))

    session_ids = None
    if len(batch) == 4:
        session_ids, Xb, Yb, Ls = batch
    else:
        Xb, Yb, Ls = batch

    if int(sample_index) >= Xb.shape[0]:
        raise IndexError(f"sample_index {sample_index} out of range for batch {batch_index} (size={Xb.shape[0]}).")

    X_dev = Xb.to(device, non_blocking=True)
    L_cpu = Ls.to(dtype=torch.long, device="cpu")  # lengths must be CPU int64
    P_dev, _ = model(X_dev, L_cpu)

    T = Ls[int(sample_index)].item()
    P_s = P_dev[int(sample_index), :T].cpu()
    Y_s = Yb[int(sample_index), :T].cpu()
    X_s = Xb[int(sample_index), :T].cpu()

    power_true = power_scaler.inverse_transform(X_s[:, [idx_power_inp]].numpy()).ravel()
    soc_true   = soc_scaler.inverse_transform(  X_s[:, [idx_soc_inp  ]].numpy()).ravel()

    H, C = P_s.shape[1], P_s.shape[2]
    sid = None if session_ids is None else session_ids[int(sample_index)]
    return SessionPredsBundle(
        batch_index=int(batch_index), sample_index=int(sample_index),
        length=T, horizon=H, num_targets=C,
        X_sample=X_s, Y_sample=Y_s, P_sample=P_s,
        true_power_unscaled=power_true, true_soc_unscaled=soc_true,
        session_id=sid
    )



def reconstruct_abs_from_bundle(bundle: SessionPredsBundle, idx_power_inp: int, idx_soc_inp: int) -> torch.Tensor:
    """
    Reconstruct absolute predictions in *scaled* space:
      P_abs[t, h, c] = X[t, c_base] + P_res[t, h, c], aligned at t+h.
    Returns (T, H, C), CPU torch tensor.
    """
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)  # (T,1,2)
    return bundle.P_sample + base


def split_data(df: pd.DataFrame, test_size: float=0.2, 
               validation_size: float=0.1, random_seed: int | None=None) -> Tuple[pd.DataFrame]:
    """
    Uses GroupShuffleSplit to create train/val/test sets.
    Groups by `charging_id`, first carve out test, then split the remainder into train/val. Keeps sessions intact.
    Returns DataFrames train_df, val_df, test_df.
    """
    if not random_seed:
        random_seed = RANDOM_SEED

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["charging_id"]))
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    adj_val = validation_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val, random_state=random_seed)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["charging_id"]))
    train_df = train_val_df.iloc[train_idx]
    val_df   = train_val_df.iloc[val_idx]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scalers_on_train(train_df: pd.DataFrame, cols_to_scale: List[str]) -> Dict[str, MinMaxScaler]:
    """
    Fit MinMax scalers on `train_df` (per modelling notebook practice).
    Returns a dict of per-column scalers for simple, explicit inverse_transform later.
    """
    scalers = {}
    for c in cols_to_scale:
        s = MinMaxScaler()
        x = train_df[[c]].astype(float).values
        s.fit(x)
        scalers[c] = s
    return scalers

def apply_scalers(df: pd.DataFrame, scalers: Dict[str, MinMaxScaler]) -> pd.DataFrame:
    """
    Apply previously fit scalers column-wise. Leaves other columns intact.
    """
    df = df.copy()
    for c, s in scalers.items():
        df[c] = s.transform(df[[c]].astype(float).values)
    return df

class ChargingSessionDataset(Dataset):
    """
    Yields full sessions as variable-length sequences.
    X: (T, input_size)
    Y: (T, H, C) residual targets: y_{t+h} - y_t, aligned with modelling code.
    """
    def __init__(self, df: pd.DataFrame, input_features: List[str], target_features: List[str],
                 horizon: int):
        self.groups = []
        self.input_features = input_features
        self.target_features = target_features
        self.horizon = horizon

        for sid, g in df.groupby("charging_id"):
            g = g.sort_values("minutes_elapsed").reset_index(drop=True)
            x = g[input_features].to_numpy(dtype=np.float32)            # (T, F)
            y_abs = g[target_features].to_numpy(dtype=np.float32)       # (T, C)
            T = len(g)
            # Build residual target tensor: (T, H, C) with valid region masked later
            y = np.zeros((T, horizon, y_abs.shape[1]), dtype=np.float32)
            for h in range(1, horizon+1):
                y[:-h, h-1, :] = y_abs[h:, :] - y_abs[:-h, :]
            self.groups.append((sid, x, y, T))

    def __len__(self): return len(self.groups)

    def __getitem__(self, idx: int):
        sid, x, y, T = self.groups[idx]
        return sid, x, y, T  # keep session_id for post-hoc grouping

class LengthBucketSampler(Sampler[List[int]]):
    """
    Simple length-bucketing sampler for variable-length sessions.
    """
    def __init__(self, lengths: List[int], batch_size: int=8, shuffle: bool=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sorted_indices = np.argsort(lengths).tolist()
        # cut into batches of similar lengths
        self.batches = [self.sorted_indices[i:i+batch_size]
                        for i in range(0, len(self.sorted_indices), batch_size)]
        if self.shuffle: random.shuffle(self.batches)

    def __iter__(self):
        if self.shuffle: random.shuffle(self.batches)
        for b in self.batches: yield b

    def __len__(self): return math.ceil(len(self.sorted_indices) / self.batch_size)

def session_collate_fn(batch):
    """
    Pads variable-length sessions to (B, T_max, .). Carries session_ids and lengths.
    """
    session_ids, all_x, all_y, lengths = zip(*batch)
    B = len(all_x)
    T_max = max(lengths)
    F    = all_x[0].shape[1]
    H    = all_y[0].shape[1]
    C    = all_y[0].shape[2]

    X = np.zeros((B, T_max, F), dtype=np.float32)
    Y = np.zeros((B, T_max, H, C), dtype=np.float32)
    for i, (x, y, L) in enumerate(zip(all_x, all_y, lengths)):
        X[i, :L] = x
        Y[i, :L] = y

    return (list(session_ids),
            torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.tensor(lengths, dtype=torch.long))

def build_loader(df: pd.DataFrame, input_features: List[str], target_features: List[str],
                 horizon: int, batch_size: int=16, shuffle: bool=False, num_workers: int=0):
    ds = ChargingSessionDataset(df, input_features, target_features, horizon)
    lengths = [T for (_, _, _, T) in ds.groups]
    sampler = LengthBucketSampler(lengths, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=session_collate_fn,
                      num_workers=num_workers, pin_memory=True)


def get_session_from_loader(test_loader: DataLoader, 
                            session_idx: int | None=None, batch_idx: int | None = None,
                            session_id: int | None=None) -> tuple:
    """
    Retrieve a charging session from test_loader by index or by charging_id.

    Args:
            session_idx (int): Index of the session in the loader batch.
            session_id (int, optional): Charging session ID to retrieve.

    Returns:
            tuple: (session_id, X, Y, length) for the selected session.
    """
    if (session_idx is None and session_id is None) or (session_idx is not None and session_id is not None):
        raise ValueError("Provide either session_id or (batch_idx, session_idx), but not both.")
    
    # Iterate over batches to find the session
    for i, batch in enumerate(test_loader):
        batch_session_ids, Xb, Yb, lengths = batch

        if session_id is not None:
            if session_id in batch_session_ids:
                idx = batch_session_ids.index(session_id)
                session_id = batch_session_ids[idx]
                X = Xb[idx:idx+1]
                Y = Yb[idx:idx+1]
                L = lengths[idx:idx+1]
                return (session_id, X, Y, L)
        elif batch_idx is not None and session_idx is not None:
            if i == batch_idx:
                session_id = batch_session_ids[session_idx]
                X = Xb[session_idx:session_idx+1]
                Y = Yb[session_idx:session_idx+1]
                L = lengths[session_idx:session_idx+1]
                return (session_id, X, Y, L)

    if session_id is not None:
        raise ValueError(f"Session ID {session_id} not found in any batch.")
    else:
        raise IndexError(f"Batch index {batch_idx} or session index {session_idx} out of range.")
    

@torch.inference_mode()
def make_bundle_from_session_df(model: nn.Module, df_scaled, sid: str,
                             device: torch.device,
                             input_features: List[str],
                             target_features: List[str],
                             horizon: int,
                             power_scaler, soc_scaler,
                             idx_power_inp: int, idx_soc_inp: int) -> SessionPredsBundle:
    """
    Build a SessionPredsBundle directly from a single session DataFrame.
    """
    g = df_scaled[df_scaled["charging_id"] == sid].sort_values("minutes_elapsed")
    X = torch.tensor(g[input_features].to_numpy(np.float32)) # (T, F)
    y_abs = g[target_features].to_numpy(np.float32)  # (T, C)
    T = y_abs.shape[0]
    Y = np.zeros((T, horizon, len(target_features)), dtype=np.float32)   # residuals
    for h in range(1, horizon+1):
        Y[:-h, h-1, :] = y_abs[h:, :] - y_abs[:-h, :]

    X_dev = X.unsqueeze(0).to(device, non_blocking=True)  # (1, T, F)
    L_cpu = torch.tensor([T], dtype=torch.long, device="cpu")
    P_dev, _ = model(X_dev, L_cpu)   # (1, T, H, C)
    P = P_dev.squeeze(0).cpu()  # (T, H, C)

    power_true = power_scaler.inverse_transform(X[:, [idx_power_inp]].numpy()).ravel()
    soc_true = soc_scaler.inverse_transform( X[:, [idx_soc_inp  ]].numpy()).ravel()

    return SessionPredsBundle(
        batch_index=0, sample_index=0,
        length=T, horizon=horizon, num_targets=len(target_features),
        X_sample=X.cpu(), Y_sample=torch.from_numpy(Y), P_sample=P,
        true_power_unscaled=power_true, true_soc_unscaled=soc_true,
        session_id=sid
    )
```

# src/mt4xai/plot.py
```python
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from .data import SessionPredsBundle


def plot_full_session(bundle: SessionPredsBundle, power_scaler, soc_scaler,
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

# ------------------------------- Curve simplification ------------------------------------------ #
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .data import SessionPredsBundle, reconstruct_abs_from_bundle


def plot_session_with_simplification(bundle: SessionPredsBundle,
                                     power_scaler, soc_scaler,
                                     idx_power_inp: int, idx_soc_inp: int,
                                     simpl_power_unscaled: np.ndarray,
                                     *,
                                     session_id: int | None = None,
                                     k: int | None = None,
                                     threshold: float | None = None,
                                     simp_error: float | None = None,
                                     orig_error: float | None = None, 
                                     label: str | None = None,
                                     decay_lambda: float | None = None,
                                     noise_std_kw: float | None = None,
                                     robust_tau: float | None = None,
                                     figsize: Tuple[float, float] = (12.0, 5.0),
                                     dpi: int = 110, 
                                     t_min_eval: int = 1):
    """
    plots true power, all-horizon predicted power, and an ors-simplified power curve.
    adds a descriptive title and a caption with classifier/ors settings.
    """
    T, H = bundle.length, bundle.horizon
    t = np.arange(T)
    true = np.asarray(bundle.true_power_unscaled, dtype=float)
    sid = int(session_id) if session_id is not None else bundle.session_id

    plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(x=t, y=true, color="black", linewidth=2.2, label="True power")

    # multi-horizon predictions (power)
    palette = sns.color_palette("deep", n_colors=H)
    P_abs = reconstruct_abs_from_bundle(bundle, idx_power_inp, idx_soc_inp)
    for h0 in range(H):
        i_valid = np.arange(t_min_eval, T - (h0 + 1))
        if i_valid.size == 0:
            continue
        t_pred = i_valid + (h0 + 1)
        preds = power_scaler.inverse_transform(P_abs[i_valid, h0, 0].numpy().reshape(-1, 1)).ravel()
        sns.lineplot(x=t_pred, y=preds, linestyle="--", linewidth=1.6, color=palette[h0],
                     label=f"H={h0+1}", marker="o", markersize=2.5)

    # ors simplification
    sns.lineplot(x=t, y=simpl_power_unscaled, color="tab:red", linewidth=2.6, label="ORS simplification")

    # required title format
    ttl = f"ORS Simplification of Charging Session {sid}. Classification = {label}, k = {k}"
    plt.title(ttl)

    # figure description (caption)
    cap_parts = []
    if label is not None: cap_parts.append(f"Classification: {label}")
    if simp_error is not None:  cap_parts.append(f"prediction error on simplification = {simp_error:.3f}")
    if orig_error is not None:  cap_parts.append(f"prediction error on original = {orig_error:.3f}")
    if threshold is not None: cap_parts.append(f"classification threshold = {threshold}")
    if k is not None:  cap_parts.append(f"k = {k}")
    if decay_lambda is not None: cap_parts.append(f"λ = {decay_lambda}")
    if noise_std_kw is not None: cap_parts.append(f"noise_std_kw = {noise_std_kw}")
    if robust_tau is not None: cap_parts.append(f"robust_tau = {robust_tau}")
    fig_desc = ", ".join(cap_parts)

    plt.xlabel("Time index"); plt.ylabel("Power (kW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # add bottom caption with a little extra bottom margin
    plt.subplots_adjust(bottom=0.18)
    plt.gcf().text(0.01, 0.02, fig_desc, fontsize=9, ha="left")
    plt.show()


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
```

# src/mt4xai/model.py
```python
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import weight_norm
import torch.nn.functional as F


# LSTM
class MultiHorizonLSTM(nn.Module):
    """LSTM multi-horizon residual model"""
    def __init__(self, input_size: int, hidden_dim: int, horizon: int, num_targets: int,
                 num_layers: int, dropout: float=0.0):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, horizon * num_targets)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor):
        packed_x = rnn_utils.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, out_lengths = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        out = self.linear(out).view(out.shape[0], out.shape[1], self.horizon, self.num_targets)
        # enforce non-negative SOC deltas via softplus
        out = torch.cat([out[:, :, :, 0:1], F.softplus(out[:, :, :, 1:2])], dim=-1)
        return out, out_lengths
    

def build_model_lstm(cfg):
    return MultiHorizonLSTM(
        input_size=len(cfg["input_features"]),
        hidden_dim=int(cfg["hidden_dim"]),
        horizon=cfg["horizon"],
        num_targets=len(cfg["target_features"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(cfg["device"])


def load_lstm_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    model = MultiHorizonLSTM(
        input_size=len(checkpoint["input_features"]),
        hidden_dim=int(cfg["hidden_dim"]),
        horizon=cfg["horizon"],
        num_targets=len(checkpoint["target_features"]),
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# TCN
class MultiHorizonTCN(nn.Module):
    """
    WaveNet-style TCN:
      - input 1x1 projection
      - L residual blocks with exponentially increasing dilations
      - global skip accumulation -> head
    """
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int,
                 kernel_size: int, horizon: int, num_targets: int,
                 dropout: float=0.0, dilation_growth: int = 2):
        super().__init__()
        self.horizon, self.num_targets = horizon, num_targets

        self.input_proj = weight_norm(nn.Conv1d(input_size, hidden_dim, kernel_size=1))
        self.blocks = nn.ModuleList([
            SkipTCNBlock(hidden_dim, kernel_size, dilation=(dilation_growth ** i), dropout=dropout)
            for i in range(num_layers)
        ])
        # combine all skips, a small mixer, then predict all horizons/targets
        self.post = nn.Sequential(
            nn.ReLU(),
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(hidden_dim, horizon * num_targets, kernel_size=1)

    @staticmethod
    def receptive_field(kernel_size, num_layers, dilation_growth=2):
        # RF = 1 + (k-1) * sum_{i=0}^{L-1} (growth^i)
        return 1 + (kernel_size - 1) * sum(dilation_growth ** i for i in range(num_layers))

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor):
        # x: (B, T, F) -> (B, C, T)
        B, T, RF = x.shape
        h = self.input_proj(x.transpose(1, 2))
        skip_sum = None
        for blk in self.blocks:
            h, s = blk(h)
            skip_sum = s if skip_sum is None else (skip_sum + s)

        h = self.post(skip_sum)
        y = self.head(h).transpose(1, 2).view(B, T, self.horizon, self.num_targets)
        # enforces non-negative SOC deltas with softplus
        y = torch.cat([y[:, :, :, 0:1], F.softplus(y[:, :, :, 1:2])], dim=-1)
        return y, seq_lengths


class SkipTCNBlock(nn.Module):
    """Residual block that also emits a skip tensor."""
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        c = channels
        self.conv1 = DSConv1d(c, c, kernel_size, dilation, dropout)
        self.conv2 = DSConv1d(c, c, kernel_size, dilation, dropout)
        self.residual = weight_norm(nn.Conv1d(c, c, kernel_size=1))
        self.skip = weight_norm(nn.Conv1d(c, c, kernel_size=1))

    def forward(self, x):  # x: (B, C, T)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return self.residual(h) + x, self.skip(h)


class DSConv1d(nn.Module):
    """Causal depthwise-separable 1D conv with weight norm."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.k, self.d = kernel_size, dilation
        self.dw = weight_norm(nn.Conv1d(in_ch, in_ch, kernel_size,
                                        groups=in_ch, padding=0, dilation=dilation, bias=True))
        self.pw = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True))
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = _causal_pad(x, self.k, self.d)
        x = self.dw(x)
        x = F.relu(x)
        x = self.pw(x)
        return self.do(x)

def _causal_pad(x, k, d):
    # Left-pad so output stays length T and remains causal
    return F.pad(x, ((k-1)*d, 0 ))


def build_model_tcn(cfg):
    return MultiHorizonTCN(
        input_size=len(cfg["input_features"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        horizon=cfg["horizon"],
        num_targets=len(cfg["target_features"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(cfg["device"])
```

# src/mt4xai/inference.py
```python
from __future__ import annotations  # postpones evaluation of type hints to speed up imports
import torch
import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.preprocessing import MinMaxScaler
from .data import Session, SessionPredsBundle, reconstruct_abs_from_bundle, make_bundle_from_session_df

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


_RWSE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_EPS: float = 1e-8


def list_unique_session_ids(df_scaled: pd.DataFrame) -> np.ndarray:
    """Returns sorted unique charging_id values as a plain int ndarray."""
    return np.asarray(np.sort(np.asarray(df_scaled["charging_id"].unique())), dtype=int)

def _bundle_iter(model,
                 df_scaled: pd.DataFrame,
                 sids: Iterable[int],
                 *,
                 device,
                 input_features: list[str],
                 target_features: list[str],
                 horizon: int,
                 power_scaler,
                 soc_scaler,
                 idx_power_inp: int,
                 idx_soc_inp: int,
                 t_min_eval: int):
    """Yields (sid, residuals[T,H,C], T) for each session id, residuals in original units cropped at t>=t_min_eval."""
    for sid in sids:
        b = make_bundle_from_session_df(
            model=model, df_scaled=df_scaled, sid=int(sid), device=device,
            input_features=input_features, target_features=target_features, horizon=horizon,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp)
        # b.Y_sample and b.P_sample are residual targets/preds in *scaled* units.
        # Convert to residuals in original units by inverse-transforming absolute series.
        # Reuse inverse logic via your existing helpers:
        from .inference import inverse_targets_np  # local import to avoid cycles
        base = b.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1).numpy()          # (T,1,2) scaled
        Y_abs = base + b.Y_sample.numpy()                                                # (T,H,2) scaled
        P_abs = base + b.P_sample.numpy()                                                # (T,H,2) scaled
        Y = inverse_targets_np(Y_abs, power_scaler, soc_scaler)                          # (T,H,2)
        P = inverse_targets_np(P_abs, power_scaler, soc_scaler)                          # (T,H,2)
        R = (Y - P)                                                                      # (T,H,2)
        Tlen = R.shape[0]
        t0 = min(int(t_min_eval), max(0, Tlen - 1))
        yield int(sid), R[t0:], int(Tlen)


def fit_rwse_robust_scalers(model,
                            val_scaled_df: pd.DataFrame,
                            *,
                            device,
                            input_features: list[str],
                            target_features: list[str],
                            horizon: int,
                            power_scaler,
                            soc_scaler,
                            idx_power_inp: int,
                            idx_soc_inp: int,
                            t_min_eval: int = 1,
                            cache_key: str = "default") -> tuple[np.ndarray, np.ndarray]:
    """Fits per-(h,c) median and MAD of residuals on the validation set; returns (m, mad) with shape (H,C)."""
    key = ("robust_scalers", cache_key)
    if key in _RWSE_CACHE:
        return _RWSE_CACHE[key]

    sids = list_unique_session_ids(val_scaled_df)
    all_R: list[np.ndarray] = []
    for _, R, _ in _bundle_iter(model, val_scaled_df, sids,
                                device=device, input_features=input_features, target_features=target_features,
                                horizon=horizon, power_scaler=power_scaler, soc_scaler=soc_scaler,
                                idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp, t_min_eval=t_min_eval):
        if R.size:
            all_R.append(np.asarray(R, dtype=np.float64))

    if not all_R:
        raise RuntimeError("No residuals collected from validation set for RWSE scalers.")

    R_cat = np.asarray(np.concatenate(all_R, axis=0), dtype=np.float64)    # (sum_T, H, C)
    m   = np.median(R_cat, axis=0)                                         # (H,C)
    mad = np.median(np.abs(R_cat - m), axis=0).astype(np.float64) + _EPS   # (H,C)

    _RWSE_CACHE[key] = (m, mad)
    return m, mad


def rwse_score_from_bundle(
    bundle: SessionPredsBundle,
    m: np.ndarray,
    mad: np.ndarray,
    *,
    power_scaler: MinMaxScaler,
    soc_scaler: MinMaxScaler,
    idx_power_inp: int,
    idx_soc_inp: int,
    w_h: np.ndarray,
    w_c: np.ndarray,
    cap: float = 5.0,
    t_min_eval: int = 1,
) -> tuple[float, int]:
    """Compute RWSE for one bundle using calibrated medians and MADs; returns (score, session_length)."""
    # deltas in scaled space
    Y = np.asarray(bundle.Y_sample, dtype=np.float64)  # (T,H,C)
    P = np.asarray(bundle.P_sample, dtype=np.float64)  # (T,H,C)
    # reconstruct absolute (scaled), then inverse to original units
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1).numpy()  # (T,1,2)
    Y_abs = inverse_targets_np(base + Y, power_scaler, soc_scaler)  # (T,H,2)
    P_abs = inverse_targets_np(base + P, power_scaler, soc_scaler)  # (T,H,2)

    T = int(Y.shape[0])
    t0 = min(int(t_min_eval), max(0, T - 1))
    R = (Y_abs - P_abs)[t0:]  # (T',H,2)
    if R.size == 0:
        return 0.0, T

    m   = np.asarray(m,   dtype=np.float64)   # (H,2)
    mad = np.asarray(mad, dtype=np.float64)   # (H,2)
    w_h = np.asarray(w_h, dtype=np.float64)   # (H,)
    w_c = np.asarray(w_c, dtype=np.float64)   # (2,)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z  = np.abs(R - m[None, :, :]) / mad[None, :, :]  # (T',H,2)
        Z  = np.clip(Z, a_min=None, a_max=cap)
        Zw = Z * w_h[None, :, None] * w_c[None, None, :]
        Zw = np.nan_to_num(Zw, nan=0.0, posinf=cap, neginf=0.0)

    score = Zw.sum(axis=(1, 2)).mean()
    return float(score), T


def compute_session_MRMSE(model, loader, device: torch.device, power_scaler: MinMaxScaler,
                           soc_scaler: MinMaxScaler, power_weight: float, idx_power_inp: int, 
                           idx_soc_inp: int,  t_min_eval: int=0,
                           horizon_weights_decay: float | None=None) -> pd.DataFrame:
    """
    returns a dataframe with columns: charging_id, error, length.
    supports optional horizon weighting (exponential decay).
    """
    rows = []

    H = None
    try:
        H = int(getattr(getattr(loader, "dataset", None), "horizon", None))
    except Exception:
        H = None

    w_h = None
    if H is not None and horizon_weights_decay is not None:
        w_h = make_horizon_weights(H, decay=float(horizon_weights_decay))

    for session_ids, Xb, Yb, lengths in loader:
        P_res = predict_residuals(model, Xb, lengths, device=device)

        P_abs_scaled = reconstruct_abs_from_residuals_batch(Xb, P_res, idx_power_inp, idx_soc_inp)
        base = torch.stack([Xb[..., idx_power_inp], Xb[..., idx_soc_inp]], dim=-1).unsqueeze(2)
        Y_abs_scaled = base + Yb

        P_abs_np = inverse_targets_np(P_abs_scaled.cpu().numpy(), power_scaler, soc_scaler)
        Y_abs_np = inverse_targets_np(Y_abs_scaled.cpu().numpy(), power_scaler, soc_scaler)

        errs = macro_rmse_per_session(torch.from_numpy(P_abs_np),
                                      torch.from_numpy(Y_abs_np),
                                      lengths,
                                      power_weight=power_weight, 
                                      t_min_eval=t_min_eval,
                                      w_h=w_h)

        for sid, e, L in zip(session_ids, errs, lengths.tolist()):
            rows.append({"charging_id": sid, "error": float(e), "length": int(L)})
    return pd.DataFrame(rows)



def compute_session_RWSE(model, df_scaled: pd.DataFrame, 
                         device, input_features: list[str], target_features: list[str], 
                         power_scaler, soc_scaler, idx_power_inp: int, idx_soc_inp: int, m: np.ndarray, 
                         mad: np.ndarray, horizon: int, horizon_weights_decay: float=0.4, 
                         cap: float = 5.0, t_min_eval: int = 1, 
                         session_ids: Iterable[int] | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns: charging_id, length, error (RWSE)."""
    rows: list[dict] = []

    if not session_ids:
        session_ids = list(list_unique_session_ids(df_scaled))
    else: 
        session_ids = list(map(int, session_ids))

    # Calculate horizon and feature weights
    w_h = make_horizon_weights(horizon, decay=horizon_weights_decay)
    w_c = make_feature_weights(target_features)

    for sid in session_ids:
        b = make_bundle_from_session_df(
            model=model, df_scaled=df_scaled, sid=int(sid), device=device,
            input_features=input_features, target_features=target_features, horizon=horizon,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp)
        s, L = rwse_score_from_bundle(
            b, m, mad,
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp,
            w_h=w_h, w_c=w_c, cap=cap, t_min_eval=t_min_eval,
        )
        rows.append({"charging_id": int(sid), "length": int(L), "error": float(s)})
    return pd.DataFrame(rows)


def make_horizon_weights(H: int, decay: float = 0.4) -> np.ndarray:
    """Returns length-H exponential horizon weights that sum to 1."""
    w = np.exp(-decay * np.arange(int(H), dtype=float))
    return (w / w.sum()).astype(float)

def make_feature_weights(target_features: list[str]) -> np.ndarray:
    """Returns uniform feature weights over target features that sum to 1."""
    C = int(len(target_features))
    return np.ones(C, dtype=float) / float(C)


def compute_bundle_error(bundle: SessionPredsBundle,
                         power_scaler, soc_scaler,
                         power_weight: float,
                         idx_power_inp: int, idx_soc_inp: int, 
                         t_min_eval: int=1) -> float:
    """
    Macro-averaged RMSE across horizons, mixing power/SOC with `power_weight`.
    Matches the AD pipeline semantics (compute_session_errors).
    """
    T, H = bundle.length, bundle.horizon

    # scaled → absolute (scaled)
    P_abs_scaled = reconstruct_abs_from_bundle(bundle, idx_power_inp, idx_soc_inp)  # (T,H,2)
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)  # (T,1,2)
    Y_abs_scaled = base + bundle.Y_sample  # (T,H,2)

    # inverse-transform to original units
    P_abs = inverse_targets_np(P_abs_scaled.numpy(), power_scaler, soc_scaler)  # (T,H,2)
    Y_abs = inverse_targets_np(Y_abs_scaled.numpy(), power_scaler, soc_scaler)  # (T,H,2)

    per_h = []
    for h in range(H):
        end = T - (h + 1)
        if end <= t_min_eval:
            continue
        diff = P_abs[t_min_eval:end, h, :] - Y_abs[t_min_eval:end, h, :]
        rmse_c = np.sqrt(np.mean(diff**2, axis=0))  # (2,)
        val = power_weight * rmse_c[0] + (1.0 - power_weight) * rmse_c[1]
        per_h.append(val)
    return float(np.mean(per_h)) if per_h else float("nan")


def percentile_threshold(errors: np.ndarray, pct_thr: float=95.0) -> float:
    """
    Takes a distribution of errors and returns the error value at the given percentile (e.g. the 95th percentile).
    The inverse of percentile_of_threshold
    """
    return round(float(np.nanpercentile(errors, pct_thr)), 4)

def percentile_of_threshold(all_errs_sorted, thr: float) -> float:
    """
    Takes a threshold value and returns what percentile of the error distribution it corresponds to.
    The inverse of percentile_threshold"""
    idx = np.searchsorted(all_errs_sorted, thr, side="right")
    return 100.0 * idx / max(1, len(all_errs_sorted))


def classify_by_threshold(df_errs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Adds `label` column: 'abnormal' if error > threshold else 'normal'.
    """
    df = df_errs.copy()
    df["label"] = np.where(df["error"] > threshold, "abnormal", "normal")
    return df.sort_values("error", ascending=True).reset_index(drop=True)


# TODO: Write a new function that is more user-friendly (less parameters)
# for classifying a single charging session
def classify_session(model,
                           df_scaled: pd.DataFrame,
                           sid: int,
                           *,
                           device: torch.device,
                           input_features: list[str],
                           target_features: list[str],
                           horizon: int,
                           power_scaler: MinMaxScaler,
                           soc_scaler: MinMaxScaler,
                           idx_power_inp: int,
                           idx_soc_inp: int,
                           power_weight: float = 1.0,
                           decay: float = 0.2,
                           t_min_eval: int = 1,
                           threshold: float = 10.0) -> tuple[int, float]:
    """
    classifies a single charging session by Macro-RMSE with horizon decay.
    returns (label, error), where label=1 means abnormal (error > threshold), else 0.

    re-uses existing helpers make_bundle_from_session, compute_bundle_error, make_horizon_weights
    pass power_weight=1.0 to only classify based on power predictions.
    """
    bundle = make_bundle_from_session_df(
        model=model, df_scaled=df_scaled, sid=int(sid), device=device,
        input_features=input_features, target_features=target_features, horizon=horizon,
        power_scaler=power_scaler, soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp, idx_soc_inp=idx_soc_inp
    )

    # compute macro-rmse with horizon decay via compute_bundle_error by injecting weights
    # compute_bundle_error internally averages horizons uniformly, so we map decay to weights
    # by temporarily scaling per-horizon errors before averaging.
    # simplest route: reuse compute_bundle_error and set power_weight; decay handled by weights below
    # we reconstruct absolute preds/targets and compute rmse horizon-wise
    T, H = bundle.length, bundle.horizon
    w_h = make_horizon_weights(H, decay=decay)

    # re-implement the averaging with w_h 
    P_abs_scaled = reconstruct_abs_from_bundle(bundle, idx_power_inp, idx_soc_inp)  # (T,H,2)
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)  # (T,1,2)
    Y_abs_scaled = base + bundle.Y_sample  # (T,H,2)
    P_abs = inverse_targets_np(P_abs_scaled.numpy(), power_scaler, soc_scaler)  # (T,H,2)
    Y_abs = inverse_targets_np(Y_abs_scaled.numpy(), power_scaler, soc_scaler)  # (T,H,2)

    per_h = []
    for h in range(H):
        end = T - (h + 1)
        if end <= t_min_eval:
            continue
        diff = P_abs[t_min_eval:end, h, :] - Y_abs[t_min_eval:end, h, :]
        rmse_c = np.sqrt(np.mean(diff**2, axis=0))
        per_h.append(w_h[h] * (power_weight * rmse_c[0] + (1.0 - power_weight) * rmse_c[1]))

    error = float(np.sum(per_h)) if per_h else float("nan")
    label = int(error > float(threshold))
    return label, error`
```


# src/mt4xai/training.py
```python
# TODO: Move training and tuning loop and training utils here
```

# src/mt4xai/utils.py
```python
# TODO: Move utils here
```