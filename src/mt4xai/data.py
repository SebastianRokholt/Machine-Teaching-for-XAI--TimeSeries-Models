# src/mt4xai/data.py
from dataclasses import dataclass
from typing import List, Dict, Optional
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


# def get_nth_batch(loader, n: int):
#     it = iter(loader)
#     for _ in range(n):
#         next(it)
#     return next(it)


# @torch.no_grad()
# def fetch_sample_bundle(model: nn.Module, loader,
#                         batch_index: int, sample_index: int, device: torch.device,
#                         power_scaler, soc_scaler,
#                         idx_power_inp: int, idx_soc_inp: int) -> SampleBundle:
#     """
#     Build a SampleBundle from a DataLoader batch (mirrors modelling notebook).
#     - Forward pass on GPU, lengths kept on CPU int64 (pack_padded requirement).
#     - Returns CPU tensors inside the bundle for plotting/post-proc.
#     """
#     model.eval()
#     batch = get_nth_batch(loader, batch_index)

#     session_ids = None
#     if len(batch) == 4:
#         session_ids, Xb, Yb, Ls = batch
#     else:
#         Xb, Yb, Ls = batch

#     if sample_index >= Xb.shape[0]:
#         raise IndexError(f"sample_index {sample_index} out of range for batch {batch_index} (size={Xb.shape[0]}).")

#     X_dev = Xb.to(device, non_blocking=True)
#     Ls_cpu = Ls.to(dtype=torch.long, device="cpu")  # lengths must be CPU int64
#     P_dev, _ = model(X_dev, Ls_cpu)

#     T = Ls[sample_index].item()
#     P_s = P_dev[sample_index, :T].cpu()
#     Y_s = Yb[sample_index, :T].cpu()
#     X_s = Xb[sample_index, :T].cpu()

#     power_true = power_scaler.inverse_transform(X_s[:, [idx_power_inp]].numpy()).ravel()
#     soc_true   = soc_scaler.inverse_transform(  X_s[:, [idx_soc_inp  ]].numpy()).ravel()

#     H, C = P_s.shape[1], P_s.shape[2]
#     sid = None if session_ids is None else session_ids[sample_index]
#     return SampleBundle(
#         batch_index=batch_index, sample_index=sample_index,
#         length=T, horizon=H, num_targets=C,
#         X_sample=X_s, Y_sample=Y_s, P_sample=P_s,
#         true_power_unscaled=power_true, true_soc_unscaled=soc_true,
#         session_id=sid
#     )


def reconstruct_abs_from_bundle(bundle: SampleBundle, idx_power_inp: int, idx_soc_inp: int) -> torch.Tensor:
    """
    Reconstruct absolute predictions in *scaled* space:
      P_abs[t, h, c] = X[t, c_base] + P_res[t, h, c], aligned at t+h.
    Returns (T, H, C), CPU torch tensor.
    """
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)  # (T,1,2)
    return bundle.P_sample + base


def split_data(df: pd.DataFrame, test_size: float=0.2, validation_size: float=0.1, random_seed: int | None=None):
    """
    Reproduce the modelling split: group by `charging_id`, first carve out test,
    then split the remainder into train/val. Keeps sessions intact.
    Returns train_df, val_df, test_df.
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