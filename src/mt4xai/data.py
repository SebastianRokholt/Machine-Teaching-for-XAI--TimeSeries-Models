# src/mt4xai/data.py
from dataclasses import dataclass
from typing import Any, List, Dict, Literal, Sequence, Tuple, Optional
import math, random
from matplotlib.figure import Figure
from matplotlib.pylab import Axes
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


@dataclass(slots=True)
class ChargingSessionPrediction:
    """Holds model predictions linked to a session.

    This container stores both scaled residuals (for training-time diagnostics)
    and convenient unscaled absolute predictions for fast plotting.

    Attributes:
        Y_resid_scaled: Residual targets in scaled space, shape (T, H, C).
        P_resid_scaled: Residual predictions in scaled space, shape (T, H, C).
        power_pred_kw: Optional absolute power prediction in kW, shape (T,).
        soc_pred_pct: Optional absolute SoC prediction in percent, shape (T,).
        horizon: Prediction horizon H.
        num_targets: Number of target channels C.
    """
    Y_resid_scaled: Any
    P_resid_scaled: Any
    horizon: int
    num_targets: int
    power_pred_kw: Optional[np.ndarray] = None
    soc_pred_pct: Optional[np.ndarray] = None

    def numpy_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (Y_resid_scaled, P_resid_scaled) as numpy arrays."""
        to_np = (lambda a: a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a))
        return to_np(self.Y_resid_scaled), to_np(self.P_resid_scaled)

@dataclass
class SessionPredsBundle:
    """
    Deprecated / legacy / temporary. Use ChargingSessionPrediction instead. 
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


@dataclass
class ChargingSessionSimplification:
    """Stores simplifications for power and optionally SoC.
    The session keeps k = number of straight-line segments (knots + 1). Dense overlays are built on demand.
    """
    power_knot_idx: Optional[np.ndarray] = None
    power_knot_val_kw: Optional[np.ndarray] = None
    soc_knot_idx: Optional[np.ndarray] = None
    soc_knot_val_pct: Optional[np.ndarray] = None
    k_power: Optional[int] = None
    k_soc: Optional[int] = None
    kind: Literal["ors", "rdp"] = "ors"

    def densify_power(self, T: int) -> Optional[np.ndarray]:
        """Returns dense power simplification (kW) or None if absent."""
        if self.power_knot_idx is None or self.power_knot_val_kw is None:
            return None
        xi = np.asarray(self.power_knot_idx, dtype=int)
        yi = np.asarray(self.power_knot_val_kw, dtype=float)
        xi = np.clip(xi, 0, max(T - 1, 0))
        x = np.arange(T, dtype=float)
        return np.interp(x, xi.astype(float), yi)

    def densify_soc(self, T: int) -> Optional[np.ndarray]:
        """Returns dense SoC simplification (%) or None if absent."""
        if self.soc_knot_idx is None or self.soc_knot_val_pct is None:
            return None
        xi = np.asarray(self.soc_knot_idx, dtype=int)
        yi = np.asarray(self.soc_knot_val_pct, dtype=float)
        xi = np.clip(xi, 0, max(T - 1, 0))
        x = np.arange(T, dtype=float)
        return np.interp(x, xi.astype(float), yi)
    

@dataclass(slots=True)
class ChargingSession:
    """Canonical representation of a charging session in unscaled units.

    This object owns the raw signals; scaled views are computed on demand by invoking specific methods such as
    features(scaled=True), as_tensor(scaled=true), power_scaled. 
    Optional attributes predictions hold predictions and simplifications holds a knots-only simplification (k = segments = knots - 1).
    """
    session_id: int
    power_kw: np.ndarray
    soc_pct: Optional[np.ndarray] = None
    temp_c: Optional[np.ndarray] = None
    nominal_power_kw: Optional[np.ndarray] = None
    predictions: Optional[ChargingSessionPrediction] = None
    simplification: Optional[ChargingSessionSimplification] = None
    scaler_power: Optional[sklearn.base.BaseEstimator | MinMaxScaler] = None
    scaler_soc: Optional[sklearn.base.BaseEstimator | MinMaxScaler] = None
    scaler_temp: Optional[sklearn.base.BaseEstimator | MinMaxScaler] = None
    scaler_nominal_power: Optional[sklearn.base.BaseEstimator | MinMaxScaler] = None
    # feature_order: tuple[Literal["power", "soc", "temp", "nominal_power"]] = ("power", "soc", "temp", "nominal_power")

    @property
    def T(self) -> int:
        return int(self.power_kw.size)

    # --- scaled views (computed on demand) ---
    def power_scaled(self) -> np.ndarray:
        if self.scaler_power is None:
            raise ValueError("power scaler is missing")
        return self.scaler_power.transform(self.power_kw.reshape(-1, 1)).reshape(-1)

    def soc_scaled(self) -> Optional[np.ndarray]:
        if self.soc_pct is None:
            return None
        if self.scaler_soc is None:
            raise ValueError("soc scaler is missing")
        return self.scaler_soc.transform(self.soc_pct.reshape(-1, 1)).reshape(-1)

    def temp_scaled(self) -> Optional[np.ndarray]:
        if self.temp_c is None:
            return None
        if self.scaler_temp is None:
            raise ValueError("temp scaler is missing")
        return self.scaler_temp.transform(self.temp_c.reshape(-1, 1)).reshape(-1)

    def nominal_power_scaled(self) -> Optional[np.ndarray]:
        if self.nominal_power_kw is None:
            return None
        if self.scaler_nominal_power is None:
            raise ValueError("nominal_power scaler is missing")
        return self.scaler_nominal_power.transform(self.nominal_power_kw.reshape(-1, 1)).reshape(-1)

    # --- feature stacks / tensors ---
    def features(self, *, scaled: bool=False, 
                 include: Sequence[Literal["power", "soc", "temp", "nominal_power"]] | None = None) -> np.ndarray:
        """Constructs a feature matrix
        Args:
            scaled (bool, optional): Whether to apply scaling. Defaults to False.
            include (Sequence): Which features to include. Defaults to None.
        Raises: ValueError: When scaled=True and a requested, present channel lacks a fitted scaler
        Returns: np.ndarray: (T, F) feature matrix in requested order
        """
        order = tuple(include) if include is not None else self.feature_order
        cols: list[np.ndarray] = []
        for name in order:
            if name == "power":
                cols.append((self.power_scaled() if scaled else self.power_kw).reshape(-1, 1))
            elif name == "soc":
                v = self.soc_scaled() if scaled else self.soc_pct
                if v is not None:  # "if v" raised Numpy "truth value of array is ambiguous"
                    cols.append(v.reshape(-1, 1)) 
            elif name == "temp":
                v = self.temp_scaled() if scaled else self.temp_c
                if v is not None: 
                    cols.append(v.reshape(-1, 1)) 
            elif name == "nominal_power":
                v = self.nominal_power_scaled() if scaled else self.nominal_power_kw
                if v is not None: 
                    cols.append(v.reshape(-1, 1)) 
        return np.concatenate(cols, axis=1) if cols else np.empty((self.T, 0), dtype=float)

    def as_tensor(self, *, scaled: bool=False,
                  include: Sequence[Literal["power", "soc", "temp", "nominal_power"]] | None = None,
                  device: Optional[str] = None, dtype: Any = None) -> Tensor:
        """Constructs a Torch tensor (T, F) for modelling / inference.
        Args:
            scaled (bool, optional): Whether to apply scaling. Defaults to False.
            include (Sequence): Which features to include. Defaults to None.
        Returns: torch.Tensor: torch tensor (T, F)"""
        if torch is None:
            raise RuntimeError("torch not available; install PyTorch to use as_tensor.")
        X = self.features(scaled=scaled, include=include)
        t = torch.from_numpy(X)
        if dtype is not None: t = t.to(dtype=dtype)
        if device is not None: t = t.to(device)
        return t
    
    def plot(
        self, *, 
        soc_mode: Literal["none", "raw", "simpl"] = "none", 
        title: str | None = None, 
        y_lim: tuple[float, float] | None = None
        ) -> tuple[Figure, Axes]:
        """Plots session with optional SOC overlay on a right 0–100% axis.
        Args:
            soc_mode: Controls SOC visibility. "none" hides SOC, "raw" uses self.soc_pct,
                "simpl" uses knots from self.simplification to densify SOC.
            title: Optional plot title.
            y_lim: Optional y-axis limit for power in kW as (ymin, ymax).
        Returns:
            Matplotlib figure and axis.
        """
        # local import to avoid circular dependency
        from .plot import plot_raw_session, plot_raw_simpl_session

        self.validate(check_knots=True)
        power = np.asarray(self.power_kw, dtype=float).reshape(-1)
        T = power.size
        # decide SOC series once
        soc: np.ndarray | None = None
        if soc_mode == "raw" and self.soc_pct is not None:
            soc = np.asarray(self.soc_pct, dtype=float).reshape(-1)
            if soc.size != T:
                raise ValueError("power_kw and soc_pct must have equal length")
        elif soc_mode == "simpl" and self.simplification is not None and getattr(self.simplification, "soc_knot_idx", None) is not None:
            sidx = np.asarray(self.simplification.soc_knot_idx, dtype=int)
            sval = np.asarray(self.simplification.soc_knot_val_pct, dtype=float)
            xs = np.arange(T, dtype=float); soc = np.interp(xs, sidx.astype(float), sval).astype(float)
        else:
            soc_mode = "none"

        if self.simplification is None or getattr(self.simplification, "power_knot_idx", None) is None:
            return plot_raw_session(power_kw=power, soc=soc, soc_mode=soc_mode, title=title, y_lim=y_lim)
        else:
            idx = np.asarray(self.simplification.power_knot_idx, dtype=int)
            val = np.asarray(self.simplification.power_knot_val_kw, dtype=float)
            return plot_raw_simpl_session(power_kw=power, idx=idx, val_kw=val, soc=soc, soc_mode=soc_mode, title=title, y_lim=y_lim)
        
    def validate(self, *, check_knots: bool = True) -> None:
        """Validate internal consistency (length alignment, optional knots checks).

        Raises:
            ValueError: If any present channel has a length mismatch, or if attached
                simplification has malformed/unaligned knots when `check_knots=True`.
        """
        T = int(np.asarray(self.power_kw).shape[0])
        for name, arr in [
            ("soc_pct", self.soc_pct),
            ("temp_c", self.temp_c),
            ("nominal_power_kw", self.nominal_power_kw),
        ]:
            if arr is None:
                continue
            if int(np.asarray(arr).shape[0]) != T:
                raise ValueError(f"{name} length mismatch: expected {T}, got {len(arr)}")

        if not check_knots or self.simplification is None:
            return

        simp = self.simplification
        # power knots
        if simp.power_knot_idx is not None or simp.power_knot_val_kw is not None:
            if simp.power_knot_idx is None or simp.power_knot_val_kw is None:
                raise ValueError("power simplification must define both indices and values")
            idx = np.asarray(simp.power_knot_idx, dtype=int)
            val = np.asarray(simp.power_knot_val_kw, dtype=float)
            if idx.ndim != 1 or val.ndim != 1 or idx.size != val.size:
                raise ValueError("power knots must be 1D arrays of equal length")
            if idx.size < 2:
                raise ValueError("need at least two knots to form ≥1 segment")
            if np.any(idx < 0) or np.any(idx > T - 1):
                raise ValueError("power knot indices out of bounds")
            if np.any(np.diff(idx) <= 0):
                raise ValueError("power knot indices must be strictly increasing")
            exp_k = idx.size - 1
            if simp.k_power is not None and int(simp.k_power) != exp_k:
                raise ValueError(f"k_power mismatch: declared {simp.k_power}, inferred {exp_k}")

        # soc knots
        if simp.soc_knot_idx is not None or simp.soc_knot_val_pct is not None:
            if simp.soc_knot_idx is None or simp.soc_knot_val_pct is None:
                raise ValueError("soc simplification must define both indices and values")
            idx = np.asarray(simp.soc_knot_idx, dtype=int)
            val = np.asarray(simp.soc_knot_val_pct, dtype=float)
            if idx.ndim != 1 or val.ndim != 1 or idx.size != val.size:
                raise ValueError("soc knots must be 1D arrays of equal length")
            if idx.size < 2:
                raise ValueError("need at least two soc knots to form ≥1 segment")
            if np.any(idx < 0) or np.any(idx > T - 1):
                raise ValueError("soc knot indices out of bounds")
            if np.any(np.diff(idx) <= 0):
                raise ValueError("soc knot indices must be strictly increasing")
            exp_k = idx.size - 1
            if simp.k_soc is not None and int(simp.k_soc) != exp_k:
                raise ValueError(f"k_soc mismatch: declared {simp.k_soc}, inferred {exp_k}")

    def __post_init__(self) -> None:
        """Post-construction validation"""
        self.power_kw = np.asarray(self.power_kw, dtype=float)
        if self.soc_pct is not None:
            self.soc_pct = np.asarray(self.soc_pct, dtype=float)
            if self.soc_pct.size != self.T:
                raise ValueError("soc_pct length mismatch")
        if self.temp_c is not None:
            self.temp_c = np.asarray(self.temp_c, dtype=float)
            if self.temp_c.size != self.T:
                raise ValueError("temp_c length mismatch")
        if self.nominal_power_kw is not None:
            self.nominal_power_kw = np.asarray(self.nominal_power_kw, dtype=float)
            if self.nominal_power_kw.size != self.T:
                raise ValueError("nominal_power_kw length mismatch")



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
    Reconstruct absolute predictions for power and soc in *scaled* space:
      P_abs[t, h, c] = X[t, c_base] + P_res[t, h, c], aligned at t+h.
    Returns (T, H, C), CPU torch tensor.
    """
    base = bundle.X_sample[:, [idx_power_inp, idx_soc_inp]].unsqueeze(1)  # (T,1,2)
    return bundle.P_sample + base


def split_data(df: pd.DataFrame, test_size: float=0.2, 
               validation_size: float=0.1, random_seed: int | None=42) -> Tuple[pd.DataFrame]:
    """
    Uses GroupShuffleSplit to create train/val/test sets.
    Groups by `charging_id`, first carve out test, then split the remainder into train/val. Keeps sessions intact.
    Returns DataFrames train_df, val_df, test_df.
    """
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
    Returns: tuple: (session_id, X, Y, length) for the selected session.
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