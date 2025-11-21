# src/mt4xai/train.py
import os
import glob
import json
import tempfile
from pathlib import Path
from collections import OrderedDict
from typing import Callable, Optional
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import ray
from ray import ObjectRef
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune import ExperimentAnalysis
# custom imports
from .data import LengthBucketSampler, session_collate_fn
from .model import build_model_lstm, build_model_tcn, horizon_weights
from .inference import macro_rmse_per_session

# -------------------------- MODEL TRAINING & TUNING (HYPERPARAMETER SEARCH) ---------------------- #
def tune_train(
    config,
    num_workers: int, 
    power_min: float, 
    power_max: float, 
    idx_power: int=2,
    train_dataset_ref=None,
    val_dataset_ref=None,
    model_builder: Optional[Callable[[dict], nn.Module]] = None
):
    """ray air trainable with recoverable checkpoints.

    saves and reports a self-contained directory checkpoint every epoch:
      - checkpoint.pt: torch payload with model/optim/best_val/epoch
      - meta.json:      small json with val metric and config echo
    supports resuming via session.get_checkpoint().
    """
    # unpack hyperparams
    lr = float(config["lr"])
    weight_decay = float(config.get("weight_decay", 0.0))
    grad_clip = float(config.get("grad_clip_norm", 0.0))
    num_epochs = int(config["num_epochs"])
    batch_size = int(config.get("batch_size", 64))
    alpha_h = float(config.get("alpha_h", 0.35))
    device = config["device"]

    # inverse min–max constants (power only)
    pmin = torch.tensor(power_min, device=device)
    pmax = torch.tensor(power_max, device=device)

    # resolve datasets (local or object refs)
    train_dataset_local = _resolve_dataset_ref(train_dataset_ref)
    val_dataset_local   = _resolve_dataset_ref(val_dataset_ref)

    assert model_builder is not None, "model_builder must be provided."
    model = model_builder(config)

    optimizer = build_adamw(model.named_parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, verbose=False)
    scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())

    train_loader_local, val_loader_local = _build_loaders_for_trial_from_datasets(
        train_dataset_local, val_dataset_local, batch_size, num_workers
    )

    # resume if checkpoint present
    start_epoch, best_val = 0, float("inf")
    prev_ckpt = session.get_checkpoint()
    if prev_ckpt is not None:
        ckpt_dir = prev_ckpt.to_directory()
        payload = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload: optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        best_val = float(payload.get("best_val", float("inf")))
        model.to(device)
        print(f"[resume] restored epoch={start_epoch}, best_val={best_val:.6f}")

    for epoch in range(start_epoch, num_epochs):
        # training loop
        model.train()
        train_sum, train_count = 0.0, 0
        for _, X_batch, Y_batch, lengths in train_loader_local:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                pred_resid, _ = model(X_batch, lengths)  # (B, T, H, 1)
                B, T, H, C = pred_resid.shape
                assert C == 1, f"expected 1 output channel, got {C}"

                mask_4d = _vectorized_mask(lengths, T_max=T, horizon=H, device=device)
                w_h = horizon_weights(H, alpha_h, device)

                P_abs_s = _reconstruct_abs_from_residuals_batch(pred_resid, X_batch, idx_power)  # (B,T,H,1)
                Y_abs_s = Y_batch + X_batch[..., [idx_power]].unsqueeze(2)  # (B,T,H,1)

                P_kw = inv_minmax_channel_torch(P_abs_s, pmin, pmax, ch=0)
                Y_kw = inv_minmax_channel_torch(Y_abs_s, pmin, pmax, ch=0)

                huber_kw = F.smooth_l1_loss(P_kw, Y_kw, reduction="none")
                loss = ((w_h * huber_kw)[mask_4d]).mean()

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()

            train_sum += float(loss.detach()) * X_batch.size(0)
            train_count += X_batch.size(0)

        train_loss = train_sum / max(train_count, 1)

        # validation loop
        model.eval()
        val_sum, val_count = 0.0, 0
        mrmse_sum, mrmse_n = 0.0, 0
        with torch.no_grad():
            for _, X_batch, Y_batch, lengths in val_loader_local:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)

                pred_resid, _ = model(X_batch, lengths)  # (B, T, H, 1)

                # rebuild absolute in scaled space (power target only)
                P_abs_s = _reconstruct_abs_from_residuals_batch(pred_resid, X_batch, idx_power)
                Y_abs_s = Y_batch + X_batch[..., [idx_power]].unsqueeze(2)

                # val Huber (kW) for monitoring
                P_kw = inv_minmax_channel_torch(P_abs_s, pmin, pmax, ch=0)
                Y_kw = inv_minmax_channel_torch(Y_abs_s, pmin, pmax, ch=0)
                huber_kw = F.smooth_l1_loss(P_kw, Y_kw, reduction="none")

                # 4D mask + horizon weights
                B, T, H, _ = P_abs_s.shape
                mask_4d = _vectorized_mask(lengths, T_max=T, horizon=H, device=device)
                w_h = horizon_weights(H, alpha_h, device)
                vloss = ((w_h * huber_kw)[mask_4d]).mean()

                # Macro-RMSE (per sequence, horizon-weighted, then macro-avg)
                per_seq = macro_rmse_per_session(P_abs_s, Y_abs_s, lengths, pmin, pmax, alpha_h)
                valid = per_seq[~torch.isnan(per_seq)]
                mrmse_sum += float(valid.sum())
                mrmse_n   += int(valid.numel())

                val_sum += float(vloss) * X_batch.size(0)
                val_count += X_batch.size(0)

        val_loss = val_sum / max(val_count, 1)
        val_macro_rmse = mrmse_sum / max(mrmse_n, 1)

        scheduler.step(val_macro_rmse)
        if val_macro_rmse < best_val:
            best_val = val_macro_rmse

        # checkpointing and reporting (val_metric = Macro-RMSE in kW)
        with tempfile.TemporaryDirectory() as ckpt_dir:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val": float(best_val),
                    "config": dict(config),
                },
                os.path.join(ckpt_dir, "checkpoint.pt"),
            )
            with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch + 1, "val_metric": float(val_macro_rmse)}, f)

            session.report(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,  # Huber loss
                    "val_rmse_power": val_macro_rmse, # Macro-RMSE
                    "val_metric": val_macro_rmse, 
                    "epoch": epoch + 1,
                    "hyperband_info": {"budget": epoch},
                },
                checkpoint=Checkpoint.from_directory(ckpt_dir),
            )

# wrappers to keep LSTM/TCN trainables distinct in Ray Tune UI --
def tune_train_lstm(config, train_dataset_ref=None, val_dataset_ref=None, *,
                    num_workers: int, power_min: float, power_max: float, idx_power: int = 2):
    return tune_train(config, num_workers, power_min, power_max, idx_power,
                      train_dataset_ref, val_dataset_ref, model_builder=build_model_lstm)

def tune_train_tcn(config, train_dataset_ref=None, val_dataset_ref=None, *,
                   num_workers: int, power_min: float, power_max: float, idx_power: int = 2):
    return tune_train(config, num_workers, power_min, power_max, idx_power,
                      train_dataset_ref, val_dataset_ref, model_builder=build_model_tcn)



# ----------------------------------------------------- TRAINING UTILITIES -------------------------------------------- #

# Optimizer with decoupled weight decay, no decay on biases/1D params
def build_adamw(named_params, lr, weight_decay):
    decay, no_decay = [], []
    for _, p in named_params:
        if not p.requires_grad: continue
        (decay if p.ndim > 1 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )


def _resolve_dataset_ref(maybe_ref):
    if isinstance(maybe_ref, ObjectRef): return ray.get(maybe_ref)
    if hasattr(maybe_ref, "__len__") and hasattr(maybe_ref, "__getitem__"): return maybe_ref
    if isinstance(maybe_ref, type):
        raise ValueError("Got a class instead of a dataset instance/ObjectRef. Use ray.put(dataset).")
    raise ValueError(f"Invalid dataset ref type: {type(maybe_ref)}")


def inv_minmax_channel_torch(t_scaled: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor, ch: int):
    """ Inverse MinMax for a specific channel of shape (B,T,H,C) to (B,T,H,1)"""
    return t_scaled[..., ch:ch+1] * (max_val - min_val) + min_val


def _reconstruct_abs_from_residuals_batch(pred_resid: torch.Tensor, X_batch: torch.Tensor, idx_power: int):
    """Reconstructs absolute power predictions from residuals (scaled space).
    pred_resid: (B, T, H, 1) residuals; X_batch: (B, T, C_in).
    Returns: (B, T, H, 1) absolute predictions in scaled space.
    """
    base = X_batch[..., [idx_power]].unsqueeze(2)  # (B, T, 1) -> (B, T, 1, 1)
    return pred_resid + base


# Valid mask 1 <= i < len-horizon --> (B,T,H,1)
def _vectorized_mask(lengths: torch.Tensor, T_max: int, horizon: int, device):
    B = lengths.shape[0]
    t = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
    end = lengths.to(device).unsqueeze(1) - horizon
    mask_2d = (t >= 1) & (t < end)
    return mask_2d.unsqueeze(-1).expand(-1, -1, horizon).unsqueeze(-1)


def _build_loaders_for_trial_from_datasets(train_dataset_local, 
                                           val_dataset_local, 
                                           batch_size: int, 
                                           num_workers: int):
    # derive per-item sequence lengths from the dataset layout (ds.groups: (sid, x, y, T))
    train_sampler_local = LengthBucketSampler(train_dataset_local, batch_size=batch_size, shuffle=True)
    val_sampler_local = LengthBucketSampler(val_dataset_local, batch_size=batch_size, shuffle=False)

    train_loader_local = DataLoader(
        train_dataset_local,
        batch_sampler=train_sampler_local,
        collate_fn=session_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=(num_workers > 0),
    )

    val_loader_local = DataLoader(
        val_dataset_local,
        batch_sampler=val_sampler_local,
        collate_fn=session_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=(num_workers > 0),
    )
    
    return train_loader_local, val_loader_local


def tune_status_df(result_grid):
    rows = []
    for result in result_grid:
        row = OrderedDict()
        row["Trial"] = result.path.split("/")[-1]
        row.update(result.metrics)
        row.update(result.config)
        rows.append(row)
    if not rows: return pd.DataFrame()
    preferred_data = [
        "epoch","train_loss","val_loss","val_metric","val_rmse_power",
        "done","epoch","time_total_s",
        "batch_size","dropout","grad_clip_norm","hidden_dim","lr","num_epochs","num_layers",
        "kernel_size","weight_decay","alpha_h",
    ]
    df = pd.DataFrame(rows) 
    cols = [c for c in preferred_data if c in df.columns]
    return df[cols].sort_values(by="val_metric") if cols else df


# Restore a Ray ResultGrid from a run folder. if present
def restore_resultgrid(path: str):
    try:
        ea = ExperimentAnalysis(path)
        return tune.ResultGrid(experiment_analysis=ea)
    except Exception as e:
        print(f"[restore] Could not restore ResultGrid from '{path}': {e}")
        return None
    

def tune_run_status(run_root: str) -> dict:
    """Return {'exists', 'finished', 'num_trials', 'statuses'} for a Tune run dir.
    'finished' is True iff all trials are in {TERMINATED, ERROR}, i.e. no CREATED,
    PENDING, RUNNING, or other transient states. PAUSED counts as finished here to
    avoid BOHB resume glitches after short bracket runs."""
    cand = sorted(glob.glob(os.path.join(run_root, "experiment_state*.json")))
    if not cand:
        return {"exists": False, "finished": False, "num_trials": 0, "statuses": set()}
    path = cand[-1]
    with open(path, "r") as f:
        state = json.load(f)
    trials = state.get("trials", [])
    statuses = {t.get("status", "") for t in trials}
    # treat PAUSED as finished (HB/BOHB often leaves it after short runs)
    done = {"TERMINATED", "ERROR", "PAUSED"}
    finished = len(trials) > 0 and all(s in done for s in statuses)
    return {"exists": True, "finished": finished, "num_trials": len(trials), "statuses": statuses}


def save_final_model_pth(model, best_result, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "input_features": best_result.config["input_features"],
        "target_features": best_result.config["target_features"],
        "config": {k: (float(v) if isinstance(v, (int, float)) else v)
                   for k, v in best_result.config.items()},
    }
    torch.save(payload, out_path)