# src/mt4xai/model.py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import weight_norm
import torch.nn.functional as F


# --------------------------------- LSTM ----------------------------------- #

class MultiHorizonLSTM(nn.Module):
    """LSTM multi-horizon residual model with a linear (fully connected) head for 
    predicting residuals on variable-length sequences."""
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

    def forward(self, x: Tensor, seq_lengths: Tensor):
        packed_x = rnn_utils.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, out_lengths = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        out = self.linear(out).view(out.shape[0], out.shape[1], self.horizon, self.num_targets)
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
        horizon=int(cfg["horizon"]),
        num_targets=len(checkpoint["target_features"]),
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# --------------------------------- TCN ----------------------------------- #
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

    def forward(self, x: Tensor, seq_lengths: Tensor):
        # x: (B, T, F) -> (B, C, T)
        B, T, RF = x.shape
        h = self.input_proj(x.transpose(1, 2))
        skip_sum = None
        for blk in self.blocks:
            h, s = blk(h)
            skip_sum = s if skip_sum is None else (skip_sum + s)
        h = self.post(skip_sum)
        y = self.head(h).transpose(1, 2).view(B, T, self.horizon, self.num_targets)
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
        horizon=int(cfg["horizon"]),
        num_targets=len(cfg["target_features"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(cfg["device"])


# --------------------------------- COMMON MODELLING UTILITIES ----------------------------------- #
def horizon_weights(H: int, alpha: float | Tensor, device: torch.device, *,
                    normalise: bool = True, as_vector: bool = False, eps: float = 1e-12) -> Tensor:
    """Return exponentially decaying horizon weights w[h] = exp(-alpha*(h-1)).

    The weights emphasise shorter horizons as alpha increases. By default they are
    normalised to sum to 1 across horizons so that the aggregate loss/metric is
    scale-stable with respect to alpha.

    Args:
        H: number of prediction horizons.
        alpha: exponential decay parameter; larger values increase decay.
        device: torch device for the returned tensor.
        normalise: if True, divide by the sum across horizons (default: True).
        as_vector: if True, return shape (H,); otherwise return (1, 1, H, 1).
        eps: small constant to avoid division by zero during normalisation.

    Returns:
        Tensor: horizon weights as (H,) if as_vector is True, else (1, 1, H, 1).
    """
    h = torch.arange(1, H + 1, device=device, dtype=torch.float32)
    if isinstance(alpha, Tensor):
        alpha = alpha.to(device=device, dtype=torch.float32)
    w = torch.exp(-alpha * (h - 1))
    if normalise:
        w = w / w.sum().clamp_min(eps)
    return w if as_vector else w.view(1, 1, H, 1)
