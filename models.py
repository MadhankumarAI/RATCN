"""
Two models:
  1. VanillaTCN  — standard dilated causal TCN (baseline)
  2. RATCN        — Regime-Adaptive TCN with shift detection + FiLM kernel modulation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  Building blocks
# ──────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Dilated causal 1-D convolution with left-padding."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))  # left-pad
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual block: 2x (CausalConv -> BN -> ReLU -> Dropout) + skip."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.drop(F.relu(self.bn2(self.conv2(out))))
        return out + res


# ──────────────────────────────────────────────
#  1.  Vanilla TCN
# ──────────────────────────────────────────────

class VanillaTCN(nn.Module):
    def __init__(self, in_ch=1, hidden=64, n_layers=4, kernel_size=7,
                 dropout=0.1, horizon=10):
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        """x: (B, 1, T) -> (B, horizon)"""
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        # take last timestep's features
        out = self.head(h[:, :, -1])
        return out


# ──────────────────────────────────────────────
#  2.  Regime-Adaptive TCN  (RA-TCN)
# ──────────────────────────────────────────────

class ShiftDetector(nn.Module):
    """
    Lightweight shift detector: compares running statistics of the input
    window against a learned reference distribution. Outputs a compact
    "shift embedding" that describes *how* the current regime differs.
    """

    def __init__(self, stat_dim=4, embed_dim=32):
        super().__init__()
        # stat_dim: [mean, std, skew, kurtosis]  of the input window
        self.net = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    @staticmethod
    def compute_stats(x):
        """x: (B, 1, T) -> (B, 4)"""
        vals = x.squeeze(1)  # (B, T)
        mean = vals.mean(dim=1, keepdim=True)
        std = vals.std(dim=1, keepdim=True) + 1e-6
        centered = (vals - mean) / std
        skew = centered.pow(3).mean(dim=1, keepdim=True)
        kurt = centered.pow(4).mean(dim=1, keepdim=True) - 3.0
        return torch.cat([mean, std, skew, kurt], dim=1)  # (B, 4)

    def forward(self, x):
        stats = self.compute_stats(x)
        return self.net(stats)  # (B, embed_dim)


class FiLMModulator(nn.Module):
    """
    Feature-wise Linear Modulation: generates per-channel scale & bias
    from the shift embedding to re-calibrate frozen-style conv features.
    """

    def __init__(self, embed_dim, channels):
        super().__init__()
        self.scale_net = nn.Linear(embed_dim, channels)
        self.bias_net = nn.Linear(embed_dim, channels)

    def forward(self, features, shift_embed):
        """
        features:    (B, C, T)
        shift_embed: (B, embed_dim)
        returns:     (B, C, T)  modulated features
        """
        gamma = self.scale_net(shift_embed).unsqueeze(2)  # (B, C, 1)
        beta = self.bias_net(shift_embed).unsqueeze(2)     # (B, C, 1)
        return features * (1.0 + gamma) + beta


class AdaptiveTCNBlock(nn.Module):
    """TCN block + FiLM modulation after each sub-block."""

    def __init__(self, channels, kernel_size, dilation, embed_dim, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.film1 = FiLMModulator(embed_dim, channels)

        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.film2 = FiLMModulator(embed_dim, channels)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, shift_embed):
        res = x
        out = self.drop(F.relu(self.film1(self.bn1(self.conv1(x)), shift_embed)))
        out = self.drop(F.relu(self.film2(self.bn2(self.conv2(out)), shift_embed)))
        return out + res


class RegimeMemoryBank(nn.Module):
    """
    Stores K prototype shift embeddings (one per learned regime).
    At inference, finds the closest prototype and blends it with the
    current shift embedding for smoother adaptation.
    """

    def __init__(self, n_regimes=8, embed_dim=32, momentum=0.9):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_regimes, embed_dim) * 0.1)
        self.momentum = momentum

    def forward(self, shift_embed):
        """shift_embed: (B, D) -> blended (B, D)"""
        # cosine similarity to prototypes
        sim = F.cosine_similarity(
            shift_embed.unsqueeze(1),          # (B, 1, D)
            self.prototypes.unsqueeze(0),       # (1, K, D)
            dim=2,
        )  # (B, K)
        weights = F.softmax(sim * 10.0, dim=1)  # sharp selection
        retrieved = torch.einsum("bk,kd->bd", weights, self.prototypes)
        blended = self.momentum * shift_embed + (1 - self.momentum) * retrieved
        return blended


class RATCN(nn.Module):
    """
    Regime-Adaptive TCN.

    Architecture:
      1. ShiftDetector   — extracts statistical fingerprint of input window
      2. RegimeMemoryBank — retrieves closest known regime prototype
      3. AdaptiveTCNBlocks — standard dilated causal convs + FiLM modulation
         conditioned on the shift embedding
      4. Prediction head
    """

    def __init__(self, in_ch=1, hidden=64, n_layers=4, kernel_size=7,
                 dropout=0.1, horizon=10, embed_dim=32, n_regimes=8):
        super().__init__()
        self.shift_detector = ShiftDetector(stat_dim=4, embed_dim=embed_dim)
        self.memory_bank = RegimeMemoryBank(n_regimes=n_regimes, embed_dim=embed_dim)

        self.input_proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([
            AdaptiveTCNBlock(hidden, kernel_size, dilation=2**i,
                             embed_dim=embed_dim, dropout=dropout)
            for i in range(n_layers)
        ])
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        """x: (B, 1, T) -> (B, horizon)"""
        # 1. detect distribution shift
        shift_embed = self.shift_detector(x)
        shift_embed = self.memory_bank(shift_embed)

        # 2. TCN backbone with FiLM modulation
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, shift_embed)

        out = self.head(h[:, :, -1])
        return out
