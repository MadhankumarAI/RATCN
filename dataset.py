"""
Synthetic dataset with clear regime shifts for TCN demo.
Three regimes: Normal -> Shock -> Recovery (with different statistical properties)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_regime_data(n_total=6000, seed=42):
    """
    Generate a synthetic time series with 3 distinct regimes:
      Regime A (0 - 2000):    Smooth sinusoidal + low noise, mean ~0
      Regime B (2000 - 4000): Abrupt shift — higher frequency, higher variance, mean jump
      Regime C (4000 - 6000): Return to sinusoidal but with different phase/amplitude
    """
    np.random.seed(seed)
    t = np.arange(n_total, dtype=np.float32)

    # --- Regime A: calm, low-freq sine ---
    seg_a = np.sin(2 * np.pi * t[:2000] / 200) * 1.0 + np.random.randn(2000) * 0.1

    # --- Regime B: shock — mean shift + high freq + high noise ---
    seg_b = (
        3.0  # mean jump
        + np.sin(2 * np.pi * t[2000:4000] / 50) * 2.5  # higher amplitude & freq
        + np.random.randn(2000) * 0.6
    )

    # --- Regime C: recovery — similar shape to A but different phase/amplitude ---
    seg_c = (
        -1.0
        + np.sin(2 * np.pi * t[4000:6000] / 180 + np.pi / 3) * 1.5
        + np.random.randn(2000) * 0.15
    )

    series = np.concatenate([seg_a, seg_b, seg_c]).astype(np.float32)

    # Regime labels for visualization
    regimes = np.array([0] * 2000 + [1] * 2000 + [2] * 2000)

    return series, regimes


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for time series forecasting."""

    def __init__(self, series, lookback=100, horizon=10):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.length = len(series) - lookback - horizon + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.lookback].unsqueeze(0)  # (1, lookback)
        y = self.series[idx + self.lookback : idx + self.lookback + self.horizon]  # (horizon,)
        return x, y
