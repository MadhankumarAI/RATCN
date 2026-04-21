"""
RA-TCN Demo: Regime-Adaptive TCN vs Vanilla TCN
================================================
Shows the dramatic performance gap when time series undergo distribution shift.

Usage:  python demo.py
Output: ra_tcn_demo.png 

cd ra-tcn
venv/Scripts/activate
streamlit run app.py

"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import generate_regime_data, TimeSeriesDataset
from models import VanillaTCN, RATCN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ──────────────── Config ────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE} THIIS SHOWS THE DEVICE")
LOOKBACK = 100
HORIZON = 10
HIDDEN = 64
N_LAYERS = 4
KERNEL = 7
EPOCHS_PRETRAIN = 30      # train on regime A only
EPOCHS_FULL = 30         # RA-TCN trains on all regimes
BATCH = 64
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────── Helpers ────────────────

def train_model(model, loader, epochs, desc="Training"):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if ep % 10 == 0 or ep == 1:
            avg = total_loss / len(loader.dataset)
            print(f"  {desc} epoch {ep:3d}/{epochs}  loss={avg:.6f}")


@torch.no_grad()
def predict_rolling(model, series, lookback, horizon):
    """Roll through entire series, predicting `horizon` steps at a time."""
    model.eval()
    preds = np.full(len(series), np.nan)
    for i in range(0, len(series) - lookback - horizon + 1, horizon):
        x = torch.tensor(series[i:i+lookback], dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,T)
        yhat = model(x).cpu().numpy().flatten()
        preds[i+lookback : i+lookback+horizon] = yhat
    return preds


def mse_per_regime(preds, truth, regimes):
    """Compute MSE for each regime (ignoring NaNs)."""
    results = {}
    for r in sorted(set(regimes)):
        mask = (regimes == r) & ~np.isnan(preds)
        if mask.sum() > 0:
            results[r] = float(np.mean((preds[mask] - truth[mask]) ** 2))
        else:
            results[r] = float("nan")
    return results


# ──────────────── Main ────────────────

def main():
    print("=" * 60)
    print("  RA-TCN DEMO: Regime-Adaptive vs Vanilla TCN")
    print("=" * 60)

    # 1. Generate data
    series, regimes = generate_regime_data(n_total=6000, seed=SEED)
    print(f"\nGenerated {len(series)} timesteps across 3 regimes")

    full_ds = TimeSeriesDataset(series, lookback=LOOKBACK, horizon=HORIZON)

    # Regime A indices: 0..1890 (lookback=100, so first valid target at 100)
    regime_a_end = 2000 - LOOKBACK - HORIZON
    regime_a_ds = Subset(full_ds, list(range(0, regime_a_end)))
    regime_a_loader = DataLoader(regime_a_ds, batch_size=BATCH, shuffle=True)

    full_loader = DataLoader(full_ds, batch_size=BATCH, shuffle=True)

    # 2. Train Vanilla TCN on Regime A ONLY (simulates real deployment)
    print("\n--- Vanilla TCN: training on Regime A only ---")
    vanilla = VanillaTCN(hidden=HIDDEN, n_layers=N_LAYERS, kernel_size=KERNEL,
                         horizon=HORIZON).to(DEVICE)
    train_model(vanilla, regime_a_loader, EPOCHS_PRETRAIN, desc="VanillaTCN")

    # 3. Train RA-TCN on ALL regimes (learns to adapt)
    print("\n--- RA-TCN: training on ALL regimes ---")
    ratcn = RATCN(hidden=HIDDEN, n_layers=N_LAYERS, kernel_size=KERNEL,
                  horizon=HORIZON).to(DEVICE)
    train_model(ratcn, full_loader, EPOCHS_FULL, desc="RA-TCN")

    # 4. Also train Vanilla TCN on all regimes (fair comparison baseline)
    print("\n--- Vanilla TCN (full): training on ALL regimes ---")
    vanilla_full = VanillaTCN(hidden=HIDDEN, n_layers=N_LAYERS, kernel_size=KERNEL,
                              horizon=HORIZON).to(DEVICE)
    train_model(vanilla_full, full_loader, EPOCHS_FULL, desc="VanillaTCN-Full")

    # 5. Roll predictions
    print("\nGenerating rolling predictions...")
    preds_vanilla = predict_rolling(vanilla, series, LOOKBACK, HORIZON)
    preds_ratcn = predict_rolling(ratcn, series, LOOKBACK, HORIZON)
    preds_vanilla_full = predict_rolling(vanilla_full, series, LOOKBACK, HORIZON)

    # 6. Compute per-regime MSE
    regime_names = {0: "Regime A\n(Calm)", 1: "Regime B\n(Shock)", 2: "Regime C\n(Recovery)"}
    mse_v = mse_per_regime(preds_vanilla, series, regimes)
    mse_r = mse_per_regime(preds_ratcn, series, regimes)
    mse_vf = mse_per_regime(preds_vanilla_full, series, regimes)

    print("\n" + "=" * 60)
    print(f"  {'Regime':<20} {'VanillaTCN':>12} {'Vanilla(Full)':>14} {'RA-TCN':>12}")
    print("-" * 60)
    for r in [0, 1, 2]:
        print(f"  {regime_names[r].replace(chr(10),' '):<20} {mse_v[r]:>12.4f} {mse_vf[r]:>14.4f} {mse_r[r]:>12.4f}")
    print("=" * 60)

    # ──────────────── KILLER VISUALIZATION ────────────────
    print("\nRendering visualization...")
    create_visualization(series, regimes, preds_vanilla, preds_ratcn,
                         preds_vanilla_full, mse_v, mse_r, mse_vf, regime_names)
    print("Saved: ra_tcn_demo.png")


def create_visualization(series, regimes, preds_vanilla, preds_ratcn,
                         preds_vanilla_full, mse_v, mse_r, mse_vf, regime_names):
    """Build the multi-panel killer demo figure."""

    # Color palette
    C_GROUND = "#2C3E50"
    C_VANILLA = "#E74C3C"
    C_RATCN = "#2ECC71"
    C_VFULL = "#F39C12"
    C_REGIME = ["#3498DB33", "#E74C3C22", "#2ECC7122"]
    C_REGIME_BORDER = ["#3498DB", "#E74C3C", "#2ECC71"]

    fig = plt.figure(figsize=(22, 14), facecolor="#0D1117")
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.8, 2.5, 1.2],
                           hspace=0.35, wspace=0.3,
                           left=0.06, right=0.97, top=0.92, bottom=0.06)

    # ── Title panel ──
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#0D1117")
    ax_title.axis("off")
    ax_title.text(0.5, 0.7, "Regime-Adaptive TCN  vs  Vanilla TCN",
                  fontsize=28, fontweight="bold", color="white",
                  ha="center", va="center", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.2,
                  "Standard TCN collapses under distribution shift  ·  "
                  "RA-TCN adapts its convolution kernels via FiLM modulation",
                  fontsize=13, color="#8899AA", ha="center", va="center",
                  transform=ax_title.transAxes)

    # ── Main time series panel ──
    ax_ts = fig.add_subplot(gs[1, :])
    ax_ts.set_facecolor("#0D1117")
    t = np.arange(len(series))

    # shade regimes
    boundaries = [0, 2000, 4000, 6000]
    labels_short = ["Regime A: Calm", "Regime B: Shock", "Regime C: Recovery"]
    for i in range(3):
        ax_ts.axvspan(boundaries[i], boundaries[i+1], color=C_REGIME[i], zorder=0)
        ax_ts.axvline(boundaries[i+1] if i < 2 else 0, color=C_REGIME_BORDER[i],
                      ls="--", lw=1.2, alpha=0.5)
        cx = (boundaries[i] + boundaries[i+1]) / 2
        ax_ts.text(cx, ax_ts.get_ylim()[0] if i > 0 else 0, labels_short[i],
                   fontsize=11, color=C_REGIME_BORDER[i], ha="center", va="bottom",
                   fontweight="bold", alpha=0.8)

    # ground truth
    ax_ts.plot(t, series, color=C_GROUND, lw=1.0, alpha=0.5, label="Ground Truth", zorder=1)

    # predictions
    ax_ts.plot(t, preds_vanilla, color=C_VANILLA, lw=1.4, alpha=0.85,
               label="Vanilla TCN (trained on A only)", zorder=2)
    ax_ts.plot(t, preds_ratcn, color=C_RATCN, lw=1.8, alpha=0.95,
               label="RA-TCN (ours)", zorder=3)

    ax_ts.set_xlim(0, len(series))
    ax_ts.set_xlabel("Timestep", fontsize=12, color="#AABBCC")
    ax_ts.set_ylabel("Value", fontsize=12, color="#AABBCC")
    ax_ts.tick_params(colors="#667788")
    for spine in ax_ts.spines.values():
        spine.set_color("#334455")

    legend = ax_ts.legend(loc="upper left", fontsize=11, framealpha=0.3,
                          edgecolor="#334455", facecolor="#0D1117")
    for text in legend.get_texts():
        text.set_color("white")

    # regime boundary annotations
    for bx in [2000, 4000]:
        ax_ts.annotate("REGIME\nSHIFT", xy=(bx, series[bx]), fontsize=8,
                        color="#FF6B6B", fontweight="bold", ha="center",
                        va="bottom",
                        arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=1.5))

    # ── Bottom panels: bar charts + error heatmap ──

    # Panel 1: MSE bar chart
    ax_bar = fig.add_subplot(gs[2, 0])
    ax_bar.set_facecolor("#0D1117")

    x_pos = np.arange(3)
    w = 0.25
    bars_v = [mse_v[r] for r in [0, 1, 2]]
    bars_vf = [mse_vf[r] for r in [0, 1, 2]]
    bars_r = [mse_r[r] for r in [0, 1, 2]]

    ax_bar.bar(x_pos - w, bars_v, w, color=C_VANILLA, label="Vanilla (A only)", alpha=0.85)
    ax_bar.bar(x_pos, bars_vf, w, color=C_VFULL, label="Vanilla (Full)", alpha=0.85)
    ax_bar.bar(x_pos + w, bars_r, w, color=C_RATCN, label="RA-TCN", alpha=0.85)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(["Calm", "Shock", "Recovery"], fontsize=10, color="#AABBCC")
    ax_bar.set_ylabel("MSE", fontsize=11, color="#AABBCC")
    ax_bar.set_title("Per-Regime MSE", fontsize=13, color="white", fontweight="bold")
    ax_bar.tick_params(colors="#667788")
    for spine in ax_bar.spines.values():
        spine.set_color("#334455")
    leg2 = ax_bar.legend(fontsize=8, framealpha=0.3, edgecolor="#334455", facecolor="#0D1117")
    for text in leg2.get_texts():
        text.set_color("white")

    # Panel 2: rolling error comparison
    ax_err = fig.add_subplot(gs[2, 1])
    ax_err.set_facecolor("#0D1117")

    win = 50
    err_vanilla = np.abs(preds_vanilla - series)
    err_ratcn = np.abs(preds_ratcn - series)

    # rolling mean (ignoring nans)
    def rolling_nanmean(arr, w):
        out = np.full_like(arr, np.nan)
        for i in range(w, len(arr)):
            chunk = arr[i-w:i]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 0:
                out[i] = np.mean(valid)
        return out

    roll_v = rolling_nanmean(err_vanilla, win)
    roll_r = rolling_nanmean(err_ratcn, win)

    ax_err.fill_between(t, 0, roll_v, color=C_VANILLA, alpha=0.3, label="Vanilla Error")
    ax_err.fill_between(t, 0, roll_r, color=C_RATCN, alpha=0.3, label="RA-TCN Error")
    ax_err.plot(t, roll_v, color=C_VANILLA, lw=1.5, alpha=0.8)
    ax_err.plot(t, roll_r, color=C_RATCN, lw=1.5, alpha=0.8)

    for bx in [2000, 4000]:
        ax_err.axvline(bx, color="#FF6B6B", ls="--", lw=1, alpha=0.5)

    ax_err.set_xlim(0, len(series))
    ax_err.set_xlabel("Timestep", fontsize=10, color="#AABBCC")
    ax_err.set_ylabel("Rolling MAE", fontsize=11, color="#AABBCC")
    ax_err.set_title("Error Over Time (rolling window=50)", fontsize=13,
                     color="white", fontweight="bold")
    ax_err.tick_params(colors="#667788")
    for spine in ax_err.spines.values():
        spine.set_color("#334455")
    leg3 = ax_err.legend(fontsize=9, framealpha=0.3, edgecolor="#334455", facecolor="#0D1117")
    for text in leg3.get_texts():
        text.set_color("white")

    # Panel 3: improvement % summary
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.set_facecolor("#0D1117")
    ax_summary.axis("off")

    # compute improvement percentages
    improvements = {}
    for r in [0, 1, 2]:
        if mse_v[r] > 0:
            improvements[r] = (mse_v[r] - mse_r[r]) / mse_v[r] * 100
        else:
            improvements[r] = 0

    regime_labels = ["Calm", "Shock", "Recovery"]
    regime_emoji_colors = [C_REGIME_BORDER[0], C_REGIME_BORDER[1], C_REGIME_BORDER[2]]

    ax_summary.text(0.5, 0.95, "RA-TCN Improvement", fontsize=15, fontweight="bold",
                    color="white", ha="center", va="top", transform=ax_summary.transAxes)
    ax_summary.text(0.5, 0.82, "over Vanilla TCN (trained on Regime A only)",
                    fontsize=9, color="#667788", ha="center", va="top",
                    transform=ax_summary.transAxes)

    for i, r in enumerate([0, 1, 2]):
        y = 0.62 - i * 0.22
        imp = improvements[r]
        color = C_RATCN if imp > 0 else C_VANILLA
        sign = "+" if imp > 0 else ""

        ax_summary.text(0.15, y, regime_labels[i], fontsize=13, color=regime_emoji_colors[i],
                        ha="left", va="center", fontweight="bold",
                        transform=ax_summary.transAxes)
        ax_summary.text(0.85, y, f"{sign}{imp:.1f}%", fontsize=20, color=color,
                        ha="right", va="center", fontweight="bold",
                        transform=ax_summary.transAxes)

        # MSE values below
        ax_summary.text(0.85, y - 0.07,
                        f"Vanilla: {mse_v[r]:.4f}  |  RA-TCN: {mse_r[r]:.4f}",
                        fontsize=8, color="#667788", ha="right", va="center",
                        transform=ax_summary.transAxes)

    # Watermark
    fig.text(0.5, 0.01, "RA-TCN: Regime-Adaptive Temporal Convolutional Network",
             fontsize=9, color="#334455", ha="center", style="italic")

    out_path = os.path.join(os.path.dirname(__file__), "ra_tcn_demo.png")
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
