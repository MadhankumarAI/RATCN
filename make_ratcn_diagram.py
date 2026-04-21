"""Render an RA-TCN architecture diagram styled to match image.png."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 2.3]}
)
for ax in (ax_top, ax_bot):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_aspect("equal")

# palette matching image.png
C_CONV = "#CFE2F3"
C_BN = "#FFF2CC"
C_FILM = "#F9CB9C"
C_RELU = "#FFF2CC"
C_SIDE = "#D9EAD3"
C_MEM = "#D5A6BD"
C_EDGE = "#6FA8DC"
C_OUT = "#93C47D"

def block(ax, x, y, w, h, text, fc, ec="#444", fs=10, fw="normal"):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        fc=fc, ec=ec, lw=1.2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw)

def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.4, style="-|>"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        color=color, lw=lw,
    )
    ax.add_patch(a)

# ── Top panel: overall architecture ──
ax_top.text(4.0, 3.55, "Regime-Adaptive TCN (RA-TCN)",
            ha="center", fontsize=13, fontweight="bold")

# dotted container
ax_top.add_patch(FancyBboxPatch(
    (0.3, 1.7), 7.4, 1.6, boxstyle="round,pad=0.05,rounding_size=0.1",
    fc="none", ec=C_EDGE, lw=1.1, ls=(0, (2, 2)),
))

block(ax_top, 0.6, 2.15, 1.8, 0.9, "Input", C_SIDE)
block(ax_top, 2.8, 2.15, 2.0, 0.9, "Adaptive\nTCN Block", C_CONV, fs=9)
block(ax_top, 5.1, 2.15, 2.0, 0.9, "Adaptive\nTCN Block", C_CONV, fs=9)
block(ax_top, 7.4, 2.15, 1.5, 0.9, "Head", C_OUT, fs=10)

arrow(ax_top, 2.4, 2.6, 2.8, 2.6)
arrow(ax_top, 4.8, 2.6, 5.1, 2.6)
arrow(ax_top, 7.1, 2.6, 7.4, 2.6)
arrow(ax_top, 8.9, 2.6, 9.6, 2.6)
ax_top.text(9.8, 2.6, "y", fontsize=11, va="center")

# side channel: shift detector + memory bank
block(ax_top, 9.8, 1.35, 2.0, 0.8, "Shift\nDetector", C_FILM, fs=9)
block(ax_top, 12.1, 1.35, 1.7, 0.8, "Memory\nBank", C_MEM, fs=9)
arrow(ax_top, 9.7, 1.75, 9.8, 1.75)  # from input trunk
arrow(ax_top, 11.8, 1.75, 12.1, 1.75)
ax_top.text(10.8, 0.95, "stats → embed", fontsize=8, color="#666", ha="center")
ax_top.text(12.95, 0.95, "soft attention\nover prototypes",
            fontsize=8, color="#666", ha="center")

# shift_embed fans into blocks
ax_top.plot([12.95, 12.95], [1.35, 0.55], color="#A64D79", lw=1.3)
ax_top.plot([3.8, 12.95], [0.55, 0.55], color="#A64D79", lw=1.3)
for bx in (3.8, 6.1):
    arrow(ax_top, bx, 0.55, bx, 2.12, color="#A64D79", lw=1.3)
ax_top.text(8.4, 0.33, "shift_embed (γ, β source)",
            fontsize=8.5, color="#A64D79", style="italic", ha="center")

# tap from input to shift detector
ax_top.plot([1.5, 1.5], [2.15, 1.75], color="#666", lw=1, ls="--")
ax_top.plot([1.5, 9.8], [1.75, 1.75], color="#666", lw=1, ls="--")

# ── Bottom panel: the block internals ──
ax_bot.text(7.0, 3.75, "Adaptive TCN Block  (FiLM sits between BatchNorm and ReLU)",
            ha="center", fontsize=12, fontweight="bold")

# dotted container
ax_bot.add_patch(FancyBboxPatch(
    (0.15, 0.3), 13.7, 3.0, boxstyle="round,pad=0.05,rounding_size=0.1",
    fc="none", ec="#93C47D", lw=1.1, ls=(0, (3, 3)),
))

# input label
ax_bot.text(0.45, 1.95, "x", fontsize=12, va="center", fontweight="bold")

y0 = 1.6
h0 = 0.8
boxes = [
    (0.7, "Dilated\nCausal Conv", C_CONV),
    (2.4, "Batch\nNorm",          C_BN),
    (4.1, "FiLM\nγ, β",            C_FILM),
    (5.8, "ReLU",                   C_RELU),
    (7.3, "Dilated\nCausal Conv", C_CONV),
    (9.0, "Batch\nNorm",          C_BN),
    (10.7, "FiLM\nγ, β",           C_FILM),
    (12.4, "ReLU",                  C_RELU),
]
widths = [1.5, 1.5, 1.5, 1.2, 1.5, 1.5, 1.5, 1.2]
for (x, t, fc), w in zip(boxes, widths):
    block(ax_bot, x, y0, w, h0, t, fc, fs=9)

# arrows between boxes
edges_x = []
for (x, _, _), w in zip(boxes, widths):
    edges_x.append((x, x + w))
prev_end = 0.55
for (start, end) in edges_x:
    arrow(ax_bot, prev_end, y0 + h0 / 2, start, y0 + h0 / 2)
    prev_end = end

# exit arrow into sum
arrow(ax_bot, prev_end, y0 + h0 / 2, prev_end + 0.35, y0 + h0 / 2)
# sum node
sx, sy = prev_end + 0.55, y0 + h0 / 2
ax_bot.add_patch(plt.Circle((sx, sy), 0.18, fc="white", ec="#444", lw=1.2))
ax_bot.text(sx, sy, "+", ha="center", va="center", fontsize=13, fontweight="bold")
arrow(ax_bot, sx + 0.18, sy, sx + 0.75, sy)
ax_bot.text(sx + 0.95, sy, "out", fontsize=11, va="center")

# residual skip
ax_bot.plot([0.55, 0.55], [y0 + h0 / 2, 0.85], color="#444", lw=1.3)
ax_bot.plot([0.55, sx], [0.85, 0.85], color="#444", lw=1.3)
arrow(ax_bot, sx, 0.85, sx, sy - 0.18)
ax_bot.text(6.5, 0.62, "residual (skip)", fontsize=9, color="#666", style="italic",
            ha="center")

# shift_embed feeds into FiLM blocks from above
for fx in (4.85, 11.45):
    ax_bot.plot([fx, fx], [3.15, y0 + h0], color="#A64D79", lw=1.3)
    arrow(ax_bot, fx, y0 + h0 + 0.02, fx, y0 + h0 - 0.02, color="#A64D79", lw=1.3)
ax_bot.plot([4.85, 11.45], [3.15, 3.15], color="#A64D79", lw=1.3)
ax_bot.text(8.15, 3.28, "shift_embed",
            fontsize=9.5, color="#A64D79", style="italic", ha="center")

# annotation under FiLM
ax_bot.text(4.85, 1.25, "feature · (1 + γ) + β",
            fontsize=8.5, color="#B45F06", ha="center", style="italic")
ax_bot.text(11.45, 1.25, "feature · (1 + γ) + β",
            fontsize=8.5, color="#B45F06", ha="center", style="italic")

plt.tight_layout()
out = "image_ratcn.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
