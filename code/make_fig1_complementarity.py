#!/usr/bin/env python
"""Figure 1 (Path A): the complementarity finding — the best outlier mechanism
flips with contamination. Shallow variants, translation error (mean, deg),
log-y, datasets ordered by RESfM-paper Outliers%. Colorblind-safe (Okabe-Ito).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# datasets ordered by contamination (RESfM-paper Outliers%)
labels = ["Strecha\n2.7%", "BlendedMVS\n3.6%", "MegaDepth\n25.6%\n(in-dist.)", "1DSfM\n31.9%"]
x = list(range(4))

# shallow translation error, mean (deg) — frozen from Table 2
weight     = [1.945, 0.114, 0.460, 15.884]   # soft reweight (U-ESFM)
madweight  = [2.660, 0.121, 0.511,  5.502]   # label-free MAD-remove + head-weight (U-ESFM)
resfm_rel  = [0.197, 0.330, 0.209, 11.124]   # supervised baseline

# Okabe-Ito (CVD-safe): blue, vermillion; baseline neutral gray dashed
C_W, C_M, C_R = "#0072B2", "#D55E00", "#555555"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_yscale("log")

def plot(series, color, label, ls="-", marker="o"):
    ax.plot(x, series, ls=ls, color=color, lw=2.0, marker=marker, ms=8,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
            label=label, zorder=3, clip_on=False)

plot(resfm_rel, C_R, "RESfM (released, supervised)", ls="--", marker="s")
plot(weight,    C_W, "U-ESFM · weight (soft)")
plot(madweight, C_M, "U-ESFM · madweight (label-free removal)")

# direct end-labels at 1DSfM
for series, color, txt, dy in [(madweight, C_M, "madweight", 0),
                               (resfm_rel, C_R, "RESfM", 0),
                               (weight, C_W, "weight", 0)]:
    ax.annotate(txt, xy=(3, series[3]), xytext=(8, 0), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center")

# mark the winner per dataset with a subtle underline of the lowest point
for i in range(4):
    vals = {C_W: weight[i], C_M: madweight[i], C_R: resfm_rel[i]}
    best_c = min(vals, key=vals.get)
    ax.scatter([i], [vals[best_c]], s=200, facecolors="none",
               edgecolors=best_c, linewidths=1.8, zorder=4, clip_on=False)

# regime annotations (shaded bands; labels pinned to the bottom to avoid collisions)
ax.axvspan(-0.35, 1.5, color="#f2f7fb", zorder=0)
ax.axvspan(2.5, 3.35, color="#fdf3ee", zorder=0)
ax.text(0.5, 0.108, "low contamination", ha="center", va="bottom",
        fontsize=9, color=MUTED, style="italic")
ax.text(2.95, 0.108, "high contamination", ha="center", va="bottom",
        fontsize=9, color=MUTED, style="italic")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.set_xlim(-0.35, 3.35)
ax.set_ylabel("Translation error (deg, log scale)", fontsize=10, color=INK)
ax.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "0.3", "1", "3", "10"]))
ax.tick_params(colors=MUTED)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

ax.set_title("The optimal outlier mechanism flips with contamination",
             fontsize=12, color=INK, fontweight="bold", pad=12, loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=1,
          bbox_to_anchor=(0.02, 0.99))

# caption note
fig.text(0.5, -0.02,
         "Circled point = best method per dataset. weight wins low-outlier BlendedMVS; "
         "madweight wins high-outlier 1DSfM;\nRESfM wins clean Strecha and in-distribution "
         "MegaDepth. Shallow (1x3), mean translation error, seed 20.",
         ha="center", fontsize=7.5, color=MUTED)

plt.tight_layout()
out = "../claude specs/PATHA_fig1_complementarity"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf", "and .png")
