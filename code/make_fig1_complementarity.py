#!/usr/bin/env python
"""Figure 1 (Path A): the complementarity finding — the best outlier mechanism
flips with contamination. Shallow variants, RESfM-aligned (finelr) schedule,
translation error (mean, deg), log-y, datasets ordered by measured Outliers%.
Colorblind-safe (Okabe-Ito).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# datasets ordered by contamination (measured on our tracks)
labels = ["Strecha\n1.7%", "BlendedMVS\n3.1%", "MegaDepth\n25.4%\n(in-dist.)",
          "1DSfM\n43.3%", "1DSfM-hard\n~60%"]
x = list(range(5))

# shallow translation error, mean (deg) — RESfM-aligned (finelr) schedule, seed 20
weight     = [2.003, 0.100, 0.423, 12.787, 21.941]   # soft reweight (U-ESFM)
wttt       = [1.977, 0.136, 0.390, 16.128, 26.435]   # soft reweight + TTT (U-ESFM; MD = mean of 3 eval repeats)
madweight  = [2.498, 0.146, 0.422, 12.392, 12.211]   # label-free MAD-remove + head-weight (U-ESFM)
resfm      = [0.006, 0.379, 0.344, 11.032, 16.218]   # supervised from-scratch baseline (twin config)

# Okabe-Ito (CVD-safe): blue, sky, vermillion; baseline neutral gray dashed
C_W, C_T, C_M, C_R = "#0072B2", "#56B4E9", "#D55E00", "#555555"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_yscale("log")

def plot(series, color, label, ls="-", marker="o"):
    ax.plot(x, series, ls=ls, color=color, lw=2.0, marker=marker, ms=8,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
            label=label, zorder=3, clip_on=False)

plot(resfm,     C_R, "RESfM-from-scratch (supervised)", ls="--", marker="s")
plot(weight,    C_W, "U-ESFM · weight (soft)")
plot(wttt,      C_T, "U-ESFM · weight+TTT")
plot(madweight, C_M, "U-ESFM · madweight (label-free removal)")

# direct end-labels at 1DSfM-hard
for series, color, txt in [(madweight, C_M, "madweight"), (resfm, C_R, "RESfM"),
                           (weight, C_W, "weight"), (wttt, C_T, "weight+TTT")]:
    ax.annotate(txt, xy=(4, series[4]), xytext=(8, 0), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center")

# mark the winner per dataset with a subtle ring on the lowest point
for i in range(5):
    vals = {C_W: weight[i], C_T: wttt[i], C_M: madweight[i], C_R: resfm[i]}
    best_c = min(vals, key=vals.get)
    ax.scatter([i], [vals[best_c]], s=200, facecolors="none",
               edgecolors=best_c, linewidths=1.8, zorder=4, clip_on=False)

# regime annotations (shaded bands; labels pinned to the bottom to avoid collisions)
ax.axvspan(-0.35, 1.5, color="#f2f7fb", zorder=0)
ax.axvspan(2.5, 4.35, color="#fdf3ee", zorder=0)
ax.text(0.5, 0.0045, "low contamination", ha="center", va="bottom",
        fontsize=9, color=MUTED, style="italic")
ax.text(3.45, 0.0045, "high contamination", ha="center", va="bottom",
        fontsize=9, color=MUTED, style="italic")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.set_xlim(-0.35, 4.35)
ax.set_ylabel("Translation error (deg, log scale)", fontsize=10, color=INK)
ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 0.3, 1, 3, 10, 30]))
ax.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "0.3", "1", "3", "10", "30"]))
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
         "label-free madweight wins extreme 1DSfM-hard;\nsupervised RESfM retains clean "
         "Strecha, in-distribution MegaDepth, and moderate 1DSfM. "
         "Shallow (1x3), RESfM-aligned schedule, seed 20.",
         ha="center", fontsize=7.5, color=MUTED)

plt.tight_layout()
out = "../claude specs/PATHA_fig1_complementarity"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf", "and .png")
