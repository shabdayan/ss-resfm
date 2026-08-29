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
labels = ["Strecha\n1.7%", "BMVS\n3.1%", "MegaDepth\n25.4%\n(in-dist.)",
          "1DSfM\n43.3%", "1DSfM-hard\n~60%"]
x = list(range(5))

# shallow translation error, mean (deg) — RESfM-aligned (finelr) schedule, seed 20
weight     = [2.003, 0.100, 0.423, 12.787, 21.941]   # soft reweight (SS-RESfM)
wttt       = [1.977, 0.136, 0.390, 16.128, 26.435]   # soft reweight + TTT (SS-RESfM; MD = mean of 3 eval repeats)
madweight  = [2.498, 0.146, 0.422, 12.392, 12.211]   # label-free MAD-remove + head-weight (SS-RESfM)
resfm      = [0.006, 0.379, 0.344, 11.032, 16.218]   # supervised from-scratch baseline (twin config)
# rotation means (deg), same order [Strecha, BMVS, MD, 1DSfM, 1DSfM-hard]
weight_r   = [7.48, 6.65, 3.04, 8.39, 25.6]
wttt_r     = [7.41, 7.91, 2.51, 11.19, 32.0]
madweight_r= [14.62, 9.60, 3.49, 8.31, 14.4]
resfm_r    = [0.02, 36.10, 2.01, 9.79, 22.7]

# Okabe-Ito (CVD-safe): blue, sky, vermillion; baseline neutral gray dashed
C_W, C_T, C_M, C_R = "#0072B2", "#56B4E9", "#D55E00", "#555555"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.3),
                              gridspec_kw={"wspace": 0.3})
ax.set_yscale("log"); ax2.set_yscale("log")

def plot(series, color, label, ls="-", marker="o"):
    ax.plot(x, series, ls=ls, color=color, lw=2.0, marker=marker, ms=9,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
            label=label, zorder=3, clip_on=False)

plot(resfm,     C_R, "RESfM (supervised)", ls="--", marker="s")
plot(weight,    C_W, "weight")
plot(wttt,      C_T, "weight+TTT")
plot(madweight, C_M, "madweight")

# direct end-labels at 1DSfM-hard
for series, color, txt, dy in [(madweight, C_M, "madweight", -3), (resfm, C_R, "RESfM", -1),
                               (weight, C_W, "weight", 2), (wttt, C_T, "weight+TTT", 7)]:
    ax.annotate(txt, xy=(4, series[4]), xytext=(8, dy), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center")
for series, color, txt, dy in [(madweight_r, C_M, "madweight", -4), (resfm_r, C_R, "RESfM", -3),
                               (weight_r, C_W, "weight", 3), (wttt_r, C_T, "weight+TTT", 7)]:
    ax2.annotate(txt, xy=(4, series[4]), xytext=(8, dy), textcoords="offset points",
                 color=color, fontsize=9, fontweight="bold", va="center")

# mark the winner per dataset with a subtle ring on the lowest point
for i in range(5):
    vals = {C_W: weight[i], C_T: wttt[i], C_M: madweight[i], C_R: resfm[i]}
    best_c = min(vals, key=vals.get)
    ax.scatter([i], [vals[best_c]], s=200, facecolors="none",
               edgecolors=best_c, linewidths=1.8, zorder=4, clip_on=False)

# rotation panel
def plot2(series, color, ls="-", marker="o"):
    ax2.plot(x, series, ls=ls, color=color, lw=2.0, marker=marker, ms=9,
             markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
             zorder=3, clip_on=False)
plot2(resfm_r, C_R, ls="--", marker="s")
plot2(weight_r, C_W); plot2(wttt_r, C_T); plot2(madweight_r, C_M)
for i in range(5):
    vals2 = {C_W: weight_r[i], C_T: wttt_r[i], C_M: madweight_r[i], C_R: resfm_r[i]}
    bc = min(vals2, key=vals2.get)
    ax2.scatter([i], [vals2[bc]], s=200, facecolors="none", edgecolors=bc,
                linewidths=1.8, zorder=4, clip_on=False)
ax2.axvspan(-0.35, 1.5, color="#f2f7fb", zorder=0)
ax2.axvspan(2.5, 5.6, color="#fdf3ee", zorder=0)
ax2.set_ylabel("Rotation error (deg, log scale)", fontsize=14, color=INK)
for s in ["top", "right"]: ax2.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax2.spines[s].set_color(GRID)
ax2.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax2.set_axisbelow(True)
ax2.tick_params(colors=MUTED)

# regime annotations (shaded bands; labels pinned to the bottom to avoid collisions)
ax.axvspan(-0.35, 1.5, color="#f2f7fb", zorder=0)
ax.axvspan(2.5, 5.6, color="#fdf3ee", zorder=0)
ax.text(0.28, 0.03, "low contamination", transform=ax.transAxes, ha="center",
        fontsize=10, color=MUTED, style="italic")
ax2.text(0.22, 0.03, "low contamination", transform=ax2.transAxes, ha="center",
        fontsize=10, color=MUTED, style="italic")
ax.text(0.72, 0.03, "high contamination", transform=ax.transAxes, ha="center",
        fontsize=10, color=MUTED, style="italic")
ax2.text(0.72, 0.03, "high contamination", transform=ax2.transAxes, ha="center",
        fontsize=10, color=MUTED, style="italic")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=INK)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10, color=INK)
ax.set_xlim(-0.35, 5.6); ax2.set_xlim(-0.35, 5.6)
ax.set_ylabel("Translation error (log scale)", fontsize=14, color=INK)
ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 0.3, 1, 3, 10, 30]))
ax.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "0.3", "1", "3", "10", "30"]))
ax.tick_params(colors=MUTED)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

ax2.set_title("Rotation", fontsize=13, color=INK, fontweight="bold", pad=10, loc="left")
ax.set_title("Translation",
             fontsize=14, color=INK, fontweight="bold", pad=12, loc="left")
h, l = ax.get_legend_handles_labels()
ax2.legend(h, l, loc="lower center", frameon=False, fontsize=9.5, ncol=2,
           bbox_to_anchor=(0.5, 0.07))


plt.tight_layout()
out = "../claude specs/PATHA_fig1_complementarity"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf", "and .png")
