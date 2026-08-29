#!/usr/bin/env python
"""Figure 2 (Path A): method schematic.
Top: the shared ESFM pipeline (tracks -> equivariant net -> per-point score s +
reprojection error e -> outlier mechanism -> robust BA).
Bottom: the four mechanisms as weight-vs-signal transfer functions.
Colorblind-safe (Okabe-Ito); consistent with Figure 1.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

INK, MUTED, GRID = "#1a1a1a", "#666666", "#d5d5d5"
C_REMOVE, C_WEIGHT, C_HYBRID, C_MAD = "#009E73", "#0072B2", "#CC79A7", "#D55E00"

fig = plt.figure(figsize=(9.4, 6.0))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.5], hspace=0.55, wspace=0.34,
                      left=0.06, right=0.985, top=0.99, bottom=0.11)

# ---------- top: pipeline strip ----------
axp = fig.add_subplot(gs[0, :]); axp.axis("off")
axp.set_xlim(0, 104); axp.set_ylim(0, 20)
boxes = [
    (2,  "Point tracks\n$M\\ (2n{\\times}m)$", "#eef1f4"),
    (19.5, "ESFM\nequivariant net", "#eef1f4"),
    (38, "per-point\nscore $s$,  reproj $e$", "#eef1f4"),
    (57, "Outlier\nmechanism", "#fff2e8"),
    (74, "Robust\nBA", "#eef1f4"),
    (89, "Cameras\n+ 3D points", "#eef1f4"),
]
BW, BH, BY = 13.5, 9, 6
centers = []
for x, txt, fc in boxes:
    ec = C_MAD if "mechanism" in txt else "#b9c2cc"
    lw = 1.8 if "mechanism" in txt else 1.2
    axp.add_patch(FancyBboxPatch((x, BY), BW, BH, boxstyle="round,pad=0.15,rounding_size=1.2",
                                 fc=fc, ec=ec, lw=lw, zorder=2))
    axp.text(x + BW/2, BY + BH/2, txt, ha="center", va="center", fontsize=9,
             color=INK, zorder=3)
    centers.append((x, x + BW))
for (l, r) in list(zip(centers[:-1], centers[1:])):
    x0 = centers[centers.index(l)][1]; x1 = centers[centers.index(r)][0]
    axp.add_patch(FancyArrowPatch((x0, BY + BH/2), (x1, BY + BH/2),
                  arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.4, zorder=1))
axp.text(52, 18.5, "Shared pipeline (identical across all arms — only the mechanism box changes)",
         ha="center", fontsize=9, color=MUTED, style="italic")
# tie mechanism box to the panels below
axp.add_patch(FancyArrowPatch((57 + BW/2, BY), (57 + BW/2, 1.5),
              arrowstyle="-|>", mutation_scale=13, color=C_MAD, lw=1.4, zorder=1))

# ---------- bottom: four transfer functions ----------
s = np.linspace(0, 1, 400)
def panel(col, color, title, subtitle, wfun, extra=None, tau=None):
    ax = fig.add_subplot(gs[1, col])
    ax.plot(s, wfun(s), color=color, lw=2.4, zorder=3, clip_on=False, solid_capstyle="round")
    if tau is not None:
        for t in np.atleast_1d(tau):
            ax.axvline(t, color=MUTED, ls=":", lw=1.0, zorder=1)
    if extra: extra(ax)
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("outlier score $s$", fontsize=8.5, color=INK)
    ax.set_ylabel("weight $w$", fontsize=8.5, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(GRID)
    ax.set_title(title, color=color, fontsize=9.5, fontweight="bold", pad=13, loc="left")
    ax.text(0, 1.12, subtitle, transform=ax.transAxes, fontsize=7.8, color=MUTED)
    return ax

# remove: hard step at tau=0.6 (0.8 for clean sets)
def rm_extra(ax):
    ax.axvline(0.8, color=GRID, ls=":", lw=1.0, zorder=1)
    ax.text(0.58, 0.07, r"$\tau{=}0.6$", ha="right", fontsize=7.5, color=MUTED)
    ax.text(0.82, 0.45, "0.8\n(clean)", ha="left", fontsize=7, color=MUTED, va="center")
panel(0, C_REMOVE, "remove  (RESfM analogue)", r"drop if $s>\tau$   ($\tau{=}0.6$; 0.8 clean sets)",
      lambda s: np.where(s < 0.6, 1.0, 0.0), tau=0.6, extra=rm_extra)
# weight: soft 1-s
panel(1, C_WEIGHT, "weight  (SS-RESfM, ours)", "soft: $w = 1-s$, keep all",
      lambda s: 1 - s)
# hybrid: 3-band (percentile bands, not fixed tau)
def hyb(s):
    lo, hi = 0.35, 0.75
    w = np.where(s < lo, 1.0, np.where(s < hi, 1 - s, 0.0)); return w
panel(2, C_HYBRID, "hybrid  (SS-RESfM, ours)", "3-band: keep / soft / drop (percentiles)", hyb, tau=[0.35, 0.75])
# madweight: soft 1-s on MAD survivors; MAD removes by reproj error (any s)
def mad_extra(ax):
    xs = [0.15, 0.45, 0.82]
    ax.scatter(xs, [0.85, 0.55, 0.18], marker="x", s=45, color=C_MAD, lw=2, zorder=4, clip_on=False)
    ax.annotate("MAD removes high-reproj\npoints (any $s$)", xy=(0.45, 0.55),
                xytext=(0.30, 0.86), fontsize=7.2, color=C_MAD,
                arrowprops=dict(arrowstyle="->", color=C_MAD, lw=1))
ax_mad = panel(3, C_MAD, "madweight  (SS-RESfM, ours)", "MAD-remove (reproj $e$) + soft $1-s$",
               lambda s: 1 - s, extra=mad_extra)

# two column-separated callouts: left ties to the (s, e) signal box, right to the mechanism box
fig.text(0.24, 0.665,
         "Only  madweight  uses the\nreprojection-error branch $e$ (MAD removal);\n"
         "survivors are weighted by the head score $s$.",
         ha="center", va="center", fontsize=8.5, color=C_MAD)
fig.text(0.627, 0.665,
         "remove / weight / hybrid\nuse the head score $s$ alone.",
         ha="center", va="center", fontsize=8.5, color=C_MAD)
arr = ConnectionPatch(xyA=(0.30, 0.705), coordsA=fig.transFigure,
                      xyB=(43, 6.0), coordsB=axp.transData,
                      arrowstyle="-|>", mutation_scale=13, lw=1.6, color=C_MAD, zorder=5)
fig.add_artist(arr)

out = "../claude specs/PATHA_fig2_method"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf and .png")
