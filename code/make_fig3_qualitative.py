#!/usr/bin/env python
"""Figure 3 (Path A): qualitative 1DSfM reconstruction — madweight vs RESfM-from-scratch
on NYC_Library, where label-free madweight wins (1.32 deg vs 5.42 deg translation).
Overlays BA-aligned predicted camera centers on GT in a shared PCA view.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, glob

SCENE = "NYC_Library"
ARMS = [
    ("uesfm_finelr_rf_madweight_1dsfm_eval", "madweight (ours, label-free)", 2.94, "#D55E00"),
    ("resfm_finelr_1dsfm_eval",              "RESfM (from-scratch, supervised)", 6.28, "#555555"),
]
INK, MUTED, GTC = "#1a1a1a", "#777777", "#9aa7b4"

def load(root):
    f = glob.glob(f"results/crossdataset/{root}/{SCENE}_ba/forFigures/*Final_Cameras.npz")[0]
    d = np.load(f, allow_pickle=True)
    Rg, tg = np.asarray(d["Rs_gt"]), np.asarray(d["ts_gt"])
    Rp, tp = np.asarray(d["Rs_ba_final_fixed"]), np.asarray(d["ts_ba_final_fixed"])
    idx = np.asarray(d["validCamIndices_ba_final"]).reshape(-1).astype(int)
    def cen(R, t):
        return np.array([-Ri.T @ ti.reshape(-1) for Ri, ti in zip(R, t)])
    Cg = cen(Rg, tg)                      # all GT camera centers
    Cp = cen(Rp, tp)                      # predicted (registered subset), aligned to GT frame
    Cg_reg = Cg[idx] if idx.max() < len(Cg) and len(idx) == len(Cp) else Cg[:len(Cp)]
    return Cg, Cp, Cg_reg

# shared PCA frame from the first arm's GT centers
Cg0, _, _ = load(ARMS[0][0])
mu = Cg0.mean(0)
_, _, Vt = np.linalg.svd(Cg0 - mu, full_matrices=False)
V = Vt[:2].T
proj = lambda C: (C - mu) @ V
# scene scale (3D) -> common gross-drift threshold for highlighting worst cameras
scale3d = np.percentile(np.linalg.norm(Cg0 - np.median(Cg0, 0), axis=1), 85)
GROSS = 0.5 * scale3d

fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.3))
# common square limits from GT extent — tight so the coherent structure fills the
# panel and gross-error cameras shoot off the edges (the visual signal of drift)
allg = proj(Cg0)
c = np.median(allg, 0); r = np.percentile(np.abs(allg - c), 85) * 1.15
xlim = (c[0] - r, c[0] + r); ylim = (c[1] - r, c[1] + r)
for ax, (root, name, err, col) in zip(axes, ARMS):
    Cg, Cp, Cg_reg = load(root)
    Pg, Pp, Pgr = proj(Cg), proj(Cp), proj(Cg_reg)
    # per-camera 3D drift (registered GT -> predicted); flag gross-drift cameras
    n = min(len(Cp), len(Cg_reg))
    d3 = np.linalg.norm(Cp[:n] - Cg_reg[:n], axis=1)
    gross = d3 > GROSS
    ng = int(gross.sum())
    # GT camera centers (context)
    ax.scatter(Pg[:, 0], Pg[:, 1], s=14, facecolors="none", edgecolors=GTC, lw=0.9,
               label="GT cameras", zorder=2)
    # connectors + emphasis ONLY for gross-drift cameras (declutter)
    for i in np.where(gross)[0]:
        ax.plot([Pgr[i, 0], Pp[i, 0]], [Pgr[i, 1], Pp[i, 1]], color=col, lw=0.7,
                alpha=0.55, zorder=1)
    # well-placed predicted cameras: small, faint
    ax.scatter(Pp[:n][~gross][:, 0], Pp[:n][~gross][:, 1], s=9, color=col, alpha=0.45,
               edgecolors="none", label="predicted (on GT)", zorder=3)
    # gross-drift predicted cameras: emphasized
    ax.scatter(Pp[:n][gross][:, 0], Pp[:n][gross][:, 1], s=26, color=col,
               edgecolors="white", lw=0.5, label=f"gross drift  (n={ng})", zorder=4)
    ax.set_title(f"{name}\ntranslation error {err:.2f}$\\degree$   ·   {ng} gross-drift cams",
                 fontsize=13, color=col if col != "#555555" else INK, pad=8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#dddddd")
    ax.legend(loc="upper right", frameon=False, fontsize=11)

fig.suptitle(f"1DSfM  ·  {SCENE}  (32.6% outliers)  —  BA-aligned camera centers vs ground truth",
             fontsize=11.5, color=INK, y=1.02, fontweight="bold")
fig.text(0.5, -0.03,
         "Open markers = GT camera centers; small faint dots = well-placed estimates; large dots + connectors = "
         "gross-drift cameras (>0.5·scene scale off GT).\nLabel-free madweight leaves far fewer gross-drift "
         "cameras than the supervised from-scratch baseline. Cameras aligned to GT; same PCA view for both.",
         ha="center", fontsize=7.8, color=MUTED)

plt.tight_layout()
out = "../claude specs/PATHA_fig3_qualitative"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf and .png")
