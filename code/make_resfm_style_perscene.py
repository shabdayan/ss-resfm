#!/usr/bin/env python
"""Regenerate supp per-scene Tables 9/10 in RESfM Table-1 style:
Scene | Nc (input images) | %out | per model: Nr, rot, trans.
Winner per scene (min trans, min rot) in bold+underline. MegaDepth split into
Group 1 (<1000 images) above the midrule and Group 2 (300-camera subsamples)
below, following RESfM. Prints LaTeX tabular bodies to stdout.
"""
import glob, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "results", "multiscene")
C = os.path.join(HERE, "results", "crossdataset")

ARMS = [("madweight", "madweight"), ("weight", "weight"),
        ("weight_ttt", "weight+TTT"), (None, "RESfM")]


def root(arm, ds):
    if arm is None:
        return f"{M}/resfm_finelr_megadepth_eval" if ds == "megadepth" \
            else f"{C}/resfm_finelr_{ds}_eval"
    md_arm = "wttt" if arm == "weight_ttt" else arm   # MD roots use the short name
    return f"{M}/uesfm_finelr_{md_arm}_megadepth_eval" if ds == "megadepth" \
        else f"{C}/uesfm_finelr_rf_{arm}_{ds}_eval"


def per_scene(r):
    d = {}
    for f in glob.glob(f"{r}/*_ba/Results_FINE_TUNE_stage_1_*.xlsx"):
        s = f.split("/")[-2][:-3]
        row = pd.read_excel(f).iloc[0]
        d[s] = (float(row["ts_ba_final_mean"]), float(row["Rs_ba_final_mean"]),
                int(row["#registered_cams_final"]))
    return d


def scene_meta(ds, scan):
    f = os.path.join(HERE, "datasets", ds, f"{scan}.npz")
    z = np.load(f, allow_pickle=True)
    nc = z["M"].shape[0] // 2
    op = float(z["outlier_pct"])
    if op <= 1.0:            # MegaDepth stores a fraction; OOD sets store percent
        op *= 100.0
    return nc, op


def fmt(v, best, nd=2):
    s = f"{v:.{nd}f}"
    return f"$\\underline{{\\mathbf{{{s}}}}}$" if best else s


# RESfM paper Table 1's exact groups and row order (outlier fraction descending):
MD_G1 = ["0238", "0060", "0197", "0094", "0265", "0083", "0076", "0185",
         "0048", "0024", "0223", "5016", "0046"]
MD_G2 = ["0099", "1001", "0231", "0411", "0377", "0102", "0147", "0148",
         "0446", "0022", "0327", "0015", "0455", "0496", "1589", "0012",
         "0104", "0019", "0063", "0130", "0080", "0240", "0007"]


def build(ds, dsdir, order=None):
    data = [per_scene(root(a, ds)) for a, _ in ARMS]
    scenes = sorted(set.intersection(*[set(d) for d in data]))
    rows = {}
    for sc in scenes:
        nc, op = scene_meta(dsdir, sc)
        tvals = [data[i][sc][0] for i in range(4)]
        rvals = [data[i][sc][1] for i in range(4)]
        bt, br = int(np.argmin(tvals)), int(np.argmin(rvals))
        cells = []
        for i in range(4):
            t, r, nr = data[i][sc]
            cells.append(f"{nr} & {fmt(r, i == br)} & {fmt(t, i == bt)}")
        name = sc.replace("_", "\\_")
        rows[sc] = (op, f"{name} & {nc} & {op:.0f}\\% & " + " & ".join(cells) + "\\\\")
    if order is None:                       # sort by outlier fraction descending
        order = sorted(rows, key=lambda s: -rows[s][0])
    return [rows[s][1] for s in order if s in rows]


print("%%%%%%%%%% 1dsfm %%%%%%%%%%")
print("\n".join(build("1dsfm", "1dsfm")))
print()
print("%%%%%%%%%% megadepth %%%%%%%%%%")
print("\n".join(build("megadepth", "megadepth", MD_G1)))
print("\\midrule")
print("\n".join(build("megadepth", "megadepth", MD_G2)))
