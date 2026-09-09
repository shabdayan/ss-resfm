#!/usr/bin/env python
"""ESFM* track builder: copy each dataset's npz with GT-labeled outlier
observations removed (RESfM's ESFM* protocol: train AND test on outlier-free
tracks). Uses the robust GT-pose 4px labels already stored as 'outliers2'.

Output: datasets/<name>_star/<scene>.npz  (M zeroed at outlier obs, tracks with
<2 remaining views dropped, outlier_pct=0)
"""
import numpy as np, glob, os, sys

SRC = ["megadepth", "1dsfm", "1dsfm_hard_300", "strecha", "blendedmvs", ("olsson_id", "olsson_star")]

for name in SRC:
    if isinstance(name, tuple):
        name, outname = name
    else:
        outname = f"{name}_star"
    outdir = f"datasets/{outname}"
    os.makedirs(outdir, exist_ok=True)
    for f in sorted(glob.glob(f"datasets/{name}/*.npz")):
        scene = os.path.basename(f)
        dst = os.path.join(outdir, scene)
        if os.path.exists(dst):
            continue
        d = dict(np.load(f, allow_pickle=True))
        M, O = d["M"], d["outliers2"].astype(bool)
        C = O.shape[0]
        vis = (M[0::2] != 0) | (M[1::2] != 0)          # (C,n)
        out = O & vis
        M2 = M.copy()
        M2[0::2][out] = 0.0
        M2[1::2][out] = 0.0
        vis2 = (M2[0::2] != 0) | (M2[1::2] != 0)
        keep = vis2.sum(axis=0) >= 2                    # tracks with >=2 views survive
        d["M"] = M2[:, keep]
        d["outliers2"] = np.zeros((C, int(keep.sum())), dtype=bool)
        d["outlier_pct"] = np.float64(0.0)
        np.savez(dst, **d)
        removed = out.sum() / max(1, vis.sum())
        print(f"{name}/{scene}: obs removed {100*removed:.1f}%  tracks {M.shape[1]}->{int(keep.sum())}", flush=True)
print("ALL DONE")
