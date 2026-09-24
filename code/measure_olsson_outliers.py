#!/usr/bin/env python
# Measure the reprojection-based outlier rate of the Olsson (Euclidean) tracks.
# Olsson has NO outlier labels (clean benchmark), so we replicate the RESfM label
# definition: triangulate each 3D point from the GT cameras (DLT over its visible
# views), reproject, and flag observations whose reprojection error exceeds 4 px.
# Reports per-scene rate + dataset mean/median. M is in pixels; Ps_gt = K[R|t].
import numpy as np, glob, os, sys

CUT = 4.0  # pixels, matches RESfM's 4px outlier cutoff
def scene_rate(f):
    d = np.load(f, allow_pickle=True)
    M = d['M']; Ps = d['Ps_gt']
    C = M.shape[0] // 2; P = M.shape[1]
    X2 = M.reshape(C, 2, P)
    vis = (X2[:, 0] != 0) | (X2[:, 1] != 0)          # (C,P) observed mask
    nobs = 0; nout = 0
    for p in range(P):
        cams = np.where(vis[:, p])[0]
        if len(cams) < 2:
            continue
        A = np.empty((2 * len(cams), 4))
        for k, c in enumerate(cams):
            Pc = Ps[c]; x = X2[c, 0, p]; y = X2[c, 1, p]
            A[2 * k]     = x * Pc[2] - Pc[0]
            A[2 * k + 1] = y * Pc[2] - Pc[1]
        _, _, Vt = np.linalg.svd(A)
        Xw = Vt[-1]
        if abs(Xw[3]) < 1e-12:
            continue
        Xw = Xw / Xw[3]
        for c in cams:
            pr = Ps[c] @ Xw
            if abs(pr[2]) < 1e-12:
                continue
            u, v = pr[0] / pr[2], pr[1] / pr[2]
            err = np.hypot(u - X2[c, 0, p], v - X2[c, 1, p])
            nobs += 1
            nout += int(err > CUT)
    return 100.0 * nout / nobs if nobs else np.nan

files = sorted(glob.glob("datasets/Euclidean/*.npz"))
rows = []
for f in files:
    r = scene_rate(f)
    rows.append((os.path.basename(f)[:-4], r))
    print(f"  {os.path.basename(f)[:-4]:45s} {r:6.2f}%", flush=True)
rs = np.array([r for _, r in rows if not np.isnan(r)])
print(f"\n=== OLSSON (Euclidean) reprojection outlier rate @ {CUT}px, n={len(rs)} ===")
print(f"  mean={rs.mean():.2f}%  median={np.median(rs):.2f}%  range=[{rs.min():.2f}, {rs.max():.2f}]")
