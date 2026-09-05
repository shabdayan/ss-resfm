#!/usr/bin/env python
"""Rebuild MegaDepth point tracks from RAW images with our Appendix-C pipeline
(track-pipeline robustness study): same builder as our OOD sets, applied to the
67 MegaDepth scenes of the RESfM split.

Per scene: take the EXACT camera set of the existing npz (namesList order, so
Group-2 subsampling and GT alignment are inherited), locate those images under
datasets/raw_megadepth/MegaDepth_v1/<scene>/dense*/imgs/, run SIFT -> exhaustive
RANSAC matching -> track chaining (appendix_c), label observations by GT-pose
triangulation at 4px (Ps_gt/K_gt from the existing npz), and write
datasets/megadepth_rebuilt/<scene>.npz in the standard format.

Usage: python build_megadepth_rebuilt.py <scene> [--raw datasets/raw_megadepth]
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from appendix_c import (extract_sift, MIN_TRACK_VIEWS, REPROJ_INLIER_PX,
                        RATIO_THR, RANSAC_PX, MIN_PAIR_INLIERS, label_outliers)
import cv2


def match_pair(d1, d2, k1, k2):
    if len(d1) == 0 or len(d2) == 0:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < RATIO_THR * n.distance]
    if len(good) < MIN_PAIR_INLIERS:
        return np.zeros((0, 2), int)
    p1 = np.float64([k1[m.queryIdx] for m in good])
    p2 = np.float64([k2[m.trainIdx] for m in good])
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, RANSAC_PX, 0.999)
    if F is None or mask is None or mask.sum() < MIN_PAIR_INLIERS:
        return np.zeros((0, 2), int)
    sel = mask.ravel().astype(bool)
    return np.array([[good[i].queryIdx, good[i].trainIdx]
                     for i in range(len(good)) if sel[i]], int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--raw", default=os.path.join(CODE, "datasets", "raw_megadepth"))
    args = ap.parse_args()

    ref = dict(np.load(os.path.join(CODE, "datasets", "megadepth", f"{args.scene}.npz"),
                       allow_pickle=True))
    names = [str(n) for n in ref["namesList"]]
    Ps, Ks, Ns = ref["Ps_gt"], ref["K_gt"], ref["Ns"]
    m = len(names)

    # locate images by basename under any dense*/imgs dir of the scene
    pool = {}
    for p in glob.glob(os.path.join(args.raw, "MegaDepth_v1", args.scene, "dense*", "imgs", "*")):
        pool[os.path.basename(p)] = p
    paths = []
    for n in names:
        base = os.path.basename(n)
        if base not in pool:
            sys.exit(f"missing image {base} for scene {args.scene} "
                     f"({len(pool)} images extracted)")
        paths.append(pool[base])
    print(f"[{args.scene}] {m} cameras, all images found", flush=True)

    kps, descs = extract_sift(paths)

    # union-find track chaining over exhaustive verified pairs
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x]); x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    npairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            mm = match_pair(descs[i], descs[j], kps[i], kps[j])
            if len(mm) == 0: continue
            npairs += 1
            for a, b in mm:
                union((i, int(a)), (j, int(b)))
    print(f"[{args.scene}] {npairs} verified pairs", flush=True)

    groups = {}
    for key in list(parent.keys()):
        groups.setdefault(find(key), []).append(key)
    tracks = []
    for members in groups.values():
        cams = [c for c, _ in members]
        if len(set(cams)) < MIN_TRACK_VIEWS or len(cams) != len(set(cams)):
            continue                      # <3 views or cycle-inconsistent
        tracks.append(members)
    n = len(tracks)
    print(f"[{args.scene}] {n} tracks", flush=True)

    M = np.zeros((2 * m, n))
    for j, members in enumerate(tracks):
        for c, k in members:
            M[2 * c, j], M[2 * c + 1, j] = kps[c][k]

    # GT-pose RANSAC-consensus triangulation labels (4px), the canonical
    # labeler shared with build_1dsfm and the other OOD builders.
    out = label_outliers(M, Ps).astype(bool)
    X2 = M.reshape(m, 2, -1); vis = (X2[:, 0] != 0) | (X2[:, 1] != 0)
    pct = 100.0 * (out & vis).sum() / max(1, vis.sum())
    print(f"[{args.scene}] outlier_pct {pct:.1f}%", flush=True)

    os.makedirs(os.path.join(CODE, "datasets", "megadepth_rebuilt"), exist_ok=True)
    np.savez(os.path.join(CODE, "datasets", "megadepth_rebuilt", f"{args.scene}.npz"),
             M=M, Ns=Ns, Ps_gt=Ps, K_gt=Ks, outliers2=out,
             outlier_pct=np.float64(pct / 100.0),   # MegaDepth convention: fraction
             namesList=ref["namesList"])
    print(f"[{args.scene}] DONE", flush=True)


if __name__ == "__main__":
    main()
