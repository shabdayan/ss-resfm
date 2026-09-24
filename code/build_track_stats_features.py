#!/usr/bin/env python
"""Design 4: per-observation GEOMETRIC track statistics as head input channels.

Unlike the dense-features program (appearance descriptors, all null), these are
statistics of the track/camera structure the network is reasoning about. For
each observed (camera i, point j) we emit 8 label-free channels:

  0 track length            (# cameras seeing point j)            / 50
  1 camera degree           (# points seen by camera i)           / 1000
  2 log10 track length
  3 mean co-visibility      (avg shared points with the other cameras on j) / 500
  4 min co-visibility       (weakest edge supporting j)           / 500
  5 coordinate radius       (|x| in normalized image coords)
  6 track coordinate spread (std of the point's coords across its cameras)
  7 camera coordinate spread(std of this camera's coords)

All are computable from M alone -- no labels, no GT poses, no images -- so they
are available at train and test time for every dataset.

Usage: build_track_stats_features.py <dataset> [--out <suffix>]
Writes datasets/<dataset>_feats_trackstats/<scene>.npz (obs_cam, obs_pt, F).
"""
import argparse, glob, os
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))


def scene_features(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    M = d["M"]; K = d["K_gt"]
    xs, ys = M[0::2], M[1::2]
    vis = (xs != 0) | (ys != 0)                       # [m, n]
    m, n = vis.shape
    tl = vis.sum(0).astype(np.float32)                # track length per point
    cd = vis.sum(1).astype(np.float32)                # camera degree
    co = (vis.astype(np.float32) @ vis.astype(np.float32).T)   # [m, m] co-visibility
    np.fill_diagonal(co, 0.0)
    # normalized coordinates (principal point removed, focal scaled)
    nx = (xs - K[:, 0, 2:3]) / K[:, 0, 0:1]
    ny = (ys - K[:, 1, 2:3]) / K[:, 1, 1:2]
    nx = np.where(vis, nx, np.nan); ny = np.where(vis, ny, np.nan)
    with np.errstate(invalid="ignore"):
        track_spread = np.nanstd(np.stack([nx, ny]), axis=(0, 1))       # per point
        cam_spread = np.nanstd(np.stack([nx, ny]), axis=(0, 2))         # per camera
    track_spread = np.nan_to_num(track_spread); cam_spread = np.nan_to_num(cam_spread)

    ci, pj = np.where(vis)
    # co-visibility of camera i with the other cameras observing point j
    mean_co = np.zeros(len(ci), np.float32); min_co = np.zeros(len(ci), np.float32)
    order = np.argsort(pj, kind="stable")
    ci_s, pj_s = ci[order], pj[order]
    bounds = np.searchsorted(pj_s, np.arange(n + 1))
    for j in range(n):
        a, b = bounds[j], bounds[j + 1]
        if b - a < 2:
            continue
        cams = ci_s[a:b]
        sub = co[np.ix_(cams, cams)]
        k = len(cams)
        s = sub.sum(1) / max(k - 1, 1)
        mean_co[a:b] = s
        min_co[a:b] = np.where(sub > 0, sub, np.inf).min(1)
    min_co[~np.isfinite(min_co)] = 0.0
    inv = np.empty(len(order), np.int64); inv[order] = np.arange(len(order))
    mean_co, min_co = mean_co[inv], min_co[inv]

    F = np.stack([
        tl[pj] / 50.0,
        cd[ci] / 1000.0,
        np.log10(tl[pj] + 1.0),
        mean_co / 500.0,
        min_co / 500.0,
        np.sqrt(np.nan_to_num(nx[ci, pj]) ** 2 + np.nan_to_num(ny[ci, pj]) ** 2),
        track_spread[pj],
        cam_spread[ci],
    ], axis=1).astype(np.float16)
    return ci.astype(np.int32), pj.astype(np.int32), F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    args = ap.parse_args()
    outdir = os.path.join(CODE, "datasets", f"{args.dataset}_feats_trackstats")
    os.makedirs(outdir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(CODE, "datasets", args.dataset, "*.npz"))):
        s = os.path.basename(p)
        out = os.path.join(outdir, s)
        if os.path.exists(out):
            continue
        ci, pj, F = scene_features(p)
        np.savez(out, obs_cam=ci, obs_pt=pj, F=F, source="trackstats")
        print(f"{s}: {len(ci)} obs, D={F.shape[1]}, "
              f"ranges {np.round(F.astype(np.float32).min(0),3)}..{np.round(F.astype(np.float32).max(0),3)}",
              flush=True)
    print("TRACKSTATS DONE", flush=True)


if __name__ == "__main__":
    main()
