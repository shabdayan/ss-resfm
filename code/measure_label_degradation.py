#!/usr/bin/env python
"""Label-degradation measurement (one scene).

Motivation: a supervised robust-SfM method (e.g. RESfM) trains on outlier
labels it must GENERATE, and the only label source when no ground-truth 3D
exists is a from-scratch SfM run (COLMAP) + a 4px reprojection cutoff. Those
COLMAP-derived labels are only as good as COLMAP's own reconstruction, which
degrades as scene contamination rises. This script quantifies that ceiling.

For one scene we compute two per-observation outlier labels on the SAME tracks:
  truth  = triangulate each track from the GT cameras (Ps_gt, DLT over its
           visible views), reproject with the GT camera, flag err > cut.   [4px]
  colmap = run pycolmap incremental mapping on the SAME correspondences, then
           reproject each observation using COLMAP's OWN 3D point + pose, flag
           err > cut. Observations COLMAP left untriangulated / on an
           unregistered camera count as COLMAP calling them outlier (a
           supervised method gets no inlier label there).

We then report, over the observations where BOTH a truth label exists and
COLMAP registered the camera:
  precision/recall/F1 of COLMAP labels vs truth  (outlier = positive class),
plus COLMAP registration & label coverage, COLMAP pose error, and the scene's
true contamination. One JSON line (prefixed JSONOUT:) goes to stdout for the
aggregator.

Truth uses the reference cameras only (method-independent); COLMAP is the
faithful stand-in for "regenerate labels on a new dataset". M is in pixels;
Ps_gt = K[R|t].

Usage: python measure_label_degradation.py <dataset> <scene>
                 [--max_cams 120] [--cut 4.0] [--min_pair 15] [--seed 0]
"""
import argparse, os, json, tempfile, shutil
import numpy as np
import pycolmap
# reuse the exact COLMAP-from-tracks builder used by the classical baseline
from colmap_baseline import build_database, add_matches, align_and_error


def load_and_subsample(dataset, scene, max_cams, seed):
    p = os.path.join(os.path.dirname(__file__), "datasets", dataset, f"{scene}.npz")
    d = np.load(p, allow_pickle=True)
    M = d["M"]; K = d["K_gt"]; Ps = d["Ps_gt"]
    O = d["outliers2"]                                  # (C,n) robust GT-pose 4px label (truth)
    C = M.shape[0] // 2
    if C > max_cams:                                    # cap cams so COLMAP stays tractable
        rng = np.random.RandomState(seed)
        sel = np.sort(rng.choice(C, max_cams, replace=False))
        Mrows = np.sort(np.concatenate([2 * sel, 2 * sel + 1]))
        M = M[Mrows]; K = K[sel]; Ps = Ps[sel]; O = O[sel]
        C = max_cams
    # drop tracks now seen by < 2 cams
    Xtmp = M.reshape(C, 2, -1)
    vis = (Xtmp[:, 0] != 0) | (Xtmp[:, 1] != 0)        # (C,n)
    keep = vis.sum(axis=0) >= 2
    M = M[:, keep]; O = O[:, keep]
    X = M.reshape(C, 2, -1); vis = (X[:, 0] != 0) | (X[:, 1] != 0)
    return M, X, vis, K, Ps, O, C


def truth_labels(O, vis):
    """Truth = the stored robust GT-pose outlier label (appendix_c: RANSAC-consensus
    triangulation under GT cameras, 4px cutoff). Returns (out, lab) both (C,n) bool:
    out[c,p]=labelled outlier ; lab[c,p]=observation is observed (has a truth label)."""
    lab = vis.copy()
    out = (O > 0.5) & vis
    return out, lab


def colmap_labels(rec, X, vis, K, kp_row, img_ids, cut):
    """Faithful RESfM-style labels from a COLMAP reconstruction. Each track
    (M column) is mapped to its COLMAP 3D point (majority vote over the
    observations COLMAP triangulated), then that point is reprojected into
    EVERY registered camera observing the track; outlier = reproj > cut. This
    labels a true-outlier observation too (COLMAP triangulates the point from
    its inlier views, so the outlier view reprojects far) instead of silently
    dropping it. Returns (cout, cver): cver[i,c]=a COLMAP verdict exists for
    that observation (track triangulated + camera registered)."""
    from collections import Counter
    C = X.shape[0]; n = X.shape[2]
    cout = np.zeros((C, n), dtype=bool); cver = np.zeros((C, n), dtype=bool)
    id2cam = {img_ids[i]: i for i in range(C)}             # colmap image id -> our cam idx
    inv = [{k: c for c, k in kp_row[i].items()} for i in range(C)]   # kp index -> track col
    # pass 1: track col -> COLMAP point3D id (majority over triangulated obs)
    col_pids = {}
    reg_img = {}
    for img_id, img in rec.images.items():
        if img_id not in id2cam:
            continue
        i = id2cam[img_id]; reg_img[i] = img
        for k, p2d in enumerate(img.points2D):
            if p2d.has_point3D():
                c = inv[i].get(k)
                if c is not None:
                    col_pids.setdefault(c, []).append(p2d.point3D_id)
    col_pid = {c: Counter(pids).most_common(1)[0][0] for c, pids in col_pids.items()}
    # pass 2: reproject each track's point into every registered camera observing it
    for i, img in reg_img.items():
        Rt = img.cam_from_world.matrix(); R = Rt[:, :3]; t = Rt[:, 3]
        cp = rec.cameras[img.camera_id].params        # COLMAP's own (refined) intrinsics
        fx, fy, cx, cy = cp[0], cp[1], cp[2], cp[3]    # PINHOLE = [fx, fy, cx, cy]
        for c in np.flatnonzero(vis[i]):
            pid = col_pid.get(int(c))
            if pid is None:                                # track never triangulated -> no verdict
                continue
            Xc = R @ rec.points3D[pid].xyz + t
            cver[i, c] = True
            if abs(Xc[2]) < 1e-9:
                cout[i, c] = True; continue
            u = fx * Xc[0] / Xc[2] + cx; v = fy * Xc[1] / Xc[2] + cy
            err = np.hypot(u - X[i, 0, c], v - X[i, 1, c])
            cout[i, c] = err > cut
    return cout, cver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("scene")
    ap.add_argument("--max_cams", type=int, default=120)
    ap.add_argument("--cut", type=float, default=4.0)
    ap.add_argument("--min_pair", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    M, X, vis, K, Ps, O, C = load_and_subsample(args.dataset, args.scene, args.max_cams, args.seed)
    nobs = int(vis.sum())
    print(f"[{args.dataset}/{args.scene}] {C} cams, {vis.shape[1]} tracks, {nobs} obs", flush=True)

    # truth
    t_out, t_lab = truth_labels(O, vis)
    contam = float(t_out[t_lab].mean()) if t_lab.any() else float("nan")

    # colmap
    work = tempfile.mkdtemp(prefix="labeldeg_")
    rec = None; nreg = 0; c_trans = float("nan")
    try:
        db_path = os.path.join(work, "database.db")
        db, img_ids, kp_row = build_database(db_path, X, vis, K, C)
        npairs = add_matches(db, X, vis, K, C, img_ids, kp_row, args.min_pair)
        db.close()
        print(f"  DB: {C} imgs, {npairs} verified pairs", flush=True)
        os.makedirs(os.path.join(work, "images"), exist_ok=True)
        out = os.path.join(work, "sparse"); os.makedirs(out, exist_ok=True)
        opts = pycolmap.IncrementalPipelineOptions()   # calibrated protocol: fix known intrinsics
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        recs = pycolmap.incremental_mapping(db_path, os.path.join(work, "images"), out, options=opts)
        if recs:
            rec = max(recs.values(), key=lambda r: len(r.images))
            nreg = len(rec.images)
            nr, _, trans = align_and_error(rec, Ps, K, C)
            if trans is not None:
                c_trans = float(trans.mean())
    finally:
        shutil.rmtree(work, ignore_errors=True)

    row = dict(dataset=args.dataset, scene=args.scene, cams=C, tracks=int(vis.shape[1]),
               obs=nobs, contamination=contam, colmap_reg_cams=nreg,
               colmap_reg_frac=(nreg / C if C else float("nan")),
               colmap_pose_trans_mean=c_trans)

    if rec is None:
        # COLMAP produced nothing: it cannot label ANY observation -> total ceiling.
        row.update(comparison_obs=0, coverage=0.0, precision=float("nan"),
                   recall=float("nan"), f1=float("nan"))
        print("  COLMAP produced NO reconstruction", flush=True)
    else:
        c_out, c_reg = colmap_labels(rec, X, vis, K, kp_row, img_ids, args.cut)
        comp = t_lab & c_reg                                # obs with BOTH a truth label and a COLMAP verdict
        ncomp = int(comp.sum())
        to = t_out[comp]; co = c_out[comp]                 # positive class = outlier
        tp = int((to & co).sum()); fp = int((~to & co).sum()); fn = int((to & ~co).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec_ = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec_ / (prec + rec_)) if (prec == prec and rec_ == rec_ and (prec + rec_) > 0) else float("nan")
        row.update(comparison_obs=ncomp,
                   coverage=(ncomp / int(t_lab.sum()) if t_lab.any() else float("nan")),
                   precision=prec, recall=rec_, f1=f1)
        print(f"  contam={contam*100:.2f}%  reg={nreg}/{C}  cov={row['coverage']*100:.1f}%  "
              f"P={prec:.3f} R={rec_:.3f} F1={f1:.3f}  pose_trans={c_trans:.3f}", flush=True)

    print("JSONOUT:" + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
