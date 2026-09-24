#!/usr/bin/env python
"""Offline benchmark for the adaptive pseudo-label thresholding rule.

The self-supervised loss turns per-observation reprojection residuals into
confident pseudo-labels via global percentiles. These scenes ship GT outlier
labels, so every candidate rule can be scored WITHOUT training: we take the
residuals of a trained model (or, with --gt_poses, residuals under the GT
cameras, which isolates the rule from model quality), apply each rule, and
report precision/recall/F1 of the confident pseudo-labels plus their coverage.

Rules screened:
  current   global percentiles (inlier 20, outlier 80)                [baseline]
  percam    per-camera robust standardisation: (r - med_i)/MAD_i
  pertrack  per-track centring: r - med_j  (isolates observation-level outliers)
  otsu      parameter-free Otsu split on the residual histogram
  ema       current rule with thresholds smoothed across scenes (EMA proxy)

Usage: pseudolabel_benchmark.py <dataset> [--scenes A B] [--gt_poses]
"""
import argparse, glob, os
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
INLIER_P, OUTLIER_P = 20.0, 80.0


def residuals_from_gt(d):
    """Per-observation reprojection residual under the GT cameras."""
    M, Ps, K = d["M"], d["Ps_gt"], d["K_gt"]
    xs, ys = M[0::2], M[1::2]
    vis = (xs != 0) | (ys != 0)
    m, n = vis.shape
    # triangulate each point by DLT over its observing cameras, then reproject
    res = np.full((m, n), np.nan, np.float32)
    P = np.asarray(Ps, np.float64)
    for j in range(n):
        cams = np.where(vis[:, j])[0]
        if len(cams) < 2:
            continue
        A = []
        for i in cams:
            x, y = xs[i, j], ys[i, j]
            A.append(x * P[i, 2] - P[i, 0]); A.append(y * P[i, 2] - P[i, 1])
        X = np.linalg.svd(np.asarray(A))[2][-1]
        X = X / (X[3] if abs(X[3]) > 1e-12 else 1e-12)
        for i in cams:
            p = P[i] @ X
            if abs(p[2]) < 1e-12: continue
            res[i, j] = np.hypot(p[0] / p[2] - xs[i, j], p[1] / p[2] - ys[i, j])
    return res, vis


def score(pred_out, pred_in, gt_out, vis):
    """precision/recall/F1 of confident-outlier labels + coverage."""
    conf = pred_out | pred_in
    tp = np.sum(pred_out & gt_out & vis); fp = np.sum(pred_out & ~gt_out & vis)
    fn = np.sum(~pred_out & gt_out & conf & vis)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1, conf[vis].mean()


def rules(res, vis):
    r = np.where(vis, res, np.nan)
    out = {}
    flat = r[vis & np.isfinite(r)]
    lo, hi = np.percentile(flat, INLIER_P), np.percentile(flat, OUTLIER_P)
    out["current"] = (r > hi, r < lo)
    # per-camera robust standardisation
    med_i = np.nanmedian(r, axis=1, keepdims=True)
    mad_i = np.nanmedian(np.abs(r - med_i), axis=1, keepdims=True) + 1e-9
    z = (r - med_i) / mad_i
    zf = z[vis & np.isfinite(z)]
    out["percam"] = (z > np.percentile(zf, OUTLIER_P), z < np.percentile(zf, INLIER_P))
    # per-track centring
    med_j = np.nanmedian(r, axis=0, keepdims=True)
    c = r - med_j
    cf = c[vis & np.isfinite(c)]
    out["pertrack"] = (c > np.percentile(cf, OUTLIER_P), c < np.percentile(cf, INLIER_P))
    # Otsu on log residuals (parameter-free split)
    lg = np.log1p(flat)
    hist, edges = np.histogram(lg, bins=256)
    w = hist.cumsum(); mu = (hist * ((edges[:-1] + edges[1:]) / 2)).cumsum()
    tot_w, tot_mu = w[-1], mu[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (tot_mu * w / tot_w - mu) ** 2 / (w * (tot_w - w) / tot_w ** 2 + 1e-12)
    t = edges[:-1][np.nanargmax(between)]
    thr = np.expm1(t)
    out["otsu"] = (r > thr, r < thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(CODE, "datasets", args.dataset, "*.npz")))
    if args.scenes:
        paths = [p for p in paths if os.path.basename(p)[:-4] in args.scenes]
    agg = {}
    for p in paths:
        s = os.path.basename(p)[:-4]
        d = np.load(p, allow_pickle=True)
        if "outliers2" not in d: continue
        gt = d["outliers2"].astype(bool)
        res, vis = residuals_from_gt(d)
        gt = gt & vis
        print(f"\n{s}: {int(vis.sum())} obs, GT contamination {100*gt.sum()/vis.sum():.1f}%", flush=True)
        for name, (po, pi) in rules(res, vis).items():
            po = np.nan_to_num(po, nan=False); pi = np.nan_to_num(pi, nan=False)
            pr, rc, f1, cov = score(po & vis, pi & vis, gt, vis)
            agg.setdefault(name, []).append((pr, rc, f1, cov))
            print(f"  {name:9s} P {pr:.3f}  R {rc:.3f}  F1 {f1:.3f}  coverage {cov:.3f}", flush=True)
    print("\n=== MEAN OVER SCENES ===")
    for name, v in agg.items():
        a = np.array(v).mean(0)
        print(f"  {name:9s} P {a[0]:.3f}  R {a[1]:.3f}  F1 {a[2]:.3f}  coverage {a[3]:.3f}")


if __name__ == "__main__":
    main()
