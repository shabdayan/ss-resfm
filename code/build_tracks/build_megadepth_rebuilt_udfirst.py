#!/usr/bin/env python
"""Undistort-first MegaDepth track rebuild (SIFT control, Design A substrate).

Builds tracks with the canonical Appendix-C pipeline on OUR COLMAP-undistorted
images (datasets/raw_megadepth/undistorted/<scene>/<comp>/), with GT expressed
in the images' own frame: decompose the npz Ps_gt = K_gt [R|t] into poses and
recombine with our per-camera undistorted pinhole K'_ud, so labels are computed
in a frame that matches the pixels by construction (no dependence on
reproducing MegaDepth's per-camera resize).

Sanity signal: outlier_pct should land near the original npz contamination
(~15-30%); a frame error would inflate it to 60%+ (the old failure mode).

Usage: python build_megadepth_rebuilt_udfirst.py <scene>
Output: datasets/megadepth_rebuilt_udfirst/<scene>.npz
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from appendix_c import build_scene


def decompose_P(P, K):
    """P = K [R|t] up to scale -> (R in SO(3), t)."""
    Rt = np.linalg.inv(K) @ P
    R_, t_ = Rt[:, :3], Rt[:, 3]
    U, S, Vt = np.linalg.svd(R_)
    lam = S.mean()
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R, lam = -R, -lam
    return R, t_ / lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    args = ap.parse_args()

    out_path = os.path.join(CODE, "datasets", "megadepth_rebuilt_udfirst",
                            f"{args.scene}.npz")
    if os.path.exists(out_path):
        print(f"[{args.scene}] already built"); return

    ref = dict(np.load(os.path.join(CODE, "datasets", "megadepth",
                                    f"{args.scene}.npz"), allow_pickle=True))
    names = [str(n) for n in ref["namesList"]]

    import pycolmap
    ud = os.path.join(CODE, "datasets", "raw_megadepth", "undistorted", args.scene)
    pool = {}   # basename -> (image path, K'_ud)
    for k in sorted(os.listdir(ud)):
        sp = os.path.join(ud, k, "sparse")
        if not os.path.isdir(sp):
            continue
        rec = pycolmap.Reconstruction(sp)
        for im in rec.images.values():
            p = os.path.join(ud, k, "images", im.name)
            if im.name in pool or not os.path.exists(p):
                continue
            fx, fy, cx, cy = rec.cameras[im.camera_id].params[:4]
            Kud = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            pool[im.name] = (p, Kud)

    keep, paths, Ks, Ps = [], [], [], []
    for idx, n in enumerate(names):
        base = os.path.basename(n.strip())
        if base not in pool:
            continue
        p, Kud = pool[base]
        R, t = decompose_P(ref["Ps_gt"][idx], ref["K_gt"][idx])
        keep.append(idx); paths.append(p); Ks.append(Kud)
        Ps.append(Kud @ np.hstack([R, t.reshape(3, 1)]))
    cov = len(keep) / max(1, len(names))
    print(f"[{args.scene}] {len(keep)}/{len(names)} cameras covered "
          f"({100*cov:.1f}%)", flush=True)
    if cov < 0.9:
        sys.exit(f"[{args.scene}] coverage {100*cov:.1f}% < 90% -- skipping")

    keep = np.asarray(keep)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ck = os.path.join(os.path.dirname(out_path), f".ck_{args.scene}.pkl")
    fields = build_scene(paths, np.asarray(Ps), np.asarray(Ks),
                         ref["namesList"][keep], checkpoint_path=ck)
    fields["covered_frac"] = np.float64(cov)
    np.savez(out_path, **fields)
    if os.path.exists(ck):
        os.remove(ck)
    print(f"[{args.scene}] DONE outlier_pct={float(fields['outlier_pct']):.2f}",
          flush=True)


if __name__ == "__main__":
    main()
