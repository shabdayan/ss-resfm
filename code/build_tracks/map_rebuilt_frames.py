#!/usr/bin/env python
"""Map megadepth_rebuilt track coordinates from the RAW-Flickr pixel frame into
the npz (COLMAP-undistorted) frame, then relabel outliers against Ps_gt.

Background (see memory megadepth-rebuild-blocker): npz K_gt/Ps_gt live in the
MegaDepth v1 undistorted frame; the rebuild fleet matched on raw images, so its
M is in raw pixels and its labels are invalid. The per-image raw intrinsics +
SIMPLE_RADIAL distortion come from the MegaDepth SfM COLMAP text models.

Mapping per camera: x_n = (u_raw - c_raw)/f_raw (distorted normalized) ->
invert SIMPLE_RADIAL (Newton) to ideal normalized -> u_ud = f_ud * x + c_ud
with (f_ud, c_ud) from the npz K_gt.

Acceptance: relabeled outlier_pct should return to the ~25% regime (raw-frame
labels showed a bogus median 63%).

Usage: python map_rebuilt_frames.py <scene>
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from appendix_c import label_outliers


def load_colmap_cams(scene):
    """image basename -> (f, cx, cy, k) from all sparse components."""
    out = {}
    for comp in sorted(glob.glob(os.path.join(
            CODE, "datasets", "raw_megadepth", "sfm_meta", "MegaDepth_v1_SfM",
            scene, "sparse", "*", "*"))):
        cam_path = os.path.join(comp, "cameras.txt")
        img_path = os.path.join(comp, "images.txt")
        if not (os.path.isfile(cam_path) and os.path.isfile(img_path)):
            continue
        cams = {}
        for line in open(cam_path):
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 8 and p[1] == "SIMPLE_RADIAL":
                cams[p[0]] = (float(p[4]), float(p[5]), float(p[6]), float(p[7]))
        skip_next = False
        for line in open(img_path):
            if line.startswith("#") or not line.strip():
                continue
            if skip_next:            # every second line is the 2D point list
                skip_next = False
                continue
            p = line.split()
            skip_next = True
            name = os.path.basename(p[-1])
            cam_id = p[-2]
            if cam_id in cams and name not in out:
                out[name] = cams[cam_id]
    return out


def undistort_simple_radial(xd, yd, k, iters=10):
    """Invert x_d = x*(1 + k*r^2) via Newton on the radius."""
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        f = 1.0 + k * r2
        x = xd / f
        y = yd / f
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    args = ap.parse_args()
    outdir = os.path.join(CODE, "datasets", "megadepth_rebuilt_ud")
    outp = os.path.join(outdir, f"{args.scene}.npz")
    if os.path.exists(outp):
        print(f"[{args.scene}] already mapped"); return

    reb = dict(np.load(os.path.join(CODE, "datasets", "megadepth_rebuilt",
                                    f"{args.scene}.npz"), allow_pickle=True))
    M = reb["M"].copy()
    K_ud = reb["K_gt"]
    names = [os.path.basename(str(n).strip()) for n in reb["namesList"]]
    raw_cams = load_colmap_cams(args.scene)
    m = M.shape[0] // 2
    unmapped = 0
    for i in range(m):
        if names[i] not in raw_cams:
            unmapped += 1
            M[2 * i] = 0; M[2 * i + 1] = 0        # drop this camera's observations
            continue
        f, cx, cy, k = raw_cams[names[i]]
        xs, ys = M[2 * i], M[2 * i + 1]
        vis = (xs != 0) | (ys != 0)
        xd = (xs[vis] - cx) / f
        yd = (ys[vis] - cy) / f
        xu, yu = undistort_simple_radial(xd, yd, k)
        fx, fy = K_ud[i][0, 0], K_ud[i][1, 1]
        cxu, cyu = K_ud[i][0, 2], K_ud[i][1, 2]
        M[2 * i][vis] = fx * xu + cxu
        M[2 * i + 1][vis] = fy * yu + cyu
    print(f"[{args.scene}] mapped {m - unmapped}/{m} cameras "
          f"({unmapped} without raw params)", flush=True)
    if unmapped > 0.1 * m:
        sys.exit(f"[{args.scene}] too many unmapped cameras -- aborting")

    old_pct = float(np.asarray(reb.get("outlier_pct", -1)))
    outliers, observed = label_outliers(M, reb["Ps_gt"])
    # convention identical to build_scene: percent of observed keypoints
    pct = 100.0 * outliers.sum() / max(observed.sum(), 1)
    reb["M"] = M
    reb["outliers2"] = outliers
    reb["outlier_pct"] = np.float64(pct)
    os.makedirs(outdir, exist_ok=True)
    np.savez(outp, **reb)
    print(f"[{args.scene}] RELABELED outlier_pct={pct:.2f} "
          f"(raw-frame bogus value was {old_pct:.2f})", flush=True)


if __name__ == "__main__":
    main()
