#!/usr/bin/env python
"""Undistort-first MegaDepth rebuild, step 1: reproduce RESfM's COLMAP
image_undistorter on the raw SfM Flickr images, restricted to the scene npz's
camera set. Runs per manhattan component, then validates the produced
undistorted PINHOLE cameras against the npz K_gt (which lives in the
COLMAP-undistorted frame). High per-camera agreement validates the
undistort-first architecture for the scene; the undistorted images are the
substrate for the dense-features track rebuilds.

Output: datasets/raw_megadepth/undistorted/<scene>/<comp>/{images,sparse,...}
and a summary line: coverage + median/max relative K error.
"""
import argparse, os, sys
import numpy as np
import pycolmap

CODE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(CODE, "datasets", "raw_megadepth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--outroot", default=os.path.join(RAW, "undistorted"))
    args = ap.parse_args()

    npz = np.load(os.path.join(CODE, "datasets", "megadepth", f"{args.scene}.npz"),
                  allow_pickle=True)
    names = [str(n).strip() for n in npz["namesList"]]
    K_gt = npz["K_gt"]
    want = set(names)
    img_dir = os.path.join(RAW, "MegaDepth_SfM", args.scene)
    on_disk = set(os.listdir(img_dir))
    man = os.path.join(RAW, "sfm_meta", "MegaDepth_v1_SfM", args.scene,
                       "sparse", "manhattan")
    covered = {}
    for k in sorted(os.listdir(man)):
        comp = os.path.join(man, k)
        try:
            rec = pycolmap.Reconstruction(comp)
        except Exception as e:
            print(f"[{args.scene}/{k}] unreadable model: {e}", flush=True)
            continue
        hit = sorted(im.name for im in rec.images.values()
                     if im.name in want and im.name in on_disk
                     and im.name not in covered)
        print(f"[{args.scene}/{k}] {len(rec.images)} model imgs, "
              f"{len(hit)} new npz matches", flush=True)
        if not hit:
            continue
        out = os.path.join(args.outroot, args.scene, k)
        os.makedirs(out, exist_ok=True)
        pycolmap.undistort_images(out, comp, img_dir, image_list=hit)
        rec2 = pycolmap.Reconstruction(os.path.join(out, "sparse"))
        for im in rec2.images.values():
            if im.name not in want:
                continue
            cam = rec2.cameras[im.camera_id]
            fx, fy, cx, cy = cam.params[:4]
            covered[im.name] = (fx, fy, cx, cy)

    errs = []
    for i, n in enumerate(names):
        if n not in covered:
            continue
        fx, fy, cx, cy = covered[n]
        K = K_gt[i]
        errs.append(max(abs(fx - K[0, 0]) / K[0, 0], abs(fy - K[1, 1]) / K[1, 1],
                        abs(cx - K[0, 2]) / max(K[0, 2], 1.0),
                        abs(cy - K[1, 2]) / max(K[1, 2], 1.0)))
    errs = np.array(errs)
    cov = len(errs) / max(1, len(names))
    med = float(np.median(errs)) if len(errs) else float("nan")
    mx = float(errs.max()) if len(errs) else float("nan")
    frac_ok = float((errs < 0.01).mean()) if len(errs) else 0.0
    print(f"[{args.scene}] SUMMARY coverage {len(errs)}/{len(names)} "
          f"({100*cov:.1f}%)  K err median {med:.4f} max {mx:.4f}  "
          f"<1% agreement {100*frac_ok:.1f}%", flush=True)


if __name__ == "__main__":
    main()
