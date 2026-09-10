#!/usr/bin/env python
"""Per-observation feature extraction for the Plan-B feature-channel arms.

For a scene npz (measurement matrix M [2m, n]) and its images, compute a feature
vector for every observed (camera, point) entry and save a sidecar npz:
  obs_cam int32 [nnz], obs_pt int32 [nnz], F float16 [nnz, D]
Alignment is by explicit (obs_cam, obs_pt) join, never by array order.

Sources: sift (B00, cv2.SIFT.compute at the observation coordinate, size=16).
DINOv2 / MASt3R sources are added by the GPU extractor (same sidecar format).

Frame check: observation coords must lie inside the image, and the median
|2*principal_point - image_size| relative error must be < 15%, else abort —
a mismatch means the npz coordinate frame is not this image's pixel frame.
"""
import argparse, glob, os, sys
import numpy as np
import cv2

CODE = os.path.dirname(os.path.abspath(__file__))


def resolve_images(scene, names, raw, dataset):
    if dataset.startswith("1dsfm"):
        # namesList entries are global camera indices ("%06d") into the scene's
        # list.txt (one line per image, first token = relative image path).
        scene_dir = os.path.join(raw, scene)
        with open(os.path.join(scene_dir, "list.txt")) as f:
            lines = [ln.split()[0] for ln in f if ln.strip()]
        out = []
        for n in names:
            c = int(n.strip())
            p = os.path.join(scene_dir, lines[c]) if c < len(lines) else None
            out.append(p if p and os.path.exists(p) else None)
        return out
    pool = {}
    for p in glob.glob(os.path.join(raw, "MegaDepth_SfM", scene, "*")):
        pool[os.path.basename(p)] = p
    return [pool.get(os.path.basename(n.strip())) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--dataset", default="megadepth",
                    choices=["megadepth", "1dsfm", "1dsfm_hard_300"])
    ap.add_argument("--source", default="sift", choices=["sift"])
    ap.add_argument("--raw", default=None)
    ap.add_argument("--kp_size", type=float, default=16.0)
    args = ap.parse_args()

    outdir = os.path.join(CODE, "datasets", f"{args.dataset}_feats_{args.source}")
    out = os.path.join(outdir, f"{args.scene}.npz")
    if os.path.exists(out):
        print(f"[{args.scene}] already extracted"); return

    if args.raw is None:
        args.raw = os.path.join(CODE, "..", "datasets", "raw", "1dsfm") \
            if args.dataset.startswith("1dsfm") \
            else os.path.join(CODE, "datasets", "raw_megadepth")
    d = np.load(os.path.join(CODE, "datasets", args.dataset, f"{args.scene}.npz"),
                allow_pickle=True)
    M = d["M"]; K = d["K_gt"]; names = [str(n) for n in d["namesList"]]
    m = M.shape[0] // 2
    paths = resolve_images(args.scene, names, args.raw, args.dataset)

    sift = cv2.SIFT_create()
    obs_cam, obs_pt, feats = [], [], []
    frame_errs = []
    for i in range(m):
        xs_i = M[2 * i]; ys_i = M[2 * i + 1]
        vis = (xs_i != 0) | (ys_i != 0)
        idx = np.where(vis)[0]
        if len(idx) == 0 or paths[i] is None:
            continue
        img = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = K[i][0, 2], K[i][1, 2]
        frame_errs.append(max(abs(2 * cx - w) / w, abs(2 * cy - h) / h))
        inb = (xs_i[idx] >= 0) & (xs_i[idx] < w) & (ys_i[idx] >= 0) & (ys_i[idx] < h)
        idx = idx[inb]
        if len(idx) == 0:
            continue
        kps = [cv2.KeyPoint(float(xs_i[j]), float(ys_i[j]), args.kp_size) for j in idx]
        _, desc = sift.compute(img, kps)
        if desc is None:
            continue
        desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8)
        obs_cam.append(np.full(len(idx), i, np.int32))
        obs_pt.append(idx.astype(np.int32))
        feats.append(desc.astype(np.float16))

    med_err = float(np.median(frame_errs)) if frame_errs else 1.0
    nnz_total = int(((M[0::2] != 0) | (M[1::2] != 0)).sum())
    covered = sum(len(a) for a in obs_cam)
    print(f"[{args.scene}] frame-check median err {med_err:.3f}; "
          f"covered {covered}/{nnz_total} obs", flush=True)
    if med_err > 0.15:
        sys.exit(f"[{args.scene}] FRAME MISMATCH (median err {med_err:.3f}) -- aborting")
    if covered < 0.9 * nnz_total:
        sys.exit(f"[{args.scene}] coverage {covered}/{nnz_total} < 90% -- aborting")

    os.makedirs(outdir, exist_ok=True)
    np.savez(out, obs_cam=np.concatenate(obs_cam), obs_pt=np.concatenate(obs_pt),
             F=np.concatenate(feats), source=args.source, kp_size=args.kp_size,
             frame_err=med_err)
    print(f"[{args.scene}] DONE D={feats[0].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
