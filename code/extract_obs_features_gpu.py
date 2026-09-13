#!/usr/bin/env python
"""B1 per-observation feature extraction: MASt3R ViT-L *encoder* patch features
(pairing-free) sampled bilinearly at track coordinates, projected 1024->64 by a
fixed seeded orthogonal projection (same for every scene; documented, leak-free).

Runs in the dense env (tools/dense/mm/envs/dense). Sidecar format identical to
the SIFT extractor: obs_cam, obs_pt, F [K,64] fp16, source='b1_mast3r'.

Image resolution follows extract_obs_features.resolve_images (1DSfM list.txt /
MegaDepth undistorted + per-camera rescale into the npz frame); here the image
is additionally resized so its long side is 512 (multiple-of-16 dims), and the
track coordinates are mapped by the same scale factors before sampling.

Usage: extract_obs_features_gpu.py <scene> --dataset {1dsfm,1dsfm_hard_300,megadepth}
"""
import argparse, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
DENSE = os.path.join(CODE, "..", "tools", "dense")
# croco's `models` package must be importable (dust3r's dpt_head does
# `from models.dpt_block import ...`); keep CODE last so our own `models/`
# package does not shadow it.
sys.path.insert(0, os.path.join(DENSE, "mast3r"))
sys.path.insert(0, os.path.join(DENSE, "mast3r", "dust3r"))
sys.path.insert(0, os.path.join(DENSE, "mast3r", "dust3r", "croco"))
sys.path.append(CODE)

import torch
import cv2
from mast3r.model import AsymmetricMASt3R
from extract_obs_features import resolve_images

CKPT = os.path.join(DENSE, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
IMAGENET_MEAN = np.array([0.5, 0.5, 0.5], np.float32)   # dust3r ImgNorm
IMAGENET_STD = np.array([0.5, 0.5, 0.5], np.float32)


def prep(img, scale_to_npz):
    """BGR uint8 -> (tensor [1,3,H,W], sx, sy) mapping npz coords -> tensor grid."""
    if abs(scale_to_npz - 1.0) > 1e-3:
        img = cv2.resize(img, None, fx=scale_to_npz, fy=scale_to_npz,
                         interpolation=cv2.INTER_AREA if scale_to_npz < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    s = 512.0 / max(h, w)
    nw, nh = max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))[None]
    return t, nw / float(w), nh / float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--dataset", default="1dsfm",
                    choices=["megadepth", "1dsfm", "1dsfm_hard_300"])
    ap.add_argument("--out_dim", type=int, default=64)
    args = ap.parse_args()

    outdir = os.path.join(CODE, "datasets", f"{args.dataset}_feats_b1mast3r")
    out = os.path.join(outdir, f"{args.scene}.npz")
    if os.path.exists(out):
        print(f"[{args.scene}] already extracted"); return

    raw = os.path.join(CODE, "..", "datasets", "raw", "1dsfm") \
        if args.dataset.startswith("1dsfm") else os.path.join(CODE, "datasets", "raw_megadepth")
    d = np.load(os.path.join(CODE, "datasets", args.dataset, f"{args.scene}.npz"),
                allow_pickle=True)
    M = d["M"]; K = d["K_gt"]; names = [str(n) for n in d["namesList"]]
    m = M.shape[0] // 2
    paths, scales = resolve_images(args.scene, names, raw, args.dataset, K_gt=K)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricMASt3R.from_pretrained(CKPT).to(dev).eval()
    enc_dim = model.enc_embed_dim  # 1024 for ViT-L
    rng = np.random.RandomState(0)
    A = np.linalg.qr(rng.randn(enc_dim, args.out_dim))[0].astype(np.float32)  # fixed projection
    proj = torch.from_numpy(A).to(dev)

    obs_cam, obs_pt, feats, frame_errs = [], [], [], []
    with torch.no_grad():
        for i in range(m):
            xs, ys = M[2 * i], M[2 * i + 1]
            vis = (xs != 0) | (ys != 0)
            idx = np.where(vis)[0]
            if len(idx) == 0 or paths[i] is None:
                continue
            img = cv2.imread(paths[i], cv2.IMREAD_COLOR)
            if img is None:
                continue
            # frame guard in the npz frame (before the 512 resize)
            h0 = img.shape[0] * scales[i]; w0 = img.shape[1] * scales[i]
            cx, cy = K[i][0, 2], K[i][1, 2]
            frame_errs.append(max(abs(2 * cx - w0) / w0, abs(2 * cy - h0) / h0))
            t, sx, sy = prep(img, scales[i])
            t = t.to(dev)
            true_shape = torch.tensor([[t.shape[2], t.shape[3]]], device=dev)
            enc_out = model._encode_image(t, true_shape)
            feat = enc_out[0]
            H16, W16 = t.shape[2] // 16, t.shape[3] // 16
            fmap = feat[0].transpose(0, 1).reshape(-1, H16, W16)[None]  # [1,C,h,w]
            # npz coords -> 512-frame pixels -> normalized grid coords
            gx = (xs[idx] * sx / (t.shape[3] - 1)) * 2 - 1
            gy = (ys[idx] * sy / (t.shape[2] - 1)) * 2 - 1
            inb = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
            sel = idx[inb]
            if len(sel) == 0:
                continue
            grid = torch.tensor(np.stack([gx[inb], gy[inb]], 1), dtype=torch.float32,
                                device=dev).view(1, -1, 1, 2)
            smp = torch.nn.functional.grid_sample(fmap.float(), grid,
                                                  align_corners=True)[0, :, :, 0].T
            smp = smp @ proj
            smp = smp / (smp.norm(dim=1, keepdim=True) + 1e-8)
            obs_cam.append(np.full(len(sel), i, np.int32))
            obs_pt.append(sel.astype(np.int32))
            feats.append(smp.cpu().numpy().astype(np.float16))
            if i % 50 == 0:
                print(f"[{args.scene}] cam {i}/{m}", flush=True)

    med_err = float(np.median(frame_errs)) if frame_errs else 1.0
    nnz = int(((M[0::2] != 0) | (M[1::2] != 0)).sum())
    covered = sum(len(a) for a in obs_cam)
    print(f"[{args.scene}] frame-check median err {med_err:.3f}; covered {covered}/{nnz} obs", flush=True)
    if med_err > 0.15:
        sys.exit(f"[{args.scene}] FRAME MISMATCH (median err {med_err:.3f}) -- aborting")
    if covered < 0.9 * nnz:
        sys.exit(f"[{args.scene}] coverage {covered}/{nnz} < 90% -- aborting")
    os.makedirs(outdir, exist_ok=True)
    np.savez(out, obs_cam=np.concatenate(obs_cam), obs_pt=np.concatenate(obs_pt),
             F=np.concatenate(feats), source="b1_mast3r", proj_seed=0,
             frame_err=med_err)
    print(f"[{args.scene}] DONE D={feats[0].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
