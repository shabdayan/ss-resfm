#!/usr/bin/env python
"""Lens-4 metrics (AUC@1/5/30 and registration-aware RA-AUC@1/5/30 for rotation,
translation direction, and pose) recovered from the kept Final_recon COLMAP
models — no re-evaluation needed.

Per scene dir:
  1. load colmap_reconstructions/Final_recon (post-BA poses, world->cam; proven
     exact vs the pruned Final_Cameras npz, transposed convention);
  2. identify each recon image's original camera: exact K-quadruple match vs
     K_gt, falling back to full-LUT 2D-coordinate voting against M (needed where
     intrinsics are shared within a scene);
  3. pairwise relative-pose errors vs GT (invariant to the global similarity):
     R_ij = R_j R_i^T, t_ij = t_j - R_ij t_i (direction only);
  4. AUC@tau = mean over registered pairs of (tau - e)^+ / tau;
     RA-AUC@tau = same over ALL Nc-choose-2 pairs, unregistered pairs = 180deg.

Usage: compute_vgpa_metrics.py <eval_root> <npz_dir> [--out out.json]
Emits one json: {scene: {nr, nc, auc: {rot/trans/pose x tau}, ra_auc: {...},
ident: {kmatch, vote, fail}}}
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
import pycolmap

TAUS = [1.0, 5.0, 30.0]


def decompose_P(P, K):
    Rt = np.linalg.inv(K) @ P
    U, S, Vt = np.linalg.svd(Rt[:, :3])
    lam = S.mean()
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R, lam = -R, -lam
    return R, Rt[:, 3] / lam


def cam_pose(im):
    try:
        cfw = im.cam_from_world()
    except TypeError:
        cfw = im.cam_from_world
    return np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation)


def identify(rec, Kgt, M):
    """recon image name -> original camera index. Returns (map, stats)."""
    m = Kgt.shape[0]
    quad = np.stack([Kgt[:, 0, 0], Kgt[:, 1, 1], Kgt[:, 0, 2], Kgt[:, 1, 2]], 1)
    lut = defaultdict(set)
    built = False
    out, stats = {}, {"kmatch": 0, "vote": 0, "fail": 0}
    for im in rec.images.values():
        p = np.asarray(rec.cameras[im.camera_id].params)[:4]
        d = np.abs(quad - p[None]).max(1)
        cand = np.where(d < 0.5)[0]
        if len(cand) == 1:
            out[im.name] = int(cand[0]); stats["kmatch"] += 1
            continue
        if not built:                       # full LUT, no truncation
            for i in range(m):
                xs, ys = M[2 * i], M[2 * i + 1]
                vis = (xs != 0) | (ys != 0)
                for x, y in zip(xs[vis], ys[vis]):
                    lut[(round(float(x), 1), round(float(y), 1))].add(i)
            built = True
        votes = defaultdict(int)
        for pt in im.points2D:
            x, y = pt.xy
            cs = lut.get((round(float(x), 1), round(float(y), 1)), ())
            for c in (cs if len(cand) == 0 else [c for c in cs if c in set(cand)]):
                votes[c] += 1
        if votes:
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            if len(ranked) == 1 or ranked[0][1] >= 2 * max(1, ranked[1][1]):
                out[im.name] = int(ranked[0][0]); stats["vote"] += 1
                continue
        stats["fail"] += 1
    return out, stats


def pair_errors(R, t, Rg, tg):
    """Vectorized pairwise relative-pose errors (upper triangle)."""
    R = np.asarray(R); t = np.asarray(t)
    Rg = np.asarray(Rg); tg = np.asarray(tg)
    n = len(R)
    iu, ju = np.triu_indices(n, 1)
    # rot: angle of R_ij G_ij^T with R_ij=R_j R_i^T, G_ij=Rg_j Rg_i^T;
    # trace = einsum over G_j = Rg_j^T R_j and A_i = R_i^T Rg_i
    A = np.einsum("iba,ibc->iac", R, Rg)          # R_i^T Rg_i
    G = np.einsum("jba,jbc->jac", Rg, R)          # Rg_j^T R_j
    tr = np.einsum("jab,iba->ij", G, A)
    c = np.clip((tr[iu, ju] - 1) / 2, -1, 1)      # symmetric in i,j order
    rot = np.degrees(np.arccos(c))
    # trans direction: t_ij = R_j (C_i - C_j) with C = -R^T t
    C = -np.einsum("iba,ib->ia", R, t)
    Cg = -np.einsum("iba,ib->ia", Rg, tg)
    V = C[iu] - C[ju]; Vg = Cg[iu] - Cg[ju]
    tij = np.einsum("jab,jb->ja", R[ju], V)
    gij = np.einsum("jab,jb->ja", Rg[ju], Vg)
    nt = np.linalg.norm(tij, axis=1); ng = np.linalg.norm(gij, axis=1)
    ok = (nt > 1e-9) & (ng > 1e-9)
    cc = np.zeros(len(iu))
    cc[ok] = np.abs(np.einsum("ja,ja->j", tij[ok], gij[ok])) / (nt[ok] * ng[ok])
    trn = np.degrees(np.arccos(np.clip(cc, -1, 1)))
    trn[~ok] = 0.0
    return rot, trn


def auc(errs, tau):
    if len(errs) == 0:
        return 0.0
    return float(np.mean(np.clip(tau - errs, 0, None) / tau))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root"); ap.add_argument("npz_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for sd in sorted(glob.glob(os.path.join(args.eval_root, "*_ba"))):
        scene = os.path.basename(sd)[:-3]
        recdir = os.path.join(sd, "colmap_reconstructions", "Final_recon")
        npz = os.path.join(args.npz_dir, f"{scene}.npz")
        if not os.path.isdir(recdir) or not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        Kgt = d["K_gt"]; M = d["M"]; nc = Kgt.shape[0]
        Rg = np.zeros((nc, 3, 3)); tg = np.zeros((nc, 3))
        for i in range(nc):
            Rg[i], tg[i] = decompose_P(d["Ps_gt"][i], Kgt[i])
        try:
            rec = pycolmap.Reconstruction(recdir)
        except Exception as e:
            print(f"[{scene}] unreadable recon: {e}", flush=True); continue
        ident, stats = identify(rec, Kgt, M)
        cams = {}
        for im in rec.images.values():
            if im.name in ident:
                cams[ident[im.name]] = cam_pose(im)
        idx = sorted(cams)
        R = [cams[i][0] for i in idx]; t = [cams[i][1] for i in idx]
        rot, trn = pair_errors(R, t, [Rg[i] for i in idx], [tg[i] for i in idx])
        pose = np.maximum(rot, trn)
        n_all = nc * (nc - 1) // 2
        n_reg = len(rot)
        pad = np.full(n_all - n_reg, 180.0)
        entry = {"nr": len(idx), "nc": nc, "ident": stats,
                 "auc": {}, "ra_auc": {}}
        for nm, e in [("rot", rot), ("trans", trn), ("pose", pose)]:
            entry["auc"][nm] = {str(tau): auc(e, tau) for tau in TAUS}
            entry["ra_auc"][nm] = {str(tau): auc(np.concatenate([e, pad]), tau)
                                   for tau in TAUS}
        res[scene] = entry
        print(f"[{scene}] nr {len(idx)}/{nc} ident {stats} "
              f"RA-pose@30 {entry['ra_auc']['pose']['30.0']:.3f}", flush=True)
    out = args.out or os.path.join(args.eval_root, "vgpa_metrics.json")
    json.dump(res, open(out, "w"), indent=1)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
