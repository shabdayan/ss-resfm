#!/usr/bin/env python
"""Run COLMAP (via pycolmap) as an SfM baseline on OUR point tracks.

We feed COLMAP the *same* correspondences our network sees: each scene npz's
track matrix M becomes per-image keypoints + two-view matches (from tracks
co-observed by camera pairs), with GT intrinsics K_gt. COLMAP then runs its
own incremental mapping + BA. We align the result to GT (Umeyama on camera
centers) and report rotation/translation error and #registered cameras --
matching the "classical run on our tracks" protocol of RESfM.

Usage: python colmap_baseline.py <dataset> <scene> [--min_pair 15]
       (dataset dir = code/datasets/<dataset>/<scene>.npz)
"""
import argparse, os, tempfile, shutil, itertools
import numpy as np
import pycolmap


def load_scene(dataset, scene):
    p = os.path.join(os.path.dirname(__file__), "datasets", dataset, f"{scene}.npz")
    d = np.load(p, allow_pickle=True)
    M = d["M"]; K = d["K_gt"]; Ps = d["Ps_gt"]
    m = M.shape[0] // 2
    X = M.reshape(m, 2, -1)                      # (m,2,n) pixel coords, 0=unobserved
    vis = (X[:, 0] != 0) | (X[:, 1] != 0)        # (m,n)
    return X, vis, K, Ps, m


def build_database(db_path, X, vis, K, m):
    db = pycolmap.Database(db_path)
    # per image: keypoints for its observed tracks + track->kp-row map
    kp_row = [dict() for _ in range(m)]          # track_col -> kp index in image i
    img_ids = []
    for i in range(m):
        cols = np.flatnonzero(vis[i])
        kps = X[i, :, cols].astype(np.float64)   # (Ni,2)
        fx, fy, cx, cy = K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]
        w, h = int(2 * cx + 1), int(2 * cy + 1)
        cam = pycolmap.Camera(model="PINHOLE", width=w, height=h,
                              params=[fx, fy, cx, cy])
        cam_id = db.write_camera(cam)
        img = pycolmap.Image(name=f"{i:06d}.jpg", camera_id=cam_id)
        img_id = db.write_image(img)
        img_ids.append(img_id)
        db.write_keypoints(img_id, kps.astype(np.float32))
        for k, c in enumerate(cols):
            kp_row[i][c] = k
    return db, img_ids, kp_row


def add_matches(db, X, vis, K, m, img_ids, kp_row, min_pair):
    # tracks observed per camera pair -> matches; verify geometry per pair
    n = vis.shape[1]
    # for each track, the observing cameras
    npairs = 0
    # accumulate matches per (i,j)
    from collections import defaultdict
    pair_m = defaultdict(list)
    for c in range(n):
        cams = np.flatnonzero(vis[:, c])
        for i, j in itertools.combinations(cams, 2):
            pair_m[(i, j)].append((kp_row[i][c], kp_row[j][c]))
    for (i, j), mm in pair_m.items():
        if len(mm) < min_pair:
            continue
        matches = np.array(mm, dtype=np.uint32)
        db.write_matches(img_ids[i], img_ids[j], matches)
        # calibrated two-view geometry from the matched keypoints
        ci = pycolmap.Camera(model="PINHOLE", width=int(2*K[i,0,2]+1), height=int(2*K[i,1,2]+1),
                             params=[K[i,0,0],K[i,1,1],K[i,0,2],K[i,1,2]])
        cj = pycolmap.Camera(model="PINHOLE", width=int(2*K[j,0,2]+1), height=int(2*K[j,1,2]+1),
                             params=[K[j,0,0],K[j,1,1],K[j,0,2],K[j,1,2]])
        ki = X[i, :, np.flatnonzero(vis[i])].astype(np.float64)
        kj = X[j, :, np.flatnonzero(vis[j])].astype(np.float64)
        try:
            tvg = pycolmap.estimate_calibrated_two_view_geometry(ci, ki, cj, kj, matches)
        except Exception:
            tvg = None
        if tvg is not None:
            db.write_two_view_geometry(img_ids[i], img_ids[j], tvg)
            npairs += 1
    return npairs


def align_and_error(rec, Ps_gt, K, m):
    # GT: Ps_gt = K[R|t], so remove K first: [R|t] = K^-1 @ P
    def gt_pose(P, Kmat):
        RT = np.linalg.inv(Kmat) @ P
        R = RT[:, :3]; t = RT[:, 3]
        U, _, Vt = np.linalg.svd(R); R = U @ Vt   # re-orthonormalize
        return -R.T @ t, R
    gt_c = {}; gt_R = {}
    for i in range(m):
        c, R = gt_pose(Ps_gt[i], K[i]); gt_c[i] = c; gt_R[i] = R
    est_c = {}; est_R = {}
    for img in rec.images.values():
        i = int(img.name.split(".")[0])
        Rt = img.cam_from_world.matrix()   # 3x4 world->cam
        R = Rt[:, :3]; t = Rt[:, 3]
        est_c[i] = -R.T @ t; est_R[i] = R
    common = sorted(set(gt_c) & set(est_c))
    if len(common) < 3:
        return len(common), None, None
    A = np.array([est_c[i] for i in common]); B = np.array([gt_c[i] for i in common])
    # Umeyama similarity A->B
    muA, muB = A.mean(0), B.mean(0)
    AA, BB = A - muA, B - muB
    U, S, Vt = np.linalg.svd((BB.T @ AA) / len(common))
    d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
    Rsim = U @ D @ Vt
    scale = S.sum() / (AA ** 2).sum() * len(common)
    trans_err = []; rot_err = []
    for i in common:
        c_al = scale * Rsim @ est_c[i] + (muB - scale * Rsim @ muA)
        trans_err.append(np.linalg.norm(c_al - gt_c[i]))
        R_al = est_R[i] @ Rsim.T
        cosv = (np.trace(R_al @ gt_R[i].T) - 1) / 2
        rot_err.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    return len(common), np.array(rot_err), np.array(trans_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("scene")
    ap.add_argument("--min_pair", type=int, default=15)
    args = ap.parse_args()
    X, vis, K, Ps, m = load_scene(args.dataset, args.scene)
    print(f"[{args.dataset}/{args.scene}] {m} cams, {vis.shape[1]} tracks, "
          f"{int(vis.sum())} obs", flush=True)
    work = tempfile.mkdtemp(prefix="colmap_")
    try:
        db_path = os.path.join(work, "database.db")
        db, img_ids, kp_row = build_database(db_path, X, vis, K, m)
        npairs = add_matches(db, X, vis, K, m, img_ids, kp_row, args.min_pair)
        db.close()
        print(f"  wrote DB: {m} images, {npairs} verified pairs", flush=True)
        out = os.path.join(work, "sparse"); os.makedirs(out, exist_ok=True)
        os.makedirs(os.path.join(work, "images"), exist_ok=True)
        recs = pycolmap.incremental_mapping(db_path, os.path.join(work, "images"), out)
        if not recs:
            print("  COLMAP produced NO reconstruction"); return
        rec = max(recs.values(), key=lambda r: len(r.images))
        nreg, rot, trans = align_and_error(rec, Ps, K, m)
        print(f"  COLMAP registered {len(rec.images)}/{m} cams", flush=True)
        if rot is not None:
            print(f"  ROT  mean={rot.mean():.3f} median={np.median(rot):.3f}")
            print(f"  TRANS mean={trans.mean():.3f} median={np.median(trans):.3f}  "
                  f"(aligned on {nreg} common cams)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
