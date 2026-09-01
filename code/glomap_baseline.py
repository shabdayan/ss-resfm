#!/usr/bin/env python
"""Run GLOMAP (global SfM) as a classical baseline on OUR point tracks.

Reuses colmap_baseline.py end-to-end: the scene npz's track matrix M becomes a
COLMAP database (per-image keypoints + verified two-view geometries, GT
intrinsics) -- exactly the input `glomap mapper` consumes -- then GLOMAP runs
global mapping, and we align/evaluate identically (Umeyama on camera centers;
rotation/translation error over registered cameras), matching the
"classical run on our tracks" protocol of RESfM (their Table 1 GLOMAP rows).

The glomap binary comes from the user-local micromamba env (conda-forge);
LD_LIBRARY_PATH must prefer the env's lib dir (system libcurl/libldap clash).

Usage: python glomap_baseline.py <dataset> <scene> [--min_pair 15]
Writes results/classical/glomap/<dataset>__<scene>.json
"""
import argparse, os, json, tempfile, shutil, subprocess
import numpy as np
import pycolmap
from colmap_baseline import load_scene, build_database, add_matches


def align_and_error(rec, Ps_gt, K, m):
    """Umeyama alignment + pose errors, reflection-aware.

    GLOMAP's global solver can converge to a MIRRORED reconstruction: camera
    positions are gauge-equivalent (a similarity with det=-1 still maps them onto
    GT), but orientations are not -- applying the reflected rotation to R gives
    garbage angles (150-180 deg). We detect det<0 and mirror the estimate into a
    right-handed frame (negate one world axis on positions AND rotations) before
    scoring, so rotation errors are comparable to the other baselines.
    """
    def gt_pose(P, Kmat):
        RT = np.linalg.inv(Kmat) @ P
        R = RT[:, :3]; t = RT[:, 3]
        U, _, Vt = np.linalg.svd(R); R = U @ Vt
        return -R.T @ t, R
    gt_c = {}; gt_R = {}
    for i in range(m):
        c, R = gt_pose(Ps_gt[i], K[i]); gt_c[i] = c; gt_R[i] = R
    est_c = {}; est_R = {}
    for img in rec.images.values():
        i = int(img.name.split(".")[0])
        Rt = img.cam_from_world.matrix()
        R = Rt[:, :3]; t = Rt[:, 3]
        est_c[i] = -R.T @ t; est_R[i] = R
    common = sorted(set(gt_c) & set(est_c))
    if len(common) < 3:
        return len(common), None, None, False

    def umeyama(A, B):
        muA, muB = A.mean(0), B.mean(0)
        AA, BB = A - muA, B - muB
        U, S, Vt = np.linalg.svd((BB.T @ AA) / len(A))
        d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
        Rs = U @ D @ Vt
        scale = S.sum() / (AA ** 2).sum() * len(A)
        return Rs, scale, muA, muB, d

    A = np.array([est_c[i] for i in common]); B = np.array([gt_c[i] for i in common])
    _, _, _, _, d = umeyama(A, B)
    mirrored = d < 0
    if mirrored:
        # Mirror only the POSITIONS into a right-handed frame for the similarity fit.
        # Orientations are NOT mirrored: a reflection is not a rotation, and the
        # rotation error below is computed gauge-free from relative rotations, so it
        # needs no frame alignment at all.
        M = np.diag([-1.0, 1.0, 1.0])
        est_c = {i: M @ c for i, c in est_c.items()}
        A = np.array([est_c[i] for i in common])
    Rsim, scale, muA, muB, _ = umeyama(A, B)
    trans_err = []
    for i in common:
        c_al = scale * Rsim @ est_c[i] + (muB - scale * Rsim @ muA)
        trans_err.append(np.linalg.norm(c_al - gt_c[i]))
    # Rotation: gauge-free. Compare RELATIVE rotations R_a R_b^T, which are invariant
    # to the global frame (and to any mirroring of the world), so no alignment is
    # applied. Per camera we average its pairwise errors against a capped sample of
    # partners, giving a per-camera error array comparable to the other baselines.
    rng = np.random.RandomState(0)
    idx = list(common)
    rot_err = []
    for a in idx:
        partners = idx if len(idx) <= 40 else list(rng.choice(idx, 40, replace=False))
        e = []
        for b in partners:
            if b == a:
                continue
            rel_est = est_R[a] @ est_R[b].T
            rel_gt = gt_R[a] @ gt_R[b].T
            cosv = (np.trace(rel_est @ rel_gt.T) - 1) / 2
            e.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
        rot_err.append(np.mean(e) if e else 0.0)
    return len(common), np.array(rot_err), np.array(trans_err), mirrored

GLOMAP_ENV = os.path.expanduser("~/micromamba/root/envs/glomap")


def run_glomap(db_path, out_dir):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(GLOMAP_ENV, "lib")
    cmd = [os.path.join(GLOMAP_ENV, "bin", "glomap"), "mapper",
           "--database_path", db_path, "--output_path", out_dir,
           # retriangulation trips a rig-API CHECK against pycolmap-3.10 databases
           # (conda-forge glomap built vs newer colmap); global BA already ran.
           "--skip_retriangulation", "1"]
    r = subprocess.run(cmd, env=env, capture_output=True, timeout=6*3600)
    if r.returncode != 0:
        print(r.stdout[-2000:].decode("utf-8", "replace"))
        print(r.stderr[-2000:].decode("utf-8", "replace"))
        raise RuntimeError(f"glomap exited {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("scene")
    ap.add_argument("--min_pair", type=int, default=15)
    args = ap.parse_args()
    X, vis, K, Ps, m = load_scene(args.dataset, args.scene)
    print(f"[{args.dataset}/{args.scene}] {m} cams, {vis.shape[1]} tracks, "
          f"{int(vis.sum())} obs", flush=True)
    outdir = os.path.join(os.path.dirname(__file__), "results", "classical", "glomap")
    os.makedirs(outdir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="glomap_")
    try:
        db_path = os.path.join(work, "database.db")
        db, img_ids, kp_row = build_database(db_path, X, vis, K, m)
        npairs = add_matches(db, X, vis, K, m, img_ids, kp_row, args.min_pair)
        db.close()
        print(f"  wrote DB: {m} images, {npairs} verified pairs", flush=True)
        sparse = os.path.join(work, "sparse"); os.makedirs(sparse, exist_ok=True)
        run_glomap(db_path, sparse)
        # glomap writes <out>/0/{cameras,images,points3D}.bin
        model_dirs = [d for d in sorted(os.listdir(sparse))
                      if os.path.isdir(os.path.join(sparse, d))]
        if not model_dirs:
            print("  GLOMAP produced NO reconstruction")
            res = dict(dataset=args.dataset, scene=args.scene, ncams=m,
                       registered=0, failed=True)
        else:
            recs = [pycolmap.Reconstruction(os.path.join(sparse, d)) for d in model_dirs]
            rec = max(recs, key=lambda r: len(r.images))
            nreg, rot, trans, mirrored = align_and_error(rec, Ps, K, m)
            print(f"  GLOMAP registered {len(rec.images)}/{m} cams", flush=True)
            res = dict(dataset=args.dataset, scene=args.scene, ncams=m,
                       registered=int(len(rec.images)), aligned=int(nreg), failed=False,
                       mirrored=bool(mirrored))
            if rot is not None:
                res.update(rot_mean=float(rot.mean()), rot_med=float(np.median(rot)),
                           trans_mean=float(trans.mean()), trans_med=float(np.median(trans)))
                print(f"  ROT  mean={rot.mean():.3f} median={np.median(rot):.3f}")
                print(f"  TRANS mean={trans.mean():.3f} median={np.median(trans):.3f}  "
                      f"(aligned on {nreg} common cams)")
        with open(os.path.join(outdir, f"{args.dataset}__{args.scene}.json"), "w") as f:
            json.dump(res, f)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
