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
from colmap_baseline import load_scene, build_database, add_matches, align_and_error

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
            nreg, rot, trans = align_and_error(rec, Ps, K, m)
            print(f"  GLOMAP registered {len(rec.images)}/{m} cams", flush=True)
            res = dict(dataset=args.dataset, scene=args.scene, ncams=m,
                       registered=int(len(rec.images)), aligned=int(nreg), failed=False)
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
