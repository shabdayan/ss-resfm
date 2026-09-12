#!/usr/bin/env python
"""RUC-SfM evaluator: paradigm-agnostic scoring of predicted camera poses.

Input: a predictions directory with one npz per scene:
    <scene>.npz: R [n_reg,3,3], t [n_reg,3] (world-to-cam),
                 registered [n_reg] (original camera indices)
GT: the benchmark suite's track npz directory (M, Ps_gt, K_gt/Ns).

Per scene it reports: similarity-aligned translation mean/median and rotation
mean over registered cameras; Nr/Nc; pairwise pose AUC@{1,5,30} and RA-AUC
(unregistered pairs = 180 deg). Across a 5-run submission, the harness reports
seed bands and the catastrophic-cell census (>30 / >100).

Usage: evaluate.py <pred_dir> <suite_npz_dir> [--out metrics.json]
Five-run submissions: evaluate.py <run_dir_glob> ... (one pred_dir per seed).
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from compute_vgpa_metrics import decompose_P, pair_errors, auc, TAUS  # noqa: E402


def eval_scene(pred, gt):
    Rg_all = np.zeros((gt["K_gt"].shape[0], 3, 3)); tg_all = np.zeros((gt["K_gt"].shape[0], 3))
    for i in range(len(Rg_all)):
        Rg_all[i], tg_all[i] = decompose_P(gt["Ps_gt"][i], gt["K_gt"][i])
    reg = np.asarray(pred["registered"]).astype(int)
    R, t = np.asarray(pred["R"]), np.asarray(pred["t"])
    Rg, tg = Rg_all[reg], tg_all[reg]
    nc = len(Rg_all); nr = len(reg)
    # aligned errors over registered cameras: the reference pipeline's own
    # alignment (chordal rotation fit + robust sum-of-norms scale/shift)
    from utils import geo_utils
    C = -np.einsum("iba,ib->ia", R, t)
    Cg = -np.einsum("iba,ib->ia", Rg, tg)
    R_fixed, C_fixed = geo_utils.align_cameras(R.astype(np.float64), Rg.astype(np.float64),
                                               C.astype(np.float64), Cg.astype(np.float64))
    terr = np.linalg.norm(Cg - C_fixed, axis=1)
    rerr = []
    for i in range(nr):
        c = np.clip((np.trace(Rg[i] @ np.asarray(R_fixed[i]).T) - 1) / 2, -1, 1)
        rerr.append(np.degrees(np.arccos(c)))
    rerr = np.array(rerr)
    # pairwise AUC / RA-AUC (alignment-free)
    rot, trn = pair_errors(R, t, Rg, tg)
    pose = np.maximum(rot, trn)
    pad = np.full(nc * (nc - 1) // 2 - len(pose), 180.0)
    out = {"nc": nc, "nr": nr,
           "trans_mean": float(terr.mean()), "trans_med": float(np.median(terr)),
           "rot_mean": float(rerr.mean()),
           "auc": {str(tau): auc(pose, tau) for tau in TAUS},
           "ra_auc": {str(tau): auc(np.concatenate([pose, pad]), tau) for tau in TAUS}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dirs", nargs="+", help="one directory per run/seed")
    ap.add_argument("suite", help="suite npz directory (GT tracks)")
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()
    runs = []
    for pd in args.pred_dirs:
        per = {}
        for f in sorted(glob.glob(os.path.join(pd, "*.npz"))):
            scene = os.path.basename(f)[:-4]
            gtf = os.path.join(args.suite, f"{scene}.npz")
            if not os.path.exists(gtf):
                continue
            per[scene] = eval_scene(np.load(f), np.load(gtf, allow_pickle=True))
        runs.append(per)
    scenes = sorted(set().union(*[set(r) for r in runs]))
    seed_means = [np.mean([r[s]["trans_mean"] for s in r]) for r in runs if r]
    cells = [r[s]["trans_mean"] for r in runs for s in r]
    summary = {
        "runs": len(runs), "scenes": len(scenes),
        "trans_mean_band": [float(np.mean(seed_means)),
                            float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None],
        "banded": len(runs) >= 5,
        "catastrophic_cells_gt30": int(sum(c > 30 for c in cells)),
        "catastrophic_cells_gt100": int(sum(c > 100 for c in cells)),
        "ra_auc30_band": [float(np.mean([np.mean([r[s]["ra_auc"]["30.0"] for s in r]) for r in runs])),
                          float(np.std([np.mean([r[s]["ra_auc"]["30.0"] for s in r]) for r in runs], ddof=1)) if len(runs) > 1 else None],
        "per_run": runs}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"runs {len(runs)} scenes {len(scenes)} banded={summary['banded']} "
          f"trans {summary['trans_mean_band']} RA-AUC30 {summary['ra_auc30_band']} "
          f"cat>30 {summary['catastrophic_cells_gt30']} (>100 {summary['catastrophic_cells_gt100']})")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
