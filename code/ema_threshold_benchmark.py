#!/usr/bin/env python
"""Real temporal test of EMA-smoothed pseudo-label thresholds (design 4b).

The static screen could not test EMA because it needs residuals that EVOLVE over
training. This walks the ARCHIVED checkpoints of a run in epoch order, and at
each epoch:
  * loads the checkpoint, runs the model on the pool's training scenes,
  * computes per-observation reprojection residuals of the CURRENT prediction,
  * derives the confident-band thresholds two ways
        plain : percentiles of this epoch's residuals  (what we ship)
        ema   : EMA over epochs, thr_t = a*thr_t + (1-a)*thr_{t-1}
  * scores both rules' pseudo-labels against the GT outlier labels.
Reports per-epoch P/R/F1 and the epoch-to-epoch threshold jitter each rule has.

Usage: ema_threshold_benchmark.py <run_dir> <conf> [--alpha 0.3] [--scenes ...]
"""
import argparse, glob, os, re, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyhocon import ConfigFactory
from datasets import SceneData
from models.SetOfSet import DeepSetOfSetOutliersNet
from utils import geo_utils

INLIER_P, OUTLIER_P = 20.0, 80.0


def residuals(model, data):
    with torch.no_grad():
        pred_cam, _ = model(data)
        Ps = pred_cam["Ps_norm"]
        p2d = Ps @ pred_cam["pts3D"]
        z = p2d[:, 2, :].clamp(min=1e-9)
        proj = p2d[:, 0:2, :] / z.unsqueeze(1)
        res = (proj - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)
        r = res[data.x.indices[0], data.x.indices[1]]
    return r.float().cpu().numpy()


def score(r, lo, hi, gt):
    po, pi = r > hi, r < lo
    conf = po | pi
    tp = np.sum(po & gt); fp = np.sum(po & ~gt); fn = np.sum(~po & gt & conf)
    p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return p, rc, 2 * p * rc / max(p + rc, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("conf")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--scenes", nargs="*", default=["NYC_Library", "Vienna_Cathedral"])
    args = ap.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    scenes = []
    for s in args.scenes:
        conf["dataset"]["scan"] = s
        d = SceneData.create_scene_data(conf).to(dev)
        gt = d.outlier_indices[d.x.indices[0], d.x.indices[1]].bool().cpu().numpy()
        scenes.append((s, d, gt))
        print(f"loaded {s}: {len(gt)} obs, GT contam {100*gt.mean():.1f}%", flush=True)

    ckpts = sorted(glob.glob(f"{args.run}/models_all/Model_Ep*.pt"),
                   key=lambda p: int(re.search(r"Ep(\d+)", p).group(1)))
    ema_lo = ema_hi = None
    rows = []
    for c in ckpts:
        ep = int(re.search(r"Ep(\d+)", c).group(1))
        model = DeepSetOfSetOutliersNet(conf).to(dev)
        model.load_state_dict(torch.load(c, map_location=dev)["model_state_dict"])
        model.eval()
        pl, el = [], []
        for s, d, gt in scenes:
            r = residuals(model, d)
            lo, hi = np.percentile(r, INLIER_P), np.percentile(r, OUTLIER_P)
            if ema_lo is None:
                e_lo, e_hi = lo, hi
            else:
                e_lo = args.alpha * lo + (1 - args.alpha) * ema_lo
                e_hi = args.alpha * hi + (1 - args.alpha) * ema_hi
            pl.append(score(r, lo, hi, gt)); el.append(score(r, e_lo, e_hi, gt))
            ema_lo, ema_hi = e_lo, e_hi
        p = np.mean(pl, 0); e = np.mean(el, 0)
        rows.append((ep, p[0], p[1], p[2], e[0], e[1], e[2], lo, hi, e_lo, e_hi))
        print(f"Ep{ep:<6d} plain P{p[0]:.3f} R{p[1]:.3f} F1{p[2]:.3f} | "
              f"ema P{e[0]:.3f} R{e[1]:.3f} F1{e[2]:.3f}", flush=True)
    a = np.array(rows)
    print(f"\n=== MEAN over {len(rows)} epochs ===")
    print(f"  plain  P {a[:,1].mean():.3f}  R {a[:,2].mean():.3f}  F1 {a[:,3].mean():.3f}")
    print(f"  ema    P {a[:,4].mean():.3f}  R {a[:,5].mean():.3f}  F1 {a[:,6].mean():.3f}")
    print(f"  threshold jitter (mean |d hi| between epochs): "
          f"plain {np.abs(np.diff(a[:,8])).mean():.4f}  ema {np.abs(np.diff(a[:,10])).mean():.4f}")
    print(f"  early epochs (first 10) F1: plain {a[:10,3].mean():.3f}  ema {a[:10,6].mean():.3f}")


if __name__ == "__main__":
    main()
