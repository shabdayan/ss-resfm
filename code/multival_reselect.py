#!/usr/bin/env python
"""Post-hoc multi-domain checkpoint re-selection.

Sweeps a training run's saved models_all/Model_Ep*.pt checkpoints over a fixed
multi-domain validation set and re-selects the best epoch by the LABEL-FREE
reprojection metric (our_repro): forward pass -> predicted cameras + points ->
mean reprojection error vs the observed tracks. No BA, no GT, no outlier
labels -- the same selection signal the training protocol uses (RESfM's
released config validates on our_repro too), so it applies to supervised and
self-supervised models identically.

Val set: scenes unseen by EVERY run in the study (so one selection is
comparable across MD-only and multids models): the 4 MegaDepth val scenes
(val-only during training), ETH3D meadow (excluded from all pools), and 4
Olsson scenes (in no pool, not an OOD test benchmark).

Usage:
  python multival_reselect.py --conf confs/<train>.conf \
      --train_results results/multiscene/<resdir> [--stride 1000] [--out CSV]
"""
import argparse, glob, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VAL_SCENES = [
    ("megadepth", "5015"), ("megadepth", "0176"), ("megadepth", "0299"), ("megadepth", "0290"), ("megadepth", "0205"),
    ("eth3d", "meadow"),
    ("olsson", "Alcatraz_Courtyard"), ("olsson", "Door_Lund"),
    ("olsson", "Porta_San_Donato_Bologna"), ("olsson", "Round_Church_Cambridge"),
]



RICH_EXTRA = [   # legitimate ONLY for models that never trained on these (MD-only runs)
    ("eth3d", "courtyard"), ("eth3d", "kicker"), ("eth3d", "relief"), ("eth3d", "terrace"),
    ("vgg", "dino"), ("vgg", "corridor"), ("vgg", "wadham"),
    ("tnt", "ignatius"), ("tnt", "truck"), ("tnt", "meetingroom"),
]

def load_ckpt_state(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("state_dict", ck if isinstance(ck, dict) else None))
    if sd is None or not all(hasattr(v, "shape") for v in sd.values()):
        sd = ck["model_state_dict"]
    # strip DDP / fabric prefixes
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "_forward_module."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--train_results", required=True)
    ap.add_argument("--stride", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rich", action="store_true", help="add ETH3D/VGG/TNT scenes (MD-only models only)")
    args = ap.parse_args()

    from pyhocon import ConfigFactory
    from utils import general_utils, geo_utils
    from utils.Phases import Phases
    from datasets import SceneData

    conf = ConfigFactory.parse_file(os.path.join(HERE, args.conf))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = conf.get_string("model.type")
    model_class = general_utils.get_class("models." + model_type)
    if model_type in ["SetOfSet.SetOfSetOutliersNet", "SetOfSet.DeepSetOfSetOutliersNet"]:
        model = model_class(conf, Phases.TRAINING).to(device)
    else:
        model = model_class(conf).to(device)
    model.eval()

    # load val scenes once
    val_list = VAL_SCENES + (RICH_EXTRA if args.rich else [])
    scenes = []
    for fam, scan in val_list:
        c2 = ConfigFactory.parse_file(os.path.join(HERE, args.conf))
        c2.put("dataset.dataset", fam)
        c2.put("dataset.scan", scan)
        try:
            data = SceneData.create_scene_data(c2).to(device)
            scenes.append((f"{fam}/{scan}", data))
        except Exception as e:
            print(f"SKIP {fam}/{scan}: {e}", flush=True)
    print(f"loaded {len(scenes)} val scenes", flush=True)

    ckpts = sorted(glob.glob(os.path.join(HERE, args.train_results, "models_all", "Model_Ep*.pt")),
                   key=lambda p: int(re.search(r"Ep(\d+)", p).group(1)))
    ckpts = [p for p in ckpts if int(re.search(r"Ep(\d+)", p).group(1)) % args.stride == 0
             or p == ckpts[-1]]
    print(f"sweeping {len(ckpts)} checkpoints (stride {args.stride})", flush=True)

    rows = []
    for p in ckpts:
        ep = int(re.search(r"Ep(\d+)", p).group(1))
        model.load_state_dict(load_ckpt_state(p))
        vals = {}
        with torch.no_grad():
            for name, data in scenes:
                pred = model(data)
                if isinstance(pred, tuple):      # OutliersNet forwards return (pred_cam, outliers)
                    pred = pred[0]
                Ns_inv = data.Ns_invT.transpose(1, 2).cpu().numpy()
                Ps = Ns_inv @ pred["Ps_norm"].cpu().numpy()
                pts = geo_utils.pflat(pred["pts3D"]).cpu().numpy()
                xs = geo_utils.M_to_xs(data.M).cpu().numpy()
                vals[name] = float(np.nanmean(
                    geo_utils.reprojection_error_with_points(Ps, pts.T, xs)))
        mean_v = float(np.mean(list(vals.values())))
        med_v = float(np.median(list(vals.values())))
        rows.append(dict(epoch=ep, mean=mean_v, median=med_v, **vals))
        print(f"Ep{ep}: mean {mean_v:.3f}  median {med_v:.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    suffix = "_rich" if args.rich else ""
    out = args.out or os.path.join(HERE, args.train_results, f"multival_reselect{suffix}.csv")
    df.to_csv(out, index=False)
    bm = df.loc[df["mean"].idxmin()]; bd = df.loc[df["median"].idxmin()]
    print(f"\nBEST by mean:   Ep{int(bm['epoch'])} ({bm['mean']:.3f})")
    print(f"BEST by median: Ep{int(bd['epoch'])} ({bd['median']:.3f})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
