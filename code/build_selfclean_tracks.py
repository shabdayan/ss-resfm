#!/usr/bin/env python
"""Self-cleaned track builder: remove outlier observations from the MegaDepth
TRAINING (and validation) scenes using the outlier head of a trained
RESfM-faithful checkpoint (no GT labels in the loop), mirroring the ESFM*
builder but with PREDICTED labels instead of 'outliers2'.

For each scene: forward pass of the checkpoint -> per-observation outlier
probability -> map back onto the (C,n) grid via get_M_valid_points (same
mapping as evaluation.prepare_outliers_predictions) -> zero observations with
p > --thresh -> drop tracks with <2 remaining views -> save to
datasets/<outname>/<scene>.npz. GT fields (outliers2 etc.) are subset to the
kept tracks so a supervised retrain on the cleaned data still has its labels.

The raw npz M and the SceneData M must match exactly (asserted); otherwise the
prediction grid would not align with the file being cleaned.

Usage (GPU node):
  python build_selfclean_tracks.py --conf confs/multiscene_resfm_shallow_faithful.conf \
      --ckpt results/multiscene/resfm_shallow_27scenes_faithful/models/Model_Ep5500.pt \
      --outname megadepth_selfclean [--thresh 0.5]
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_ckpt_state(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("state_dict", ck if isinstance(ck, dict) else None))
    if sd is None or not all(hasattr(v, "shape") for v in sd.values()):
        sd = ck["model_state_dict"]
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outname", default="megadepth_selfclean")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    from pyhocon import ConfigFactory
    from utils import general_utils, dataset_utils
    from utils.Phases import Phases
    from datasets import SceneData

    conf = ConfigFactory.parse_file(os.path.join(HERE, args.conf))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = general_utils.get_class("models." + conf.get_string("model.type"))
    model = model_class(conf, Phases.TRAINING).to(device)
    model.load_state_dict(load_ckpt_state(os.path.join(HERE, args.ckpt)))
    model.eval()

    scenes = list(conf.get_list("dataset.train_set")) + list(conf.get_list("dataset.validation_set"))
    outdir = os.path.join(HERE, "datasets", args.outname)
    os.makedirs(outdir, exist_ok=True)

    for scan in scenes:
        src = os.path.join(HERE, "datasets", "megadepth", f"{scan}.npz")
        d = dict(np.load(src, allow_pickle=True))
        c2 = ConfigFactory.parse_file(os.path.join(HERE, args.conf))
        c2.put("dataset.dataset", "megadepth")
        c2.put("dataset.scan", scan)
        data = SceneData.create_scene_data(c2).to(device)
        M_data = data.M.cpu().numpy()
        assert M_data.shape == d["M"].shape and np.allclose(M_data, d["M"]), \
            f"{scan}: SceneData M does not match raw npz M ({M_data.shape} vs {d['M'].shape})"

        with torch.no_grad():
            pred = model(data)
        pred_outliers = pred[1] if isinstance(pred, tuple) else pred
        valid_mask = dataset_utils.get_M_valid_points(data.M)
        prob = torch.zeros_like(valid_mask, dtype=torch.float32)
        prob[valid_mask] = pred_outliers.squeeze().float()
        out = (prob.cpu().numpy() > args.thresh) & valid_mask.cpu().numpy()

        M2 = d["M"].copy()
        M2[0::2][out] = 0.0
        M2[1::2][out] = 0.0
        vis2 = (M2[0::2] != 0) | (M2[1::2] != 0)
        keep = vis2.sum(axis=0) >= 2
        d["M"] = M2[:, keep]
        if "outliers2" in d:
            O = d["outliers2"].astype(bool)
            O2 = (O & ~out)[:, keep]                 # removed obs are gone; labels only on survivors
            d["outliers2"] = O2
            visk = vis2[:, keep]
            d["outlier_pct"] = np.float64(O2[visk].sum() / max(1, visk.sum()))
        np.savez(os.path.join(outdir, f"{scan}.npz"), **d)
        removed = out.sum() / max(1, valid_mask.sum().item())
        print(f"{scan}: obs removed {100*removed:.1f}%  tracks {M2.shape[1]}->{int(keep.sum())}"
              f"  new outlier_pct {100*float(d.get('outlier_pct', 0)):.2f}%", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
