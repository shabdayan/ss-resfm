#!/usr/bin/env python
"""Rebuild MegaDepth point tracks from RAW images with the canonical Appendix-C
pipeline (appendix_c.build_scene — the same entry point as our OOD builders),
for the track-pipeline robustness study.

Camera set, order, GT poses, and intrinsics are inherited from the existing
npz (namesList / Ps_gt / K_gt), so Group-2 subsampling and GT alignment are
preserved and results remain directly comparable.

Usage: python build_megadepth_rebuilt.py <scene> [--raw datasets/raw_megadepth]
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from appendix_c import build_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--raw", default=os.path.join(CODE, "datasets", "raw_megadepth"))
    args = ap.parse_args()

    out_path = os.path.join(CODE, "datasets", "megadepth_rebuilt", f"{args.scene}.npz")
    if os.path.exists(out_path):
        print(f"[{args.scene}] already built"); return

    ref = dict(np.load(os.path.join(CODE, "datasets", "megadepth", f"{args.scene}.npz"),
                       allow_pickle=True))
    names = [str(n) for n in ref["namesList"]]

    pool = {}
    for p in glob.glob(os.path.join(args.raw, "MegaDepth_v1", args.scene,
                                    "dense*", "imgs", "*")):
        pool[os.path.basename(p)] = p
    paths = []
    for n in names:
        base = os.path.basename(n)
        if base not in pool:
            sys.exit(f"[{args.scene}] missing image {base} ({len(pool)} extracted)")
        paths.append(pool[base])
    print(f"[{args.scene}] {len(paths)} cameras, all images found", flush=True)

    fields = build_scene(paths, ref["Ps_gt"], ref["K_gt"], ref["namesList"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **fields)
    print(f"[{args.scene}] DONE outlier_pct={float(fields['outlier_pct']):.2f}", flush=True)


if __name__ == "__main__":
    main()
