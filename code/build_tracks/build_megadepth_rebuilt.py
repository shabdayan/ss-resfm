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
    for p in glob.glob(os.path.join(args.raw, "MegaDepth_SfM", args.scene, "*")):
        pool[os.path.basename(p)] = p
    # MegaDepth_SfM holds the full per-scene Flickr collections (the v1 dense
    # archive covered only ~43% of the npz cameras and was abandoned). Build
    # over the covered subset, subsetting GT accordingly, with a >=90% guard.
    keep, paths = [], []
    for idx, n in enumerate(names):
        base = os.path.basename(n.strip())  # namesList entries carry a trailing newline
        if base in pool:
            keep.append(idx); paths.append(pool[base])
    cov = len(keep) / max(1, len(names))
    print(f"[{args.scene}] {len(keep)}/{len(names)} cameras covered "
          f"({100*cov:.1f}%)", flush=True)
    if cov < 0.9:
        sys.exit(f"[{args.scene}] coverage {100*cov:.1f}% < 90% -- skipping")
    import numpy as _np
    keep = _np.asarray(keep)
    fields = build_scene(paths, ref["Ps_gt"][keep], ref["K_gt"][keep],
                         ref["namesList"][keep])
    fields["covered_frac"] = _np.float64(cov)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **fields)
    print(f"[{args.scene}] DONE outlier_pct={float(fields['outlier_pct']):.2f}", flush=True)


if __name__ == "__main__":
    main()
