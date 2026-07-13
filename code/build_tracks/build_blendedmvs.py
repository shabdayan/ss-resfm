"""Build RESfM-format npz tracks for BlendedMVS candidate scenes.

RESfM Table 4 anonymizes its 4 scenes as scene0..3 with Nc = 75/51/33/66.
We build every local scene whose camera count matches one of those values
and disambiguate afterwards by outlier fraction (paper: 2.0/1.4/2.2/8.8%).

GT: cams/<id>_cam.txt in MVSNet format — `extrinsic` = 4x4 world-to-camera,
`intrinsic` = 3x3 at the low-res image resolution. Ps_gt = K @ E[:3, :].
Images: blended_images/<id>.jpg (the non-masked renders).

Usage: python build_blendedmvs.py [--scenes hash1 ...] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from appendix_c import build_scene

RAW = Path(__file__).resolve().parents[2] / "datasets/raw/blendedmvs/BlendedMVS"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "datasets/blendedmvs"
TABLE4_NC = (75, 51, 33, 66)


def parse_cam(path):
    """MVSNet cam.txt -> (E 4x4 world-to-cam, K 3x3)."""
    tokens = path.read_text().split()
    i_ext = tokens.index("extrinsic") + 1
    E = np.array(tokens[i_ext:i_ext + 16], np.float64).reshape(4, 4)
    i_int = tokens.index("intrinsic") + 1
    K = np.array(tokens[i_int:i_int + 9], np.float64).reshape(3, 3)
    return E, K


def load_gt(scene_dir):
    cams = sorted(scene_dir.glob("cams/*_cam.txt"))
    image_paths, Ps, Ks, names = [], [], [], []
    for cam in cams:
        stem = cam.name.replace("_cam.txt", "")
        img = scene_dir / "blended_images" / f"{stem}.jpg"
        if not img.exists():
            continue
        E, K = parse_cam(cam)
        image_paths.append(img)
        Ps.append(K @ E[:3, :])
        Ks.append(K)
        names.append(img.name)
    return image_paths, Ps, Ks, names


def candidate_scenes():
    out = []
    for scene_dir in sorted(RAW.iterdir()):
        n = len(list(scene_dir.glob("cams/*_cam.txt")))
        if n in TABLE4_NC:
            out.append(scene_dir.name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    scenes = args.scenes or candidate_scenes()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        print(f"[blendedmvs] {scene}", flush=True)
        image_paths, Ps, Ks, names = load_gt(RAW / scene)
        print(f"  {len(image_paths)} images", flush=True)
        data = build_scene(image_paths, Ps, Ks, names)
        out = args.out / f"{scene}.npz"
        np.savez(out, **data)
        print(f"  wrote {out} (M {data['M'].shape}, "
              f"outliers {data['outlier_pct']:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
