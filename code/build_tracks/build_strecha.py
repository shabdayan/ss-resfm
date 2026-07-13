"""Build RESfM-format npz tracks for the 4 Strecha scenes of RESfM Table 3.

GT: each image ships a `.P` file (3x4 projection, pixels) and a `.camera`
file whose first 3 rows are K. Ns = K^-1, Ps_gt = the released P.

Usage: python build_strecha.py [--scenes entry-P10 ...] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from appendix_c import build_scene

RAW = Path(__file__).resolve().parents[2] / "datasets/raw/strecha"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "datasets/strecha"
SCENES = ["entry-P10", "fountain-P11", "Herz-Jesu-P8", "Herz-Jesu-P25"]


def load_gt(scene_dir):
    """Return (image_paths, Ps, Ks, names) sorted by image name."""
    image_paths = sorted(scene_dir.glob("*.png"))
    Ps, Ks, names = [], [], []
    for img in image_paths:
        P = np.loadtxt(str(img) + ".P")
        # .camera rows: K (3x3), distortion (3), R (3x3), C (3), size (2) —
        # ragged, so parse the token stream instead of np.loadtxt.
        tokens = np.array(Path(str(img) + ".camera").read_text().split(),
                          np.float64)
        K = tokens[0:9].reshape(3, 3)
        assert P.shape == (3, 4) and K.shape == (3, 3)
        Ps.append(P)
        Ks.append(K)
        names.append(img.name)
    return image_paths, Ps, Ks, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        print(f"[strecha] {scene}", flush=True)
        image_paths, Ps, Ks, names = load_gt(RAW / scene)
        data = build_scene(image_paths, Ps, Ks, names)
        out = args.out / f"{scene}.npz"
        np.savez(out, **data)
        print(f"  wrote {out} (M {data['M'].shape}, "
              f"outliers {data['outlier_pct']:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
