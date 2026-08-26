"""Build RESfM-format npz tracks for the Oxford VGG multi-view scenes.

GT: per-image 3x4 projection matrices in pixel space — either one `<stem>.P`
file per image (corridor, dunster/house, merton.a/b/c, wadham) or a single
`dino_Ps.mat` cell array (dino, frames ordered like the sorted viff images).
K is recovered by RQ decomposition of P[:, :3] (K upper-triangular, positive
diagonal); Ns = K^-1. Tracks are rebuilt with the SAME SIFT+FLANN pipeline as
Strecha/ETH3D (appendix_c.build_scene) so contamination reflects realistic
matcher noise, labeled by robust triangulation under the GT cameras.

Usage: python build_vgg.py [--scenes dino corridor ...] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from appendix_c import build_scene

RAW = Path(__file__).resolve().parents[2] / "datasets/raw/vgg"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "datasets/vgg"
SCENES = ["dino", "corridor", "dunster", "merton.a", "merton.b", "merton.c", "wadham"]
IMG_EXTS = [".jpg", ".jpeg", ".png", ".pgm", ".ppm"]


def rq_K(P):
    """K from P=K[R|t] via RQ decomposition; K scaled to K[2,2]=1 with a
    positive diagonal (sign flips absorbed into R, which we discard)."""
    from scipy.linalg import rq
    K, R = rq(P[:, :3])
    D = np.diag(np.sign(np.diag(K)))       # make diag(K) positive
    K = K @ D
    return K / K[2, 2]


def load_gt(scene_dir, scene):
    """Return (image_paths, Ps, Ks, names) sorted by image name."""
    rows = []
    if scene == "dino":
        import scipy.io as sio
        Pcell = sio.loadmat(str(scene_dir / "dino_Ps.mat"))["P"].flatten()
        imgs = sorted(scene_dir.glob("viff.*.ppm"))
        # dino is a closed 360-degree turntable: 37 frames but 36 cameras (the
        # last frame revisits the first pose; viff.xy is 36 frames wide too).
        # Pair the first len(P) frames with the Ps in order.
        assert len(imgs) >= len(Pcell), f"dino: {len(imgs)} imgs vs {len(Pcell)} Ps"
        if len(imgs) > len(Pcell):
            print(f"  dropping {len(imgs)-len(Pcell)} trailing frame(s) without P",
                  flush=True)
            imgs = imgs[:len(Pcell)]
        for img, P in zip(imgs, Pcell):
            rows.append((img.name, img, np.asarray(P, np.float64)))
    else:
        for pf in sorted(scene_dir.glob("*.P")):
            img = next((pf.with_suffix(e) for e in IMG_EXTS
                        if pf.with_suffix(e).exists()), None)
            if img is None:
                print(f"  WARNING: no image for {pf.name}", flush=True)
                continue
            rows.append((img.name, img, np.loadtxt(pf)))
    names = [r[0] for r in rows]
    image_paths = [r[1] for r in rows]
    Ps = [r[2] for r in rows]
    Ks = [rq_K(P) for P in Ps]
    return image_paths, Ps, Ks, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        print(f"[vgg] {scene}", flush=True)
        image_paths, Ps, Ks, names = load_gt(RAW / scene, scene)
        print(f"  {len(image_paths)} images, K[0] fx={Ks[0][0,0]:.1f}", flush=True)
        data = build_scene(image_paths, Ps, Ks, names)
        out = args.out / f"{scene}.npz"
        np.savez(out, **data)
        print(f"  wrote {out} (M {data['M'].shape}, "
              f"outliers {data['outlier_pct']:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
