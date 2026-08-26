"""Build RESfM-format npz tracks for ETH3D high-res multi-view scenes.

GT ships as a COLMAP model (dslr_calibration_undistorted/): cameras.txt gives
PINHOLE intrinsics, images.txt gives per-image world->cam pose (quaternion +
translation). We rebuild tracks from scratch with the SAME SIFT+FLANN pipeline
used for Strecha (appendix_c.build_scene) so contamination reflects realistic
matcher noise, then label outliers by robust triangulation under the GT poses.
The GT model's own 2D points (2nd line per image in images.txt) are ignored.

Usage: python build_eth3d.py [--scenes pipes ...] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from appendix_c import build_scene

RAW = Path(__file__).resolve().parents[2] / "datasets/raw/eth3d"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "datasets/eth3d"
SCENES = ["pipes"]


def quat_to_R(qw, qx, qy, qz):
    """COLMAP Hamilton quaternion (w,x,y,z), world->cam rotation."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_gt(scene_dir):
    """Return (image_paths, Ps, Ks, names) sorted by image filename."""
    calib = scene_dir / "dslr_calibration_undistorted"
    img_root = scene_dir / "images"
    # cameras.txt: CAMERA_ID PINHOLE W H fx fy cx cy
    cam_K = {}
    for line in (calib / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cid = int(t[0]); model = t[1]
        assert model == "PINHOLE", f"unexpected camera model {model}"
        fx, fy, cx, cy = map(float, t[4:8])
        cam_K[cid] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    # images.txt: pose line then a points line; take only the pose lines.
    rows = []
    lines = [l for l in (calib / "images.txt").read_text().splitlines()
             if not l.startswith("#")]
    for i in range(0, len(lines), 2):          # every other line = a pose line
        t = lines[i].split()
        if len(t) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, t[1:8])
        cid = int(t[8]); name = t[9]
        R = quat_to_R(qw, qx, qy, qz); tvec = np.array([tx, ty, tz])
        K = cam_K[cid]
        P = K @ np.hstack([R, tvec.reshape(3, 1)])   # K[R|t], pixels
        img_path = img_root / name
        if not img_path.exists():                    # name may include subdir
            img_path = img_root / Path(name).name
        rows.append((img_path.name, img_path, P, K))
    rows.sort(key=lambda r: r[0])
    names = [r[0] for r in rows]
    image_paths = [r[1] for r in rows]
    Ps = [r[2] for r in rows]
    Ks = [r[3] for r in rows]
    return image_paths, Ps, Ks, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        print(f"[eth3d] {scene}", flush=True)
        image_paths, Ps, Ks, names = load_gt(RAW / scene)
        print(f"  {len(image_paths)} images, K[0] fx={Ks[0][0,0]:.1f}", flush=True)
        data = build_scene(image_paths, Ps, Ks, names)
        out = args.out / f"{scene}.npz"
        np.savez(out, **data)
        print(f"  wrote {out} (M {data['M'].shape}, "
              f"outliers {data['outlier_pct']:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
