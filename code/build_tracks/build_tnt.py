"""Build RESfM-format npz tracks for Tanks & Temples training-group scenes.

Source: the NeRF-community packaging (datasets/raw/tnt/tanksandtemples/<scene>/)
with a COLMAP sparse model (sparse/0/{cameras,images}.txt, PINHOLE, full-res)
and half-resolution frames in images_2/ (downscale_factor=2 per nb-info.json),
so K is scaled by 0.5; poses are unchanged. Scenes are video captures with
hundreds of frames, so we subsample a uniform stride of --max_cams frames
(uniform, not random: preserves sequential overlap in the view graph). Tracks
are then rebuilt with the SAME SIFT+FLANN pipeline as Strecha/ETH3D
(appendix_c.build_scene) and labeled by robust triangulation under GT poses.

Usage: python build_tnt.py [--scenes ignatius ...] [--max_cams 120] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from appendix_c import build_scene
from build_eth3d import quat_to_R

RAW = Path(__file__).resolve().parents[2] / "datasets/raw/tnt/tanksandtemples"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "datasets/tnt"
SCENES = ["barn", "caterpillar", "courthouse", "ignatius", "meetingroom", "truck"]
K_SCALE = 0.5  # images_2 is exactly half the calibrated resolution


def load_gt(scene_dir, max_cams):
    """Return (image_paths, Ps, Ks, names): uniform-stride subsample of the
    sequence, K scaled to the images_2 resolution."""
    calib = scene_dir / "sparse" / "0"
    img_root = scene_dir / "images_2"
    cam_K = {}
    for line in (calib / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cid = int(t[0]); model = t[1]
        assert model == "PINHOLE", f"unexpected camera model {model}"
        fx, fy, cx, cy = (float(v) * K_SCALE for v in t[4:8])
        cam_K[cid] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    rows = []
    lines = [l for l in (calib / "images.txt").read_text().splitlines()
             if not l.startswith("#")]
    for i in range(0, len(lines), 2):          # pose line + 2D-points line pairs
        t = lines[i].split()
        if len(t) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, t[1:8])
        cid = int(t[8]); name = Path(t[9]).name
        img_path = img_root / name
        if not img_path.exists():
            continue
        R = quat_to_R(qw, qx, qy, qz); tvec = np.array([tx, ty, tz])
        K = cam_K[cid]
        P = K @ np.hstack([R, tvec.reshape(3, 1)])   # K[R|t], images_2 pixels
        rows.append((name, img_path, P, K))
    rows.sort(key=lambda r: r[0])                    # frame order = capture order
    if len(rows) > max_cams:                          # uniform stride over the video
        idx = np.linspace(0, len(rows) - 1, max_cams).round().astype(int)
        rows = [rows[i] for i in idx]
    names = [r[0] for r in rows]
    image_paths = [r[1] for r in rows]
    Ps = [r[2] for r in rows]
    Ks = [r[3] for r in rows]
    return image_paths, Ps, Ks, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--max_cams", type=int, default=120)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        print(f"[tnt] {scene}", flush=True)
        image_paths, Ps, Ks, names = load_gt(RAW / scene, args.max_cams)
        print(f"  {len(image_paths)} frames (subsampled), K[0] fx={Ks[0][0,0]:.1f}",
              flush=True)
        data = build_scene(image_paths, Ps, Ks, names)
        out = args.out / f"{scene}.npz"
        np.savez(out, **data)
        print(f"  wrote {out} (M {data['M'].shape}, "
              f"outliers {data['outlier_pct']:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
