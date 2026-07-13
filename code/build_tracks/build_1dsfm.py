"""Build RESfM-format npz tracks for the 10 1DSfM scenes of RESfM Table 2.

Unlike Strecha/BlendedMVS we do not rerun SIFT: the 1DSfM release ships the
original SIFT point tracks (tracks.txt + coords.txt). GT cameras come from
gt_bundle.out (the dataset's reference bundler reconstruction), which stands
in for the COLMAP reconstruction of RESfM Appendix C's labeling procedure:

  1. keypoint appears in a gt_bundle point's view list -> initial inlier
  2. remove initial outliers, triangulate each cleaned track under GT poses
  3. relabel every observed keypoint by the 4-pixel reprojection threshold

Bundler conventions handled here: camera looks down -z with y up, so
P = K @ [D R | D t] with D = diag(1,-1,-1); point-view lists store
(x, y) centered on the principal point with y up, so pixel = (px + x, py - y).
coords.txt already stores top-left-origin pixel coordinates (verified at
runtime against the gt_bundle view lists; the build aborts if the two
disagree by more than DELTA_TOL pixels on average). Labeling uses the
linear K[R|t] model: coords.txt keypoints are already undistorted —
applying gt_bundle's (k1, k2) on top was tested and RAISED Notre_Dame's
outlier rate from 48.1% to 51.4%, i.e. it over-corrects.

Usage: python build_1dsfm.py [--scenes Alamo ...] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from appendix_c import MIN_TRACK_VIEWS, REPROJ_INLIER_PX, _triangulate

RAW = Path(__file__).resolve().parents[2] / "datasets/raw/1dsfm"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "datasets/1dsfm"
SCENES = ["Alamo", "Ellis_Island", "Madrid_Metropolis", "Montreal_Notre_Dame",
          "Notre_Dame", "NYC_Library", "Piazza_del_Popolo", "Tower_of_London",
          "Vienna_Cathedral", "Yorkminster"]
DELTA_TOL = 2.0  # px, coords-vs-bundle consistency check


def parse_coords(path):
    """coords.txt -> ({img: {key: (x, y)}}, {img: (px, py, focal)})."""
    feats, meta = {}, {}
    img = None
    with open(path) as f:
        for line in f:
            if line.startswith("#index"):
                parts = dict(p.split("=") for p in line.lstrip("#").split(","))
                parts = {k.strip(): v.strip() for k, v in parts.items()}
                img = int(parts["index"])
                meta[img] = (float(parts["px"]), float(parts["py"]),
                             float(parts["focal"]))
                feats[img] = {}
            else:
                t = line.split()
                if len(t) >= 3 and img is not None:
                    feats[img][int(t[0])] = (float(t[1]), float(t[2]))
    return feats, meta


def parse_tracks(path):
    """tracks.txt -> list of [(img, key), ...]."""
    tracks = []
    with open(path) as f:
        n = int(f.readline())
        for line in f:
            t = line.split()
            if not t:
                continue
            L = int(t[0])
            tracks.append([(int(t[1 + 2 * i]), int(t[2 + 2 * i]))
                           for i in range(L)])
    assert len(tracks) == n, f"{path}: expected {n} tracks, got {len(tracks)}"
    return tracks


def parse_gt_bundle(path):
    """gt_bundle.out -> (cams: {img: (f, k1, k2, R, t)},
    obs: {(img, key): (x, y)})."""
    with open(path) as f:
        header = f.readline()
        assert header.startswith("# Bundle"), header
        ncam, npts = map(int, f.readline().split())
        cams = {}
        for i in range(ncam):
            f_k1_k2 = np.array(f.readline().split(), np.float64)
            R = np.array([f.readline().split() for _ in range(3)], np.float64)
            t = np.array(f.readline().split(), np.float64)
            if f_k1_k2[0] > 0:
                cams[i] = (f_k1_k2[0], f_k1_k2[1], f_k1_k2[2], R, t)
        obs = {}
        for _ in range(npts):
            f.readline()  # xyz
            f.readline()  # rgb
            t = f.readline().split()
            nview = int(t[0])
            for v in range(nview):
                img = int(t[1 + 4 * v])
                key = int(t[2 + 4 * v])
                x = float(t[3 + 4 * v])
                y = float(t[4 + 4 * v])
                obs[(img, key)] = (x, y)
    return cams, obs


def check_coord_convention(feats, meta, obs):
    """Verify pixel = (px + x_bundle, py - y_bundle) matches coords.txt."""
    deltas = []
    for (img, key), (bx, by) in obs.items():
        if img in feats and key in feats[img]:
            px, py, _ = meta[img]
            cx, cy = feats[img][key]
            deltas.append((cx - (px + bx), cy - (py - by)))
            if len(deltas) >= 2000:
                break
    d = np.abs(np.array(deltas)).mean(axis=0)
    assert d.max() < DELTA_TOL, (
        f"coords.txt vs gt_bundle convention mismatch, mean |delta| = {d}")


def build_scene_1dsfm(scene_dir):
    feats, meta = parse_coords(scene_dir / "coords.txt")
    tracks = parse_tracks(scene_dir / "tracks.txt")
    cams, obs = parse_gt_bundle(scene_dir / "gt_bundle.out")
    check_coord_convention(feats, meta, obs)

    # Cameras: gt_bundle cameras with a valid focal, in stable index order.
    cam_ids = sorted(cams)
    cam_row = {c: i for i, c in enumerate(cam_ids)}
    m = len(cam_ids)

    D = np.diag([1.0, -1.0, -1.0])
    Ps, Ks = [], []
    for c in cam_ids:
        f, _k1, _k2, R, t = cams[c]
        px, py, _ = meta[c]
        K = np.array([[f, 0, px], [0, f, py], [0, 0, 1.0]])
        Ps.append(K @ np.hstack([D @ R, (D @ t)[:, None]]))
        Ks.append(K)
    Ps = np.array(Ps)
    Ks = np.array(Ks)

    # Tracks restricted to GT cameras; enforce Appendix C validity.
    kept = []
    for tr in tracks:
        tr = [(im, k) for im, k in tr if im in cam_row and k in feats.get(im, {})]
        imgs = [im for im, _ in tr]
        if len(set(imgs)) < MIN_TRACK_VIEWS or len(imgs) != len(set(imgs)):
            continue
        kept.append(tr)
    n = len(kept)

    M = np.zeros((2 * m, n), np.float64)
    initial_inlier = np.zeros((m, n), bool)
    for col, tr in enumerate(kept):
        for im, key in tr:
            r = cam_row[im]
            M[2 * r, col], M[2 * r + 1, col] = feats[im][key]
            if (im, key) in obs:
                initial_inlier[r, col] = True

    # Appendix C labeling with gt_bundle membership as the initial split.
    observed = M.reshape(m, 2, n).any(axis=1)
    outliers = np.zeros((m, n), np.float64)
    xs_all = M.reshape(m, 2, n)
    for col in range(n):
        views = np.flatnonzero(observed[:, col])
        clean = [v for v in views if initial_inlier[v, col]]
        X = None
        if len(clean) >= 2:
            X = _triangulate([Ps[v] for v in clean],
                             [xs_all[v, :, col] for v in clean])
            if abs(X[3]) > 1e-12:
                X = X / X[3]
        if X is None:
            outliers[views, col] = 1.0
            continue
        for v in views:
            p = Ps[v] @ X
            err = (np.linalg.norm(p[:2] / p[2] - xs_all[v, :, col])
                   if p[2] > 1e-12 else np.inf)
            if err >= REPROJ_INLIER_PX:
                outliers[v, col] = 1.0

    outlier_pct = 100.0 * outliers.sum() / max(observed.sum(), 1)
    Ns = np.stack([np.linalg.inv(K) for K in Ks])
    names = np.array([f"{c:06d}" for c in cam_ids])
    return dict(M=M, Ns=Ns, Ps_gt=Ps, K_gt=Ks, outliers2=outliers,
                outlier_pct=outlier_pct, namesList=names), m, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        print(f"[1dsfm] {scene}", flush=True)
        data, m, n = build_scene_1dsfm(RAW / scene)
        out = args.out / f"{scene}.npz"
        np.savez(out, **data)
        print(f"  wrote {out}: {m} cams, {n} tracks, "
              f"outliers {data['outlier_pct']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
