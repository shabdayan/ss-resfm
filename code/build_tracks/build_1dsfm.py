"""Build RESfM-format npz tracks for the 10 1DSfM scenes of RESfM Table 2.

Unlike Strecha/BlendedMVS we do not rerun SIFT: the 1DSfM release ships the
original SIFT point tracks (tracks.txt + coords.txt). GT cameras come from
gt_bundle.out (the dataset's reference bundler reconstruction), which stands
in for the COLMAP reconstruction of RESfM Appendix C's labeling procedure:

  1. keypoint appears in a gt_bundle point's view list -> initial inlier
  2. remove initial outliers, triangulate each cleaned track under GT poses
  3. relabel every observed keypoint by the 4-pixel reprojection threshold

Bundler conventions handled here: camera looks down -z with y up, so
P = K @ [D R | D t] with D = diag(1,-1,-1); coords.txt stores
top-left-origin pixel coordinates. CRITICAL indexing quirk: gt_bundle's
camera blocks are indexed globally (one per list.txt image), but its point
VIEW LISTS index into the list of valid (focal > 0) cameras — Notre_Dame
masks this because all its cameras are valid (identity remap), Alamo does
not (761 valid of 2915). parse_gt_bundle remaps; a per-scene
self-consistency check (projecting gt_bundle's own points through its own
cameras onto the referenced coords.txt keys, full k1/k2 bundler model)
asserts the remap holds (Alamo: median 0.6 px, 98.5% < 4 px). Track
labeling itself uses the linear K[R|t] model: coords.txt keypoints are
already undistorted — applying (k1, k2) there was tested and RAISED
Notre_Dame's outlier rate 48.1% -> 51.4% (over-correction).

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


def parse_coords(path):
    """coords.txt -> ({img: {key: (x, y)}}, {img: (px, py, focal)})."""
    feats, meta = {}, {}
    img = None
    with open(path) as f:
        for line in f:
            if line.startswith("#index"):
                # Filenames may contain '=' or ','; parse only the numeric
                # fields we need, each of the form "key = <number>".
                parts = dict(p.split("=", 1) for p in line.lstrip("#").split(",")
                             if p.count("=") >= 1 and
                             p.split("=", 1)[0].strip() in
                             ("index", "px", "py", "focal"))
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
    """gt_bundle.out -> (cams: {global_img: (f, k1, k2, R, t)},
    member: set of (global_img, key), points: [(X, [(global_img, key)])]).

    Camera blocks are indexed globally (one per list.txt image; focal 0 =
    not reconstructed). Point view lists index into the list of VALID
    (focal > 0) cameras, NOT globally — remapped here via valid_ids.
    """
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
        valid_ids = sorted(cams)
        member = set()
        points = []
        for _ in range(npts):
            X = np.array(f.readline().split(), np.float64)
            f.readline()  # rgb
            t = f.readline().split()
            nview = int(t[0])
            vl = []
            for v in range(nview):
                iv = int(t[1 + 4 * v])
                key = int(t[2 + 4 * v])
                if iv < len(valid_ids):
                    img = valid_ids[iv]
                    member.add((img, key))
                    vl.append((img, key))
            points.append((X, vl))
    return cams, member, points


def check_bundle_selfconsistency(cams, points, feats, meta, sample=8000):
    """Project gt_bundle's own 3D points (full bundler model, k1/k2) onto the
    coords.txt keypoints their view lists reference. Median error must be
    small or the view-list indexing assumption is wrong for this scene."""
    errs = []
    for X, vl in points:
        for img, key in vl:
            if img not in cams or key not in feats.get(img, {}):
                continue
            f, k1, k2, R, t = cams[img]
            pc = R @ X + t
            if pc[2] >= 0:
                continue
            p = -pc[:2] / pc[2]
            r2 = p @ p
            rad = 1.0 + k1 * r2 + k2 * r2 * r2
            px, py, _ = meta[img]
            proj = np.array([px + f * rad * p[0], py - f * rad * p[1]])
            errs.append(np.linalg.norm(proj - np.array(feats[img][key])))
        if len(errs) >= sample:
            break
    errs = np.array(errs)
    med = float(np.median(errs))
    frac4 = float(np.mean(errs < REPROJ_INLIER_PX))
    assert med < REPROJ_INLIER_PX, (
        f"gt_bundle view-list indexing failed self-consistency: "
        f"median proj err {med:.1f}px (frac<4px {frac4:.3f})")
    return med, frac4


def build_scene_1dsfm(scene_dir):
    feats, meta = parse_coords(scene_dir / "coords.txt")
    tracks = parse_tracks(scene_dir / "tracks.txt")
    cams, member, points = parse_gt_bundle(scene_dir / "gt_bundle.out")
    med, frac4 = check_bundle_selfconsistency(cams, points, feats, meta)
    print(f"  gt_bundle self-consistency: median {med:.2f}px, "
          f"frac<4px {frac4:.3f}", flush=True)

    # Cameras: gt_bundle cameras with a valid focal that also appear in
    # coords.txt (some list.txt images have no extracted features and thus
    # no principal point; they can never contribute observations).
    cam_ids = sorted(c for c in cams if c in meta and feats.get(c))
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
            if (im, key) in member:
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

    # Drop cameras that ended up with zero observations (harmless but dead
    # rows; removing them cannot break the >=3-view track constraint).
    live = np.flatnonzero(observed.any(axis=1))
    if len(live) < m:
        double = np.stack([2 * live, 2 * live + 1], axis=1).ravel()
        M = M[double]
        Ps, Ks = Ps[live], Ks[live]
        outliers, observed = outliers[live], observed[live]
        cam_ids = [cam_ids[i] for i in live]
        m = len(live)

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
