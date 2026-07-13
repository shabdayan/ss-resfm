"""Sanity-check the built npz scene files.

For every npz under code/datasets/{strecha,blendedmvs,1dsfm}: verify the
schema fields, then measure per-keypoint reprojection error of the GT-labeled
inliers against a fresh DLT triangulation under Ps_gt (should be well under
4 px if conventions are right), and report the outlier fraction.

Usage: python verify_npz.py [dirs...]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from appendix_c import _triangulate

DEFAULT_DIRS = ["strecha", "blendedmvs", "1dsfm"]


def inlier_reproj_stats(M, Ps, outliers):
    m, n = M.shape[0] // 2, M.shape[1]
    xs = M.reshape(m, 2, n)
    observed = xs.any(axis=1)
    inl = observed & (outliers == 0)
    errs = []
    cols = np.flatnonzero(inl.sum(axis=0) >= 2)
    step = max(1, len(cols) // 2000)  # sample tracks for speed
    for col in cols[::step]:
        views = np.flatnonzero(inl[:, col])
        X = _triangulate([Ps[v] for v in views], [xs[v, :, col] for v in views])
        if abs(X[3]) < 1e-12:
            continue
        X = X / X[3]
        for v in views:
            p = Ps[v] @ X
            if p[2] > 1e-12:
                errs.append(np.linalg.norm(p[:2] / p[2] - xs[v, :, col]))
    return np.array(errs)


def main():
    base = Path(__file__).resolve().parents[1] / "datasets"
    dirs = sys.argv[1:] or DEFAULT_DIRS
    for d in dirs:
        for f in sorted((base / d).glob("*.npz")):
            data = np.load(f, allow_pickle=True)
            M, Ps, Ns = data["M"], data["Ps_gt"], data["Ns"]
            out = data["outliers2"]
            m, n = M.shape[0] // 2, M.shape[1]
            assert Ps.shape == (m, 3, 4) and Ns.shape == (m, 3, 3)
            assert out.shape == (m, n) and len(data["namesList"]) == m
            KtimesN = data["K_gt"][0] @ Ns[0]
            assert np.allclose(KtimesN, np.eye(3), atol=1e-6), "Ns != K^-1"
            errs = inlier_reproj_stats(M, Ps, out)
            obs = M.reshape(m, 2, n).any(axis=1).sum()
            print(f"{d}/{f.stem}: {m} cams, {n} tracks, "
                  f"outliers {float(data['outlier_pct']):.1f}%, "
                  f"inlier reproj mean {errs.mean():.2f}px "
                  f"p99 {np.percentile(errs, 99):.2f}px "
                  f"({obs} obs)", flush=True)


if __name__ == "__main__":
    main()
