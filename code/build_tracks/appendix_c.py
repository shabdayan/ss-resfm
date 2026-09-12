"""Point-track construction per RESfM Appendix C (Khatib et al., ICLR 2025).

Pipeline: SIFT features -> exhaustive pairwise RANSAC matching -> chain
two-view matches into tracks. A track is discarded if it is viewed in fewer
than 3 cameras or contains an inconsistent cycle (two keypoints in the same
image). Outlier labels come from triangulation against the dataset's GT
cameras with a 4-pixel reprojection threshold (adaptation of Appendix C's
COLMAP-based labeling for datasets that ship real GT poses; see
BUILD_TRACKS.md).

Parameters Appendix C leaves unspecified (documented in BUILD_TRACKS.md):
SIFT capped at MAX_FEATURES per image, Lowe ratio RATIO_THR, fundamental-
matrix RANSAC threshold RANSAC_PX pixels.
"""

import numpy as np
import cv2
import os

MAX_FEATURES = 8192          # COLMAP's default cap
RATIO_THR = 0.8              # Lowe's ratio test
RANSAC_PX = 3.0              # F-matrix RANSAC inlier threshold (pixels)
MIN_PAIR_INLIERS = 15        # discard a pair's matches below this
MIN_TRACK_VIEWS = 3          # Appendix C: >= 3 cameras
REPROJ_INLIER_PX = 4.0       # Appendix C: 4-pixel inlier threshold


def extract_sift(image_paths):
    """Return (keypoints_list, descriptors_list); keypoints as (k,2) arrays."""
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    kps_all, desc_all = [], []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"cannot read image {p}")
        kps, desc = sift.detectAndCompute(img, None)
        if desc is None:
            kps, desc = [], np.zeros((0, 128), np.float32)
        kps_all.append(np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2))
        desc_all.append(desc)
    return kps_all, desc_all


def match_pair(kps_i, desc_i, kps_j, desc_j, matcher):
    """Ratio-test kNN matching + fundamental-matrix RANSAC.

    Returns an (r, 2) int array of (kp_idx_i, kp_idx_j) geometric inliers.
    """
    if len(desc_i) < 8 or len(desc_j) < 8:
        return np.zeros((0, 2), np.int64)
    knn = matcher.knnMatch(desc_i, desc_j, k=2)
    good = [m for m, n in (pair for pair in knn if len(pair) == 2)
            if m.distance < RATIO_THR * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2), np.int64)
    idx = np.array([(m.queryIdx, m.trainIdx) for m in good], np.int64)
    pts_i = kps_i[idx[:, 0]].astype(np.float32)
    pts_j = kps_j[idx[:, 1]].astype(np.float32)
    _, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC,
                                     RANSAC_PX, 0.999, 10000)
    if mask is None:
        return np.zeros((0, 2), np.int64)
    inl = idx[mask.ravel().astype(bool)]
    if len(inl) < MIN_PAIR_INLIERS:
        return np.zeros((0, 2), np.int64)
    return inl


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        root = a
        while self.parent.setdefault(root, root) != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def chain_tracks(num_images, pair_matches, kps_all):
    """Union-find chaining of two-view matches into multi-view tracks.

    pair_matches: dict {(i, j): (r, 2) array of keypoint index pairs}
    Returns M [2m, n] with pixel coords (0 = unobserved), after dropping
    tracks seen in < MIN_TRACK_VIEWS images or with two keypoints in the
    same image (inconsistent cycle).
    """
    uf = UnionFind()
    for (i, j), matches in pair_matches.items():
        for ki, kj in matches:
            uf.union((i, int(ki)), (j, int(kj)))

    groups = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)

    tracks = []
    for members in groups.values():
        imgs = [im for im, _ in members]
        if len(set(imgs)) < MIN_TRACK_VIEWS:
            continue
        if len(imgs) != len(set(imgs)):  # inconsistent cycle
            continue
        tracks.append(members)

    n = len(tracks)
    M = np.zeros((2 * num_images, n), np.float64)
    for col, members in enumerate(tracks):
        for im, kp in members:
            M[2 * im, col], M[2 * im + 1, col] = kps_all[im][kp]
    return M


def _triangulate(Ps, xs):
    """DLT triangulation. Ps: list of (3,4); xs: list of (2,). Returns (4,)."""
    A = np.zeros((2 * len(Ps), 4))
    for r, (P, x) in enumerate(zip(Ps, xs)):
        A[2 * r] = x[0] * P[2] - P[0]
        A[2 * r + 1] = x[1] * P[2] - P[1]
    _, _, vt = np.linalg.svd(A)
    return vt[-1]


def label_outliers(M, Ps_gt):
    """RANSAC triangulation of each track under GT cameras; 4 px labeling.

    For each track: triangulate every view pair, keep the 3D point with the
    largest <4 px consensus, re-triangulate from the consensus set, then
    label every observed keypoint by the 4 px reprojection threshold.
    Tracks that never reach a 2-view consensus get all views labeled outlier.

    Returns outliers [m, n] float (1 = outlier keypoint).
    """
    m = M.shape[0] // 2
    n = M.shape[1]
    outliers = np.zeros((m, n), np.float64)
    xs_all = M.reshape(m, 2, n)
    observed = (xs_all != 0).any(axis=1)  # [m, n]

    for col in range(n):
        views = np.flatnonzero(observed[:, col])
        xs = {v: xs_all[v, :, col] for v in views}

        def reproj_ok(X):
            if abs(X[3]) > 1e-12:
                X = X / X[3]
            errs = {}
            for v in views:
                p = Ps_gt[v] @ X
                if p[2] <= 1e-12:  # behind camera / at infinity
                    errs[v] = np.inf
                    continue
                errs[v] = np.linalg.norm(p[:2] / p[2] - xs[v])
            return errs

        best_X, best_cons = None, []
        for a in range(len(views)):
            for b in range(a + 1, len(views)):
                va, vb = views[a], views[b]
                X = _triangulate([Ps_gt[va], Ps_gt[vb]], [xs[va], xs[vb]])
                errs = reproj_ok(X)
                cons = [v for v in views if errs[v] < REPROJ_INLIER_PX]
                if len(cons) > len(best_cons):
                    best_cons, best_X = cons, X
        if len(best_cons) >= 2:
            X = _triangulate([Ps_gt[v] for v in best_cons],
                             [xs[v] for v in best_cons])
            errs = reproj_ok(X)
            for v in views:
                if errs[v] >= REPROJ_INLIER_PX:
                    outliers[v, col] = 1.0
        else:
            for v in views:
                outliers[v, col] = 1.0
    return outliers, observed


def build_scene(image_paths, Ps_gt, Ks, names, verbose=True, checkpoint_path=None):
    """Full Appendix C pipeline for one scene.

    Returns dict of npz fields: M, Ns, Ps_gt, outliers2, outlier_pct, namesList.
    """
    m = len(image_paths)
    if verbose:
        print(f"  SIFT on {m} images...", flush=True)
    kps_all, desc_all = extract_sift(image_paths)

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=4), dict(checks=64))
    pair_matches = {}
    total_pairs = m * (m - 1) // 2
    start_i = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        # row-granular resume across queue run limits (25-row atomic dumps)
        import pickle
        with open(checkpoint_path, "rb") as f:
            ck = pickle.load(f)
        pair_matches, start_i = ck["pair_matches"], ck["next_i"]
        if verbose:
            print(f"  resuming matching at row {start_i}/{m} "
                  f"({len(pair_matches)} pairs kept)", flush=True)
    done = start_i * (m - 1) - start_i * (start_i - 1) // 2
    for i in range(start_i, m):
        for j in range(i + 1, m):
            inl = match_pair(kps_all[i], desc_all[i], kps_all[j], desc_all[j],
                             matcher)
            if len(inl):
                pair_matches[(i, j)] = inl
            done += 1
            if verbose and done % 200 == 0:
                print(f"  matched {done}/{total_pairs} pairs", flush=True)
        if checkpoint_path and (i % 25 == 24 or i == m - 1):
            import pickle
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"pair_matches": pair_matches, "next_i": i + 1}, f,
                            protocol=4)
            os.replace(tmp, checkpoint_path)

    M = chain_tracks(m, pair_matches, kps_all)
    if verbose:
        print(f"  tracks kept: {M.shape[1]}", flush=True)

    outliers, observed = label_outliers(M, np.asarray(Ps_gt))
    outlier_pct = 100.0 * outliers.sum() / max(observed.sum(), 1)
    if verbose:
        print(f"  outliers: {outlier_pct:.1f}% of {int(observed.sum())} keypoints",
              flush=True)

    Ns = np.stack([np.linalg.inv(K) for K in Ks])
    return dict(M=M, Ns=Ns, Ps_gt=np.asarray(Ps_gt, np.float64),
                K_gt=np.asarray(Ks, np.float64),
                outliers2=outliers, outlier_pct=outlier_pct,
                namesList=np.array(names))
