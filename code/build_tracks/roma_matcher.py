#!/usr/bin/env python
"""Design A: RoMa dense matching over a FIXED pair graph, producing the same
(kps_all, pair_matches) interface the Appendix-C chainer consumes.

Matched-pairs protocol: the pair graph comes from the SIFT rebuild of the same
scene (the pairs whose SIFT matching survived RANSAC), so a RoMa-vs-SIFT track
comparison differs only in HOW each pair is matched, never in WHICH pairs are
attempted. Exhaustive RoMa matching is infeasible (45k-450k pairs/scene).

Per pair: RoMa warp + certainty -> sample matches -> keep the top
MATCHES_PER_PAIR by certainty -> the same fundamental-matrix RANSAC filter the
SIFT path uses. Keypoints are registered per image in a quantized grid so that
the same physical point matched in several pairs becomes ONE track element
(RoMa produces free-form coordinates; without the registry every pair would
invent new keypoints and chaining would never link three views).

Runs in the dense env (tools/dense/mm/envs/dense).
"""
import numpy as np
import cv2
import torch

CERT_THR = 0.5             # RoMa certainty floor for a sampled match
MATCHES_PER_PAIR = 2000    # top-certainty matches kept per pair (pre-RANSAC)
RANSAC_PX = 3.0            # identical to appendix_c.RANSAC_PX
MIN_PAIR_INLIERS = 15      # identical to appendix_c.MIN_PAIR_INLIERS
GRID_PX = 2.0              # keypoint registry quantization (pixels)


class KeypointRegistry:
    """Per-image quantized keypoint store: coordinate -> stable index."""

    def __init__(self, num_images, grid_px=GRID_PX):
        self.grid = grid_px
        self.maps = [dict() for _ in range(num_images)]
        self.pts = [[] for _ in range(num_images)]

    def get(self, img_idx, xy):
        key = (int(round(xy[0] / self.grid)), int(round(xy[1] / self.grid)))
        m = self.maps[img_idx]
        idx = m.get(key)
        if idx is None:
            idx = len(self.pts[img_idx])
            m[key] = idx
            self.pts[img_idx].append((float(xy[0]), float(xy[1])))
        return idx

    def arrays(self):
        return [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in self.pts]


def load_model(device="cuda", indoor=False):
    # use_custom_corr=False: the packaged `local_corr` CUDA extension is not
    # built in this env; the pure-PyTorch correlation path is equivalent.
    from romatch import roma_indoor, roma_outdoor
    model = (roma_indoor if indoor else roma_outdoor)(
        device=device, use_custom_corr=False, upsample_preds=False)
    return model


def match_pair_roma(model, path_i, path_j, device="cuda"):
    """Return (pts_i, pts_j) float arrays of RANSAC-surviving matches."""
    with torch.no_grad():
        warp, certainty = model.match(str(path_i), str(path_j), device=device)
        matches, conf = model.sample(warp, certainty, num=MATCHES_PER_PAIR * 2)
        keep = conf > CERT_THR
        matches, conf = matches[keep], conf[keep]
        if matches.shape[0] < 8:
            return None
        if matches.shape[0] > MATCHES_PER_PAIR:
            top = torch.topk(conf, MATCHES_PER_PAIR).indices
            matches = matches[top]
        H_i, W_i = cv2.imread(str(path_i)).shape[:2]
        H_j, W_j = cv2.imread(str(path_j)).shape[:2]
        pts_i, pts_j = model.to_pixel_coordinates(matches, H_i, W_i, H_j, W_j)
        pts_i = pts_i.cpu().numpy().astype(np.float32)
        pts_j = pts_j.cpu().numpy().astype(np.float32)
    _, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC,
                                     RANSAC_PX, 0.999, 10000)
    if mask is None:
        return None
    m = mask.ravel().astype(bool)
    if m.sum() < MIN_PAIR_INLIERS:
        return None
    return pts_i[m], pts_j[m]


def build_pair_matches(image_paths, pair_list, device="cuda", verbose=True,
                       checkpoint_path=None, model=None):
    """RoMa-match every (i, j) in pair_list; return (kps_all, pair_matches)
    in the Appendix-C chainer's format."""
    import os, pickle
    reg = KeypointRegistry(len(image_paths))
    pair_matches = {}
    start = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ck = pickle.load(f)
        reg.maps, reg.pts = ck["maps"], ck["pts"]
        pair_matches, start = ck["pair_matches"], ck["next"]
        if verbose:
            print(f"  resuming RoMa matching at pair {start}/{len(pair_list)}",
                  flush=True)
    if model is None:
        model = load_model(device)
    for t in range(start, len(pair_list)):
        i, j = pair_list[t]
        out = match_pair_roma(model, image_paths[i], image_paths[j], device)
        if out is not None:
            pts_i, pts_j = out
            idx = np.array([[reg.get(i, a), reg.get(j, b)]
                            for a, b in zip(pts_i, pts_j)], dtype=np.int64)
            # a quantized cell can absorb several raw matches; keep unique pairs
            pair_matches[(i, j)] = np.unique(idx, axis=0)
        if verbose and (t + 1) % 100 == 0:
            print(f"  RoMa matched {t+1}/{len(pair_list)} pairs", flush=True)
        if checkpoint_path and ((t + 1) % 250 == 0 or t == len(pair_list) - 1):
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"maps": reg.maps, "pts": reg.pts,
                             "pair_matches": pair_matches, "next": t + 1}, f,
                            protocol=4)
            os.replace(tmp, checkpoint_path)
    return reg.arrays(), pair_matches
