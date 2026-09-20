#!/usr/bin/env python
"""Design A: build RoMa tracks on the SIFT rebuild's pair graph (matched pairs).

For a scene already rebuilt with SIFT (datasets/megadepth_rebuilt_udfirst or a
1DSfM suite), we reuse exactly the camera set, GT poses, intrinsics and image
paths, and match the SAME pairs with RoMa instead of SIFT. The pair list is
recovered from the SIFT rebuild's matching checkpoint when present, else from
the SIFT tracks themselves (camera pairs that share >= MIN_SHARED tracks).

Output: datasets/<suite>_roma/<scene>.npz with the Appendix-C fields, so it
drops straight into training/eval as another dataset.

Usage: build_roma_tracks.py <scene> --suite megadepth|1dsfm|1dsfm_hard_300
"""
import argparse, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
MIN_SHARED = 15          # pair kept if the SIFT rebuild linked >= this many tracks


def sift_pair_list(M, min_shared=MIN_SHARED):
    """Camera pairs co-observing at least min_shared tracks in the SIFT build."""
    vis = (M[0::2] != 0) | (M[1::2] != 0)          # (m, n) bool
    co = vis.astype(np.int32) @ vis.astype(np.int32).T
    iu = np.triu_indices(co.shape[0], 1)
    keep = co[iu] >= min_shared
    return list(zip(iu[0][keep].tolist(), iu[1][keep].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--suite", default="megadepth",
                    choices=["megadepth", "1dsfm", "1dsfm_hard_300"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.suite == "megadepth":
        src_dir = os.path.join(CODE, "datasets", "megadepth_rebuilt_udfirst")
        out_dir = os.path.join(CODE, "datasets", "megadepth_roma")
    else:
        src_dir = os.path.join(CODE, "datasets", args.suite)
        out_dir = os.path.join(CODE, "datasets", f"{args.suite}_roma")
    out_path = os.path.join(out_dir, f"{args.scene}.npz")
    if os.path.exists(out_path):
        print(f"[{args.scene}] already built"); return

    src = dict(np.load(os.path.join(src_dir, f"{args.scene}.npz"), allow_pickle=True))
    M, Ps_gt, Ks = src["M"], src["Ps_gt"], src["K_gt"]
    names = [str(n) for n in src["namesList"]]

    # image paths + per-camera rescale into the npz frame (same resolver the
    # feature extractors use), imported by file path to avoid package clashes.
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("_obsfeat",
                                       os.path.join(CODE, "extract_obs_features.py"))
    ofeat = ilu.module_from_spec(spec); spec.loader.exec_module(ofeat)
    ds = "megadepth" if args.suite == "megadepth" else args.suite
    raw = os.path.join(CODE, "..", "datasets", "raw", "1dsfm") \
        if ds.startswith("1dsfm") else os.path.join(CODE, "datasets", "raw_megadepth")
    paths, scales = ofeat.resolve_images(args.scene, names, raw, ds, K_gt=Ks)
    if any(p is None for p in paths):
        sys.exit(f"[{args.scene}] unresolved images -- aborting")
    if any(abs(s - 1.0) > 1e-3 for s in scales):
        # RoMa matches on the undistorted images; coordinates are mapped into
        # the npz frame after matching (see below).
        print(f"[{args.scene}] per-camera rescale active "
              f"(median {np.median(scales):.3f})", flush=True)

    pairs = sift_pair_list(M)
    print(f"[{args.scene}] {M.shape[0]//2} cams, SIFT pair graph: {len(pairs)} pairs",
          flush=True)

    sys.path.insert(0, HERE)
    from roma_matcher import build_pair_matches
    from appendix_c import chain_tracks, label_outliers

    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.join(out_dir, f".ck_{args.scene}.pkl")
    kps_all, pair_matches = build_pair_matches(paths, pairs, device=args.device,
                                               checkpoint_path=ck)
    # map keypoints from image pixels into the npz coordinate frame
    kps_all = [k * s for k, s in zip(kps_all, scales)]

    M2 = chain_tracks(len(paths), pair_matches, kps_all)
    outliers, observed = label_outliers(M2, np.asarray(Ps_gt))
    pct = 100.0 * outliers.sum() / max(observed.sum(), 1)
    Ns = np.stack([np.linalg.inv(K) for K in Ks])
    np.savez(out_path, M=M2, Ns=Ns, Ps_gt=np.asarray(Ps_gt, np.float64),
             K_gt=np.asarray(Ks, np.float64), outliers2=outliers,
             outlier_pct=pct, namesList=np.array(names),
             matcher="roma", pair_source="sift_rebuild_graph")
    if os.path.exists(ck):
        os.remove(ck)
    print(f"[{args.scene}] DONE tracks {M.shape[1]} (SIFT) -> {M2.shape[1]} (RoMa); "
          f"outlier_pct {pct:.1f}%", flush=True)


if __name__ == "__main__":
    main()
