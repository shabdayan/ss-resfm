#!/usr/bin/env python
"""BA-from-both-mirror-hypotheses experiment on Drinking Fountain, both arms.

For each run we form two initializations for the SHARED bundle adjustment:
  A = the method's raw reconstruction (Rs, ts, Xs)
  B = a global reflection of it  (R'=S R S, t'=S t, X'=S X),  S=diag(-1,1,1)
      (this S keeps camera-frame depth z>0, so B stays cheirality-valid as a
       BA init; it reprojects to a horizontally-flipped image, so BA must
       re-fit the real observations from the reflected basin.)
Then run the SAME pycolmap BA on both, and SELECT by post-BA reprojection error
(GT-FREE — the only fair criterion). We print, for both hypotheses, the post-BA
reprojection error (selection metric) AND the rotation/position error vs GT (to
see whether reprojection-selection actually recovers the scene).
"""
import os, sys, json, glob
import numpy as np

CODE_DIR = '/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code'
sys.path.insert(0, CODE_DIR); os.chdir(CODE_DIR)
from utils import geo_utils
import evaluate_single_scene as E
import pycolmap
from utils.ba_advanced import (batch_matrix_to_pycolmap, prepare_ba_options,
                               pycolmap_to_batch_matrix)

SCENE = 'Drinking Fountain Somewhere In Zurich'      # spaced: GT npz + camera filenames
SLUG = 'Drinking_Fountain_Somewhere_In_Zurich'       # underscored: results dir
S = np.diag([-1.0, 1.0, 1.0])  # keeps camera-frame z>0


def ba_from_init(Rs, ts, Xs_h, xs_full, gt):
    """Run the shared BA from a given (Rs, ts, Xs) init; return post-BA metrics."""
    Xs = Xs_h.copy()
    visible = xs_full[:, :, 0] > 0
    keep = np.where(visible.sum(axis=0) >= 2)[0]
    xs = xs_full[:, keep]
    X = Xs[keep, :3] / Xs[keep, 3:4]
    rec = batch_matrix_to_pycolmap(xs, Rs, ts, gt['Ks'], X)
    pycolmap.bundle_adjustment(rec, prepare_ba_options())
    nR, nt, nP, _, nX = pycolmap_to_batch_matrix(rec, original_num_points=xs.shape[1])
    Rf, tf, _ = geo_utils.align_cameras(nR, gt['Rs_gt'], nt, gt['ts_gt'], return_alignment=True)
    Re, te = geo_utils.tranlsation_rotation_errors(Rf, tf, gt['Rs_gt'], gt['ts_gt'])
    reproj = geo_utils.reprojection_error_with_points(nP, nX, xs)
    return {'reproj_px': float(np.nanmean(reproj)), 'reproj_med': float(np.nanmedian(reproj)),
            'rot_deg': float(np.mean(Re)), 'pos': float(np.mean(te))}


def mirror(Rs, ts, Xs):
    """Global reflection by S of the whole reconstruction (proper rotations kept)."""
    Rm = np.einsum('ij,njk,kl->nil', S, Rs, S)     # R' = S R S
    tm = (S @ ts.T).T                              # t' = S t
    Xm = Xs.copy()
    Xm[:, :3] = (S @ Xs[:, :3].T).T                # X' = S X (homogeneous w kept)
    return Rm, tm, Xm


gt = E.load_gt(SCENE)
print(f"{'run':16} {'hyp':6} {'reproj_px':>10} {'rot_deg':>9} {'pos':>8}   pick")
print('-' * 62)
for method in ('esfm', 'uesfm'):
    for seed_dir in sorted(glob.glob(f'results/single_scene/{method}/{SLUG}/seed*')):
        cams = np.load(E.cameras_npz_path(seed_dir, method, SCENE), allow_pickle=True)
        g = E.subset_gt(cams, gt)
        Rs = cams['Rs'].astype(np.float64); ts = cams['ts'].astype(np.float64)
        Xs = cams['pts3D_pred'].astype(np.float64).T  # (n,4) homogeneous
        xs = cams['xs'].astype(np.float64)

        a = ba_from_init(Rs, ts, Xs, xs, g)
        Rm, tm, Xm = mirror(Rs, ts, Xs)
        b = ba_from_init(Rm, tm, Xm, xs, g)

        pick = 'A' if a['reproj_px'] <= b['reproj_px'] else 'B'
        tag = f"{method} {os.path.basename(seed_dir)}"
        for name, h in (('A(orig)', a), ('B(mirror)', b)):
            sel = '  <== PICK' if name[0] == pick else ''
            print(f"{tag:16} {name:9} {h['reproj_px']:10.4f} {h['rot_deg']:9.3f} {h['pos']:8.3f}{sel}")
        chosen = a if pick == 'A' else b
        print(f"{'':16} -> selected-by-reproj: {pick}  rot={chosen['rot_deg']:.3f}deg  pos={chosen['pos']:.3f}\n")
