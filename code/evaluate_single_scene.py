#!/usr/bin/env python
"""
Phase 2 of SPEC_single_scene_experiments.md: ONE shared evaluation for both
methods' raw outputs.

For every completed run under results/single_scene/<method>/<scene>/seed<k>/ this
script loads the method's own predicted cameras (raw npz written by the method's
code), recomputes all metrics with a SINGLE implementation (this repo's
utils.geo_utils), applying the same Sim(3) alignment against ground truth loaded
from the shared datasets/Euclidean/<scene>.npz:

    rot_err_deg   mean rotation error in degrees, after alignment
    pos_err       mean camera-position (translation) error, after alignment
    reproj_err_px mean reprojection error in pixels (predicted 3D points)
    reproj_med_px median reprojection error in pixels
    n_cams        number of cameras in the prediction

With --ba, the SAME bundle adjustment implementation (this repo's pycolmap-based
utils.ba_functions.euc_ba) is applied to BOTH methods' pre-BA cameras and post-BA
metrics are reported separately (rot_err_deg_ba, ...). Neither method's own BA is
ever used, so post-processing is identical by construction.

Each run's native metrics (from the method's own xlsx) are cross-checked against
the harmonized ones; a relative disagreement above --flag-threshold (default 5%)
is flagged loudly, as required by the spec.

Results are written to <run_dir>/harmonized_metrics.json.
Run from u-esfm/code with the project venv:  .venv/bin/python evaluate_single_scene.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

from utils import geo_utils  # noqa: E402  (single shared metric implementation)

DATASETS_EUC = os.path.join(os.path.dirname(CODE_DIR), 'datasets', 'Euclidean')


def unslug(scene_slug):
    return scene_slug.replace('_', ' ')


def load_gt(scene):
    d = np.load(os.path.join(DATASETS_EUC, scene + '.npz'), allow_pickle=True)
    Ps_gt = d['Ps_gt'].astype(np.float64)
    Ns = d['Ns'].astype(np.float64)
    Ks = np.linalg.inv(Ns)
    Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(Ps_gt, Ks)
    return {'Ps_gt': Ps_gt, 'Ns': Ns, 'Ks': Ks, 'Rs_gt': Rs_gt, 'ts_gt': ts_gt}


def cameras_npz_path(run_dir, method, scene):
    if method == 'esfm':
        return os.path.join(run_dir, 'raw', 'Final_Cameras.npz')
    return os.path.join(run_dir, 'raw', 'forFigures', '{}_Final_Cameras.npz'.format(scene))


def native_metrics(run_dir, method, scene):
    """Read the method's own final metrics row from its results xlsx."""
    if method == 'esfm':
        path = os.path.join(run_dir, 'raw', 'Results_OPTIMIZATION.xlsx')
    else:
        path = os.path.join(run_dir, 'raw', 'Results_OPTIMIZATION_stage_1_single_scene_bench.xlsx')
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    scene_col = df.columns[0]
    df = df[df[scene_col].astype(str) == scene]
    if df.empty:
        return {}
    row = df.iloc[-1]
    out = {}
    for k in ('Rs_mean', 'ts_mean', 'our_repro', 'Convergence time', 'best_epoch'):
        if k in row.index and pd.notna(row[k]):
            out[k] = float(row[k])
    return out


def subset_gt(cams, gt):
    """Post-stage pruned runs keep only the largest connected camera component, so
    their predictions cover a subset of the scene's cameras. In that case use the
    GT the run itself stored (decomposed from the pruned data's Ps_gt); otherwise
    use the independently loaded full-scene GT."""
    if cams['Rs'].shape[0] == gt['Rs_gt'].shape[0]:
        return gt
    Ks = cams['Ks'].astype(np.float64)
    return {'Rs_gt': cams['Rs_gt'].astype(np.float64),
            'ts_gt': cams['ts_gt'].astype(np.float64),
            'Ks': Ks, 'Ns': np.linalg.inv(Ks)}


def harmonized_from_cameras(cams, gt):
    """Recompute all comparison metrics from raw predicted cameras (pre-BA)."""
    gt = subset_gt(cams, gt)
    Rs = cams['Rs'].astype(np.float64)
    ts = cams['ts'].astype(np.float64)
    Ps = cams['Ps'].astype(np.float64)
    pts3D = cams['pts3D_pred'].astype(np.float64)
    xs = cams['xs'].astype(np.float64)

    Rs_fixed, ts_fixed, _ = geo_utils.align_cameras(Rs, gt['Rs_gt'], ts, gt['ts_gt'],
                                                    return_alignment=True)
    R_err, t_err = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed,
                                                         gt['Rs_gt'], gt['ts_gt'])
    reproj = geo_utils.reprojection_error_with_points(Ps, pts3D.T, xs)
    return {
        'rot_err_deg': float(np.mean(R_err)),
        'rot_err_deg_med': float(np.median(R_err)),
        'pos_err': float(np.mean(t_err)),
        'pos_err_med': float(np.median(t_err)),
        'reproj_err_px': float(np.nanmean(reproj)),
        'reproj_med_px': float(np.nanmedian(reproj)),
        'n_cams': int(Rs.shape[0]),
    }


def shared_ba(cams, gt):
    """Apply ONE bundle adjustment implementation (this repo's pycolmap wrapper,
    no method-specific filtering) to raw cameras — identical for both methods."""
    import pycolmap
    from utils.ba_advanced import (batch_matrix_to_pycolmap, prepare_ba_options,
                                   pycolmap_to_batch_matrix)
    gt = subset_gt(cams, gt)
    xs_full = cams['xs'].astype(np.float64)
    Rs = cams['Rs'].astype(np.float64)
    ts = cams['ts'].astype(np.float64)
    pts3D = cams['pts3D_pred'].astype(np.float64).T  # (n, 4) homogeneous

    # keep points observed by >=2 cameras (BA needs multi-view constraints)
    visible = xs_full[:, :, 0] > 0
    keep = np.where(visible.sum(axis=0) >= 2)[0]
    xs = xs_full[:, keep]
    Xs = (pts3D[keep, :3] / pts3D[keep, 3:4])

    reconstruction = batch_matrix_to_pycolmap(xs, Rs, ts, gt['Ks'], Xs)
    ba_options = prepare_ba_options()
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    new_Rs, new_ts, new_Ps, _, new_Xs = pycolmap_to_batch_matrix(
        reconstruction, original_num_points=xs.shape[1])

    Rs_f, ts_f, _ = geo_utils.align_cameras(new_Rs, gt['Rs_gt'], new_ts, gt['ts_gt'],
                                            return_alignment=True)
    R_err, t_err = geo_utils.tranlsation_rotation_errors(Rs_f, ts_f, gt['Rs_gt'], gt['ts_gt'])
    reproj = geo_utils.reprojection_error_with_points(new_Ps, new_Xs, xs)
    return {
        'rot_err_deg_ba': float(np.mean(R_err)),
        'pos_err_ba': float(np.mean(t_err)),
        'reproj_err_px_ba': float(np.nanmean(reproj)),
        'reproj_med_px_ba': float(np.nanmedian(reproj)),
    }


def rel_diff(a, b):
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-root', default=os.path.join(CODE_DIR, 'results', 'single_scene'))
    ap.add_argument('--ba', action='store_true',
                    help='also run the shared BA on both methods (reported separately)')
    ap.add_argument('--flag-threshold', type=float, default=0.05,
                    help='relative native-vs-harmonized disagreement to flag (default 5%%)')
    ap.add_argument('--force', action='store_true', help='recompute even if json exists')
    args = ap.parse_args()

    meta_files = sorted(glob.glob(os.path.join(args.results_root, '*', '*', 'seed*', 'run_meta.json')))
    if not meta_files:
        sys.exit('No runs found under {}'.format(args.results_root))

    flags, done, skipped = [], 0, 0
    gt_cache = {}
    for meta_path in meta_files:
        run_dir = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('status') != 'completed':
            continue
        out_path = os.path.join(run_dir, 'harmonized_metrics.json')
        if os.path.exists(out_path) and not args.force:
            with open(out_path) as f:
                existing = json.load(f)
            if not args.ba or 'rot_err_deg_ba' in existing:
                skipped += 1
                continue

        method, scene, seed = meta['method'], meta['scene'], meta['seed']
        if scene not in gt_cache:
            gt_cache[scene] = load_gt(scene)
        gt = gt_cache[scene]

        cams_path = cameras_npz_path(run_dir, method, scene)
        cams = np.load(cams_path, allow_pickle=True)

        rec = {'method': method, 'scene': scene, 'seed': seed,
               'wall_clock_s': meta.get('wall_clock_s')}
        # Fail-soft: a run with degenerate final cameras (e.g. NaN rotations from a
        # diverged optimization) must not kill the evaluation of every other run.
        # Such runs get no harmonized_metrics.json and are reported loudly instead.
        try:
            rec.update(harmonized_from_cameras(cams, gt))
        except Exception as e:
            finite = bool(np.isfinite(cams['Rs']).all())
            flags.append('{} seed{} {}: HARMONIZED EVAL FAILED ({}); '
                         'finite rotations: {}'.format(method, seed, scene, e, finite))
            continue

        nat = native_metrics(run_dir, method, scene)
        rec['native'] = nat
        if 'Convergence time' in nat:
            rec['convergence_time_s'] = nat['Convergence time']
        if 'best_epoch' in nat:
            rec['best_epoch'] = nat['best_epoch']

        # Native-vs-harmonized cross-check (alignment-convention guard). The
        # relative check is paired with an absolute floor: on near-zero errors
        # (e.g. 0.02 deg) float32-vs-float64 alignment noise easily exceeds 5%
        # relatively while being physically meaningless — a real convention
        # mismatch produces differences far above these floors.
        abs_floor = {'rot_err_deg': 0.05, 'pos_err': 0.02, 'reproj_err_px': 0.1}
        for native_key, harm_key in (('Rs_mean', 'rot_err_deg'), ('ts_mean', 'pos_err'),
                                     ('our_repro', 'reproj_err_px')):
            if native_key in nat:
                d = rel_diff(nat[native_key], rec[harm_key])
                if (d is not None and d > args.flag_threshold
                        and abs(nat[native_key] - rec[harm_key]) > abs_floor[harm_key]):
                    flag = ('{} seed{} {}: {} native={:.4f} vs harmonized={:.4f} '
                            '({:.1f}% apart)').format(method, seed, scene, harm_key,
                                                      nat[native_key], rec[harm_key], 100 * d)
                    flags.append(flag)

        if args.ba:
            try:
                rec.update(shared_ba(cams, gt))
            except Exception as e:
                rec['ba_error'] = str(e)
                flags.append('{} seed{} {}: shared BA failed: {}'.format(method, seed, scene, e))

        with open(out_path, 'w') as f:
            json.dump(rec, f, indent=2)
        done += 1
        print('evaluated {} seed{} {}: rot={:.3f}deg pos={:.3f} reproj={:.3f}px n_cams={}'.format(
            method, seed, scene, rec['rot_err_deg'], rec['pos_err'], rec['reproj_err_px'],
            rec['n_cams']), flush=True)

    print('\n===== harmonized evaluation summary =====')
    print('evaluated: {}  already-done: {}'.format(done, skipped))
    if flags:
        print('\n' + '!' * 70)
        print('METRIC DISAGREEMENT FLAGS (>{}%) — resolve before using these numbers:'.format(
            int(args.flag_threshold * 100)))
        for fl in flags:
            print('  ' + fl)
        print('!' * 70)
        sys.exit(2)
    print('No native-vs-harmonized disagreements above threshold.')


if __name__ == '__main__':
    main()
