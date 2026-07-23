#!/usr/bin/env python
"""
U-ESFM single-scene POST-STAGE (completes the method's protocol on top of the
stage-1 CombinedLoss runs of SPEC_single_scene_experiments):

  variant ft_learned  prune tracks with the run's own learned outlier scores
                      (Final_outliers.npz > test.outliers_threshold), largest
                      connected camera component, then fine-tune with ESFMLoss
  variant ft_mad      same, but outliers from MAD (alpha=2) statistics of the
                      stage-1 reprojection errors (multiscene eval protocol)
  variant ttt         no pruning: test-time-training continuation with ESFMLoss
                      (like the multi-scene per-scene fine-tune)

Each post-run warm-starts from the stage-1 run's BEST checkpoint and re-optimizes
for --budgets epochs (default 1000 and 5000) at the multiscene per-scene
fine-tune lr (5e-3). Results land as new "methods" (uesfm_<variant>_<1k|5k>)
under results/single_scene/, so evaluate_single_scene.py / aggregate_single_scene.py
pick them up unchanged. Resumable and fail-soft like the main sweep.

Run on a GPU node, from code/:
  .venv/bin/python uesfm_poststage.py --scene Gustav_Vasa [--seeds 0,1,2]
"""
import argparse
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import numpy as np
from pyhocon import ConfigFactory, HOCONConverter

from run_single_scene_sweep import OLSSON_SCENES, slug, gpu_info

VARIANTS = ('ft_learned', 'ft_mad', 'ttt', 'ttt_comb')
BUDGET_TAG = {1000: '1k', 5000: '5k'}

UESFM_ROOT = os.path.join(CODE_DIR, 'results', 'single_scene', 'uesfm')
RESULTS_ROOT = os.path.join(CODE_DIR, 'results', 'single_scene')
POST_DATA_ROOT = os.path.join(CODE_DIR, 'datasets', 'single_scene_post')
DATASETS_EUC = os.path.join(os.path.dirname(CODE_DIR), 'datasets', 'Euclidean')


def method_name(variant, budget):
    return 'uesfm_{}_{}'.format(variant, BUDGET_TAG.get(budget, str(budget)))


def base_run_dir(scene, seed):
    return os.path.join(UESFM_ROOT, slug(scene), 'seed{}'.format(seed))


def best_checkpoint(base_raw, scene):
    """Best-epoch checkpoint of the stage-1 run (best_epoch from its xlsx)."""
    import pandas as pd
    xlsx = os.path.join(base_raw, 'Results_OPTIMIZATION_stage_1_single_scene_bench.xlsx')
    df = pd.read_excel(xlsx)
    row = df[df[df.columns[0]].astype(str) == scene].iloc[-1]
    best_ep = int(row['best_epoch'])
    path = os.path.join(base_raw, 'models', 'Model_Ep{}.pt'.format(best_ep))
    if not os.path.exists(path):
        # fall back to the latest saved checkpoint
        cands = sorted(os.listdir(os.path.join(base_raw, 'models')),
                       key=lambda f: int(''.join(c for c in f if c.isdigit()) or 0))
        path = os.path.join(base_raw, 'models', cands[-1])
    return path, best_ep


def outlier_mask(variant, base_raw, scene, thr):
    """[m, n] float mask of observations to remove (1 = outlier)."""
    if variant == 'ft_learned':
        d = np.load(os.path.join(base_raw, 'OPTIMIZATION', scene, 'outliers_results',
                                 'Final_outliers.npz'), allow_pickle=True)
        return (d['outliers_pred'] > thr).astype(np.float32)
    # ft_mad: MAD statistics on the stage-1 final reprojection errors
    import torch
    from utils import geo_utils
    from datasets.Euclidean import detect_outliers_statistical
    cams = np.load(os.path.join(base_raw, 'forFigures', '{}_Final_Cameras.npz'.format(scene)),
                   allow_pickle=True)
    rep = geo_utils.reprojection_error_with_points(
        cams['Ps'].astype(np.float64), cams['pts3D_pred'].astype(np.float64).T,
        cams['xs'].astype(np.float64))
    valid = ~np.isnan(rep)
    flags = detect_outliers_statistical(torch.from_numpy(rep[valid]).float(),
                                        weight_method='mad', alpha=2.0)
    mask = np.zeros(rep.shape, dtype=np.float32)
    mask[valid] = flags.numpy()
    return mask


def write_pruned_npz(scene, mask, out_dir):
    """Zero masked observations, keep largest connected camera component, write npz."""
    import torch
    from utils import dataset_utils
    src = np.load(os.path.join(DATASETS_EUC, scene + '.npz'), allow_pickle=True)
    data = {k: src[k] for k in src.files}
    M = data['M'].astype(np.float64).copy()          # [2m, n]
    m = M.shape[0] // 2
    rm = mask > 0
    M[0::2][rm] = 0
    M[1::2][rm] = 0
    _, valid_cams = dataset_utils.check_if_M_connected(
        torch.from_numpy(M), thr=1, return_largest_component=True)
    valid_cams = sorted(int(c) for c in valid_cams)
    rows = [j for c in valid_cams for j in (2 * c, 2 * c + 1)]
    data['M'] = M[rows]
    for key in ('Ps_gt', 'Ns', 'K_gt', 'R_gt', 'T_gt'):
        if key in data and getattr(data[key], 'shape', ()) and data[key].shape[0] == m:
            data[key] = data[key][valid_cams]
    if 'namesList' in data and getattr(data['namesList'], 'shape', ()) and data['namesList'].shape[0] == m:
        data['namesList'] = data['namesList'][valid_cams]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, scene + '.npz'), **data)
    return len(valid_cams), m, int(rm.sum())


def build_conf(base_conf_path, scene, seed, variant, budget, raw_dir, dataset_name):
    conf = ConfigFactory.parse_file(base_conf_path)
    conf.put('exp_name', 'ss_post_{}_{}_seed{}'.format(method_name(variant, budget),
                                                       slug(scene), seed))
    conf.put('results_path', raw_dir)
    conf.put('dataset.dataset', dataset_name)
    # fine-tune objective and schedule: multiscene per-scene fine-tune protocol.
    # ttt_comb keeps the base run's adaptive CombinedLoss (and the output_mode-3
    # outlier head it needs) — the multiscene TTT 'comb' variant; every other
    # variant continues with plain ESFMLoss ('reproj_only').
    if variant != 'ttt_comb':
        conf.put('loss.func', 'ESFMLoss')
        conf.put('train.output_mode', 1)
    conf.put('train.num_epochs', budget)
    conf.put('train.eval_intervals', 250)
    conf.put('train.lr', 5e-3)
    conf.put('train.scheduler_milestone', [])
    conf.put('train.early_stopping_patience', 0)
    conf.put('train.extract_reproj_errors', False)
    # keys normally injected by general_utils.init_exp (bypassed here)
    conf.put('wandb', 0)
    conf.put('exp_version', 'single_scene_post')
    conf.put('resume', False)
    conf.put('resuming_epoch', 0)
    return conf


def run_one(scene, seed, variant, budget, force=False):
    import torch
    method = method_name(variant, budget)
    run_dir = os.path.join(RESULTS_ROOT, method, slug(scene), 'seed{}'.format(seed))
    raw_dir = os.path.join(run_dir, 'raw')
    cams_path = os.path.join(raw_dir, 'forFigures', '{}_Final_Cameras.npz'.format(scene))
    meta_path = os.path.join(run_dir, 'run_meta.json')
    if not force and os.path.exists(meta_path):
        with open(meta_path) as f:
            if json.load(f).get('status') == 'completed' and os.path.exists(cams_path):
                print('SKIP (done): {} seed{} {}'.format(method, seed, scene), flush=True)
                return {'status': 'completed', 'skipped': True}
    os.makedirs(raw_dir, exist_ok=True)

    base = base_run_dir(scene, seed)
    base_raw = os.path.join(base, 'raw')

    meta = {'method': method, 'scene': scene, 'seed': seed, 'variant': variant,
            'epochs': budget, 'base_run': base,
            'host': socket.gethostname(), 'gpu': gpu_info(),
            'start': datetime.now().isoformat(timespec='seconds'), 'status': 'running'}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    t0 = time.monotonic()
    try:
        # dataset: pruned copy (per seed — masks are run-specific) or the original
        if variant in ('ttt', 'ttt_comb'):
            dataset_name = 'Euclidean'
        else:
            dataset_name = os.path.join('single_scene_post',
                                        '{}_{}_seed{}'.format(method, slug(scene), seed))
            base_conf = ConfigFactory.parse_file(os.path.join(base, 'run.conf'))
            thr = base_conf.get_float('test.outliers_threshold', default=0.6)
            mask = outlier_mask(variant, base_raw, scene, thr)
            kept, total, n_removed = write_pruned_npz(
                scene, mask, os.path.join(CODE_DIR, 'datasets', dataset_name))
            meta['cams_kept'] = kept
            meta['cams_total'] = total
            meta['observations_removed'] = n_removed

        conf = build_conf(os.path.join(base, 'run.conf'), scene, seed, variant,
                          budget, raw_dir, dataset_name)
        with open(os.path.join(run_dir, 'run.conf'), 'w') as f:
            f.write(HOCONConverter.convert(conf, 'hocon'))

        from utils.Phases import Phases
        from utils import general_utils
        import train as train_mod
        from datasets import SceneData, ScenesDataSet
        from single_scene_optimization import initialize_fabric

        phase = Phases.OPTIMIZATION
        fabric = initialize_fabric(seed=seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        scene_data = SceneData.create_scene_data(conf, phase, stage=1)
        loader = torch.utils.data.DataLoader(
            ScenesDataSet.ScenesDataSet([scene_data], return_all=True),
            collate_fn=ScenesDataSet.collate_fn, num_workers=0, shuffle=False)

        model_class = general_utils.get_class('models.' + conf.get_string('model.type'))
        model = model_class(conf, phase).to(device)
        ckpt_path, best_ep = best_checkpoint(base_raw, scene)
        state = torch.load(ckpt_path, map_location=device)['model_state_dict']
        # stage-1 checkpoints were saved from a compiled model (_orig_mod prefix)
        state = { (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
                  for k, v in state.items() }
        model.load_state_dict(state)
        meta['warm_start'] = {'checkpoint': ckpt_path, 'epoch': best_ep}
        model = torch.compile(model, mode='reduce-overhead')

        train_stat, train_errors, _, _ = train_mod.train(conf, loader, model, phase,
                                                         fabric=fabric)
        train_errors.drop('Mean', inplace=True)
        train_stat['Scene'] = train_errors.index
        train_stat.set_index('Scene', inplace=True)
        general_utils.write_results(conf, train_errors.join(train_stat),
                                    file_name='Results_OPTIMIZATION_stage_1_single_scene_bench',
                                    append=False)
        ok = os.path.exists(cams_path)
        meta['status'] = 'completed' if ok else 'failed'
    except Exception as e:
        import traceback
        traceback.print_exc()
        meta['status'] = 'failed'
        meta['error'] = str(e)
    meta['end'] = datetime.now().isoformat(timespec='seconds')
    meta['wall_clock_s'] = round(time.monotonic() - t0, 2)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print('{}: {} seed{} {} ({}s)'.format(meta['status'].upper(), method, seed, scene,
                                          meta['wall_clock_s']), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help='scene name or slug')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--budgets', default='1000,5000')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    by_slug = {slug(s): s for s in OLSSON_SCENES}
    scene = by_slug.get(args.scene, args.scene)
    if scene not in OLSSON_SCENES:
        sys.exit('Unknown scene: {!r}'.format(args.scene))
    seeds = [int(s) for s in args.seeds.split(',') if s.strip() != '']
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    budgets = [int(b) for b in args.budgets.split(',') if b.strip()]
    for v in variants:
        if v not in VARIANTS:
            sys.exit('Unknown variant: {!r}'.format(v))

    failures = 0
    for seed in seeds:
        for budget in budgets:
            for variant in variants:
                meta = run_one(scene, seed, variant, budget, force=args.force)
                failures += int(meta.get('status') != 'completed')
    print('post-stage for {}: done ({} failures)'.format(scene, failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
