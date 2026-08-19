#!/usr/bin/env python
"""
ESFM two-stage postprocessing outlier handling (report Sec 2.2.1) on the Olsson
single-scene benchmark, at a budget that is a FAIR comparison with ESFM.

The report ran the two-stage framework at 10k (Stage 1) + 10k (Stage 2) on
MegaDepth, i.e. EACH stage got the full baseline budget (its baseline was also
10k). ESFM's single-scene protocol (our esfm/esfm_rc arms) is 100k epochs. Two
fair scalings against that baseline:

  design B (default, --stage1-source esfm_rc): each stage = full baseline budget,
    exactly as the report. Stage 1 is a full 100k ESFM reconstruction on all
    points -- which IS the esfm_rc baseline, so we REUSE it (no new Stage-1
    compute) and Stage 1 == the baseline we compare against. Stage 2 = 100k on the
    cleaned tracks. Total 200k, disclosed as 2x time exactly like the report's
    Table 5. Cleanest isolation of the outlier-handling effect.

  design A (--stage1-source fresh --stage1-epochs 50000): keep the 50/50 split but
    scale so TOTAL = 100k (50k + 50k), the stricter equal-total-budget comparison.

Per-stage schedule mirrors the baseline: milestones [0.5,0.7,0.9]x(stage epochs),
gamma 0.1, lr 1e-4, eval every 5000, best-by-reprojection.

Base architecture is the ESFM base (the esfm_rc arm: SetOfSetOutliersNet 1x3 with
output_mode=1 and ESFMLoss, no layer-norm/residual/dropout), so these runs are
"ESFM MAD/STD/Huber" exactly as in the report and compare directly to the esfm_rc
100k anchor.

Methods (report Sec 2.2.1):
  mad    remove obs with error > median + delta*1.4826*MAD, Stage 2 on inliers
  std    remove obs with error > mean   + delta*std,        Stage 2 on inliers
  huber  keep-weight min(1, delta/error), Stage 2 weights the reprojection loss

Stage 1 is shared across methods/deltas for a given (scene, seed) and cached under
results/single_scene/esfm_ts_s1/. Stage 2 lands as esfm_{mad,std,huber}_d{delta}
so the existing evaluator/aggregator pick it up unchanged.

Run on a GPU node, from code/:
  ../.venv/bin/python esfm_twostage.py --scene Gustav_Vasa [--methods mad,std,huber]
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import numpy as np
from pyhocon import ConfigFactory, HOCONConverter

from run_single_scene_sweep import OLSSON_SCENES, ESFM_RC_CONF, slug, gpu_info
from uesfm_poststage import write_pruned_npz   # reuse the pruning + connectivity logic

RESULTS_ROOT = os.path.join(CODE_DIR, 'results', 'single_scene')
STAGE1_METHOD = 'esfm_ts_s1'
DATASETS_EUC = os.path.join(os.path.dirname(CODE_DIR), 'datasets', 'Euclidean')

# Fair-vs-ESFM budget: 50k + 50k = 100k total (see module docstring).
STAGE1_EPOCHS = 50000
STAGE2_EPOCHS = 50000

METHODS = ('mad', 'std', 'huber')
# Report's reported-best deltas: MAD delta=2, STD delta=3, Huber delta=1. The
# report's sweep set is {0.5,1,1.5,2,2.5}; STD delta=3 is outside it (report Table
# 2 header), so we expose it too. Default here is the reported-best per method.
DEFAULT_DELTA = {'mad': 2.0, 'std': 3.0, 'huber': 1.0}


def milestones(epochs):
    return [int(epochs * 0.5), int(epochs * 0.7), int(epochs * 0.9)]


def dfmt(delta):
    """delta -> compact tag, e.g. 2.0 -> 'd2', 1.5 -> 'd15'."""
    return 'd' + (str(delta).replace('.', '').rstrip('0') or '0') if delta != int(delta) \
        else 'd{}'.format(int(delta))


def method_name(method, delta, label=''):
    base = 'esfm_{}_{}'.format(method, dfmt(delta))
    return '{}_{}'.format(base, label) if label else base


def run_dir_for(method_dir, scene, seed):
    return os.path.join(RESULTS_ROOT, method_dir, slug(scene), 'seed{}'.format(seed))


def cams_npz(raw_dir, scene):
    """RESfM-lineage artifact path (same as the esfm_rc / uesfm arms)."""
    return os.path.join(raw_dir, 'forFigures', '{}_Final_Cameras.npz'.format(scene))


def is_done(run_dir, scene):
    meta = os.path.join(run_dir, 'run_meta.json')
    if not os.path.exists(meta):
        return False
    with open(meta) as f:
        if json.load(f).get('status') != 'completed':
            return False
    return os.path.exists(cams_npz(os.path.join(run_dir, 'raw'), scene))


def base_conf(scene, seed, raw_dir, epochs):
    """ESFM-base conf (esfm_rc architecture) at a given epoch budget."""
    text = ESFM_RC_CONF.format(scene=scene, scene_slug=slug(scene), seed=seed,
                               raw_dir=raw_dir, epochs=epochs,
                               milestones=milestones(epochs), eval_intervals=5000)
    conf = ConfigFactory.parse_string(text)
    # Keys normally injected by general_utils.init_exp, which we bypass by calling
    # train() directly (same as uesfm_poststage.py). Without 'wandb' train() raises
    # ConfigMissingException at its WandB-logging block; 'resume' avoids a warning.
    conf.put('wandb', 0)
    conf.put('resume', False)
    conf.put('resuming_epoch', 0)   # train() overwrites this to -1 for a fresh run
    conf.put('exp_version', 'single_scene_twostage')
    return conf


def train_esfm_base(conf, scene, seed, proj_err_weight=None):
    """Run one ESFM-base optimization. If proj_err_weight ([m,n] outlier-ness in
    [0,1]) is given, use ESFMLoss_weighted_by_rep_err (Huber Stage 2); otherwise
    plain ESFMLoss. Writes RESfM-lineage artifacts + the stage_1 xlsx."""
    import torch
    from utils.Phases import Phases
    from utils import general_utils
    import train as train_mod
    from datasets import SceneData, ScenesDataSet
    from single_scene_optimization import initialize_fabric

    phase = Phases.OPTIMIZATION
    fabric = initialize_fabric(seed=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scene_data = SceneData.create_scene_data(conf, phase, stage=1)
    if proj_err_weight is not None:
        # Static per-observation weight for the Huber Stage 2. The loss computes
        # (1 - proj_err_weight) * reproj_err, so proj_err_weight is an outlier-ness
        # in [0,1]; we pass 1 - huber_keep_weight so the effective factor is the
        # Huber keep-weight itself.
        scene_data.proj_err_weight = torch.from_numpy(proj_err_weight).float()

    loader = torch.utils.data.DataLoader(
        ScenesDataSet.ScenesDataSet([scene_data], return_all=True),
        collate_fn=ScenesDataSet.collate_fn, num_workers=0, shuffle=False)

    model_class = general_utils.get_class('models.' + conf.get_string('model.type'))
    model = model_class(conf, phase).to(device)
    model = torch.compile(model, mode='reduce-overhead')

    train_stat, train_errors, _, _ = train_mod.train(conf, loader, model, phase, fabric=fabric)
    train_errors.drop('Mean', inplace=True)
    train_stat['Scene'] = train_errors.index
    train_stat.set_index('Scene', inplace=True)
    general_utils.write_results(conf, train_errors.join(train_stat),
                                file_name='Results_OPTIMIZATION_stage_1_single_scene_bench',
                                append=False)


def esfm_rc_stage1(scene, seed):
    """Reuse the existing esfm_rc 100k baseline run as Stage 1 (design B: each
    stage = full baseline budget). Stage 1 is then IDENTICAL to the baseline we
    compare against, so the only difference in Stage 2 is the cleaned tracks, and
    no new Stage-1 compute is spent. Returns the run dir or None if not available."""
    run_dir = run_dir_for('esfm_rc', scene, seed)
    if is_done(run_dir, scene):
        print('Stage1 REUSE esfm_rc baseline (100k): {} seed{}'.format(scene, seed), flush=True)
        return run_dir
    print('Stage1 esfm_rc baseline not found/complete for {} seed{}'.format(scene, seed), flush=True)
    return None


def ensure_stage1(scene, seed, force=False):
    """Run (or reuse) the shared 50k ESFM-base Stage 1 for this (scene, seed)."""
    run_dir = run_dir_for(STAGE1_METHOD, scene, seed)
    raw_dir = os.path.join(run_dir, 'raw')
    if not force and is_done(run_dir, scene):
        print('Stage1 SKIP (done): {} seed{}'.format(scene, seed), flush=True)
        return run_dir
    os.makedirs(raw_dir, exist_ok=True)
    meta = {'method': STAGE1_METHOD, 'scene': scene, 'seed': seed,
            'epochs': STAGE1_EPOCHS, 'host': socket.gethostname(), 'gpu': gpu_info(),
            'start': datetime.now().isoformat(timespec='seconds'), 'status': 'running'}
    with open(os.path.join(run_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    t0 = time.monotonic()
    try:
        conf = base_conf(scene, seed, raw_dir, STAGE1_EPOCHS)
        with open(os.path.join(run_dir, 'run.conf'), 'w') as f:
            f.write(HOCONConverter.convert(conf, 'hocon'))
        train_esfm_base(conf, scene, seed)
        meta['status'] = 'completed' if os.path.exists(cams_npz(raw_dir, scene)) else 'failed'
    except Exception as e:
        import traceback
        traceback.print_exc()
        meta['status'] = 'failed'
        meta['error'] = str(e)
    meta['end'] = datetime.now().isoformat(timespec='seconds')
    meta['wall_clock_s'] = round(time.monotonic() - t0, 2)
    with open(os.path.join(run_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Stage1 {}: {} seed{} ({}s)'.format(meta['status'].upper(), scene, seed,
                                              meta['wall_clock_s']), flush=True)
    return run_dir if meta['status'] == 'completed' else None


def stage1_reproj_errors(stage1_run_dir, scene):
    """Per-observation reprojection error [m,n] (NaN where not observed) from the
    Stage-1 best reconstruction."""
    from utils import geo_utils
    cams = np.load(cams_npz(os.path.join(stage1_run_dir, 'raw'), scene), allow_pickle=True)
    return geo_utils.reprojection_error_with_points(
        cams['Ps'].astype(np.float64), cams['pts3D_pred'].astype(np.float64).T,
        cams['xs'].astype(np.float64))


def detect(method, rep, delta):
    """Return either a removal mask [m,n] (mad/std) or a huber outlier-ness [m,n].
    rep is [m,n] with NaN at unobserved entries."""
    import torch
    from datasets.Euclidean import detect_outliers_statistical, compute_huber_weights
    valid = ~np.isnan(rep)
    errs = torch.from_numpy(rep[valid]).float()
    if method in ('mad', 'std'):
        flags = detect_outliers_statistical(errs, weight_method=method, alpha=delta)
        mask = np.zeros(rep.shape, dtype=np.float32)
        mask[valid] = flags.numpy()          # 1 = outlier -> removed in Stage 2
        return 'remove', mask
    # huber: keep-weight in (0,1]; store outlier-ness = 1 - keepweight so the loss
    # factor (1 - outlier-ness) equals the keep-weight. Unobserved -> 0 (no effect).
    keep = compute_huber_weights(errs, threshold=delta)
    ow = np.zeros(rep.shape, dtype=np.float32)
    ow[valid] = (1.0 - keep.numpy())
    return 'weight', ow


def run_stage2(scene, seed, method, delta, stage1_run_dir, stage1_epochs, label='', force=False):
    import torch
    mname = method_name(method, delta, label)
    run_dir = run_dir_for(mname, scene, seed)
    raw_dir = os.path.join(run_dir, 'raw')
    if not force and is_done(run_dir, scene):
        print('SKIP (done): {} seed{} {}'.format(mname, seed, scene), flush=True)
        return {'status': 'completed', 'skipped': True}
    os.makedirs(raw_dir, exist_ok=True)
    meta = {'method': mname, 'scene': scene, 'seed': seed, 'variant': method,
            'delta': delta, 'epochs': STAGE2_EPOCHS,
            'stage1_run': stage1_run_dir, 'stage1_epochs': stage1_epochs,
            'total_epochs': stage1_epochs + STAGE2_EPOCHS,
            'host': socket.gethostname(), 'gpu': gpu_info(),
            'start': datetime.now().isoformat(timespec='seconds'), 'status': 'running'}
    with open(os.path.join(run_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    t0 = time.monotonic()
    try:
        rep = stage1_reproj_errors(stage1_run_dir, scene)
        kind, arr = detect(method, rep, delta)

        conf = base_conf(scene, seed, raw_dir, STAGE2_EPOCHS)
        proj_err_weight = None
        if kind == 'remove':
            # Stage 2 on inliers only: prune the observations, keep the largest
            # connected camera component, train plain ESFMLoss on the pruned scene.
            dataset_name = os.path.join('single_scene_post',
                                        '{}_{}_seed{}'.format(mname, slug(scene), seed))
            kept, total, n_removed = write_pruned_npz(
                scene, arr, os.path.join(CODE_DIR, 'datasets', dataset_name))
            conf.put('dataset.dataset', dataset_name)
            meta.update({'cams_kept': kept, 'cams_total': total,
                         'observations_removed': n_removed})
        else:
            # Huber: full scene, weight the reprojection loss by the Huber keep-weight.
            conf.put('loss.func', 'ESFMLoss_weighted_by_rep_err')
            proj_err_weight = arr
            meta['mean_downweight'] = float(arr[~np.isnan(rep)].mean())

        with open(os.path.join(run_dir, 'run.conf'), 'w') as f:
            f.write(HOCONConverter.convert(conf, 'hocon'))
        train_esfm_base(conf, scene, seed, proj_err_weight=proj_err_weight)
        meta['status'] = 'completed' if os.path.exists(cams_npz(raw_dir, scene)) else 'failed'
    except Exception as e:
        import traceback
        traceback.print_exc()
        meta['status'] = 'failed'
        meta['error'] = str(e)
    meta['end'] = datetime.now().isoformat(timespec='seconds')
    meta['wall_clock_s'] = round(time.monotonic() - t0, 2)
    with open(os.path.join(run_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('{}: {} seed{} {} ({}s)'.format(meta['status'].upper(), mname, seed, scene,
                                          meta['wall_clock_s']), flush=True)
    return meta


def main():
    global STAGE1_EPOCHS, STAGE2_EPOCHS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help='scene name or slug')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--deltas', default='',
                    help='comma-separated deltas applied to ALL chosen methods '
                         '(sweep). Empty = each method\'s reported-best default '
                         '(mad=2, std=3, huber=1).')
    ap.add_argument('--stage1-source', choices=('esfm_rc', 'fresh'), default='esfm_rc',
                    help='esfm_rc (design B, default): reuse the existing esfm_rc 100k '
                         'baseline as Stage 1 (no new compute; Stage 1 == the baseline '
                         'we compare against). fresh (design A): run a new Stage 1 at '
                         '--stage1-epochs (use 50000 for the equal-total-budget 50k+50k).')
    ap.add_argument('--stage1-epochs', type=int, default=STAGE1_EPOCHS,
                    help='epochs for a FRESH Stage 1 (ignored when --stage1-source=esfm_rc)')
    ap.add_argument('--stage2-epochs', type=int, default=100000,
                    help='Stage 2 epochs (default 100000: full-budget refine, design B)')
    ap.add_argument('--label', default='',
                    help='suffix appended to the Stage-2 method name, so different '
                         'budget designs land in separate result dirs (e.g. "full" for '
                         'design B, "eqbud" for design A) instead of colliding.')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    STAGE1_EPOCHS, STAGE2_EPOCHS = args.stage1_epochs, args.stage2_epochs

    by_slug = {slug(s): s for s in OLSSON_SCENES}
    scene = by_slug.get(args.scene, args.scene)
    if scene not in OLSSON_SCENES:
        sys.exit('Unknown scene: {!r}'.format(args.scene))
    seeds = [int(s) for s in args.seeds.split(',') if s.strip() != '']
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    for m in methods:
        if m not in METHODS:
            sys.exit('Unknown method: {!r}'.format(m))
    deltas = [float(d) for d in args.deltas.split(',') if d.strip()]

    failures = 0
    for seed in seeds:
        if args.stage1_source == 'esfm_rc':
            s1, s1_epochs = esfm_rc_stage1(scene, seed), 100000
        else:
            s1, s1_epochs = ensure_stage1(scene, seed, force=args.force), STAGE1_EPOCHS
        if s1 is None:
            print('Stage1 unavailable for {} seed{}; skipping Stage2'.format(scene, seed))
            failures += 1
            continue
        for method in methods:
            ds = deltas if deltas else [DEFAULT_DELTA[method]]
            for delta in ds:
                meta = run_stage2(scene, seed, method, delta, s1, s1_epochs,
                                  label=args.label, force=args.force)
                failures += int(meta.get('status') != 'completed')
    print('esfm two-stage for {}: done ({} failures)'.format(scene, failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
