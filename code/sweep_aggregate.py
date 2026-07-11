"""
Aggregate R12 loss-component sweep results (SPEC_uesfm_combined §2.C) into one
tidy CSV per sweep: config columns + metric columns, one row per (value, scene).

Reads results/multiscene/sweeps/<sweep>/<tag>/<scan>_ba/Results_FINE_TUNE*.xlsx
(the layout produced by run_sweeps.sh).

Usage: python sweep_aggregate.py --sweep percentiles [--root results/multiscene/sweeps] [--seed 20]
"""
import argparse
import glob
import os
import pandas as pd

from utils.experiment_guard import git_hash

# tag suffix -> config columns (tag format: <sweep>_<value>, ':' already '_')
PARAM_COLUMNS = {
    'percentiles': lambda v: {'inlier_percentile': float(v.split('_')[0]),
                              'outlier_percentile': float(v.split('_')[1])},
    'warmup': lambda v: {'warmup_epochs': int(v)},
    'beta': lambda v: {'classification_loss_weight': float(v)},
    'removal_threshold': lambda v: {'outliers_threshold': float(v)},
    'mad_alpha': lambda v: {'mad_alpha': float(v)},
}

METRICS = {'our_repro': 'our_repro', 'repro_ba_final': 'repro_ba_final',
           '#registered_cams_final': 'Nr', 'Rs_ba_final_mean': 'Rot',
           'ts_ba_final_mean': 'Trans', 'Convergence time': 'convergence_seconds'}

# Loss sweeps fine-tune with the full CombinedLoss; removal sweeps keep the
# protocol's plain-reprojection fine-tune (must match run_sweeps.sh).
LOSS_VARIANT = {'percentiles': 'CombinedLoss', 'warmup': 'CombinedLoss',
                'beta': 'CombinedLoss', 'removal_threshold': 'ESFMLoss',
                'mad_alpha': 'ESFMLoss'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', required=True, choices=sorted(PARAM_COLUMNS))
    parser.add_argument('--root', default='results/multiscene/sweeps')
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(
            args.root, args.sweep, f'{args.sweep}_*', '*_ba', 'Results_FINE_TUNE*.xlsx'))):
        run_dir = os.path.dirname(f)
        scan = os.path.basename(run_dir).replace('_ba', '')
        value = os.path.basename(os.path.dirname(run_dir))[len(args.sweep) + 1:]
        row = {'sweep': args.sweep, 'scan': scan, 'dataset': 'megadepth',
               'loss_variant': LOSS_VARIANT[args.sweep], 'seed': args.seed,
               'git': git_hash(), 'source': 'ours',
               'conf_snapshot': os.path.join(
                   'confs/sweeps_generated', args.sweep,
                   f'{scan}_{args.sweep}_{value}_seed{args.seed}.conf')}
        row.update(PARAM_COLUMNS[args.sweep](value))
        r = pd.read_excel(f).iloc[0]
        for src, dst in METRICS.items():
            row[dst] = r.get(src)
        rows.append(row)

    if not rows:
        print('No Results_FINE_TUNE*.xlsx found under',
              os.path.join(args.root, args.sweep))
        return
    df = pd.DataFrame(rows)
    out = os.path.join(args.root, f'sweep_{args.sweep}_seed{args.seed}.csv')
    df.to_csv(out, index=False)
    print(df.to_string())
    print('\nWrote:', out)


if __name__ == '__main__':
    main()
