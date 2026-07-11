"""
Assemble the paper's Table 1 as one tidy CSV (SPEC_uesfm_combined R13/R14):
our measured rows merged with the checked-in published transcription. Every row
carries a `source` column (ours / published) — protocols are never mixed silently.

Row roles (R13):
  - "U-ESFM (ours)"        source=ours       our method, our pipeline
  - "RESfM (reproduced)"   source=ours       released .pth through OUR pipeline —
                                             the bolding/delta target
  - published methods      source=published  reference/calibration only, from
                                             data/published/resfm_table1.csv

One row per (scene, method, seed); published rows have seed empty. Pivot/median
for presentation happens downstream (aggregate_seed_results.py / paper scripts).

Usage:
  python assemble_table1.py \
      [--uesfm_root results/multiscene/uesfm_stage1_eval] [--uesfm_seeds 20,21,22,23,24] \
      [--resfm_root results/multiscene/resfm_repro_env38] [--resfm_seeds 20,21,22,23,24] \
      [--out results/multiscene/table1_assembled.csv]
"""
import argparse
import glob
import os
import pandas as pd

from utils.experiment_guard import git_hash, TEST_SCENES_MEGADEPTH

PUBLISHED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'published', 'resfm_table1.csv')

# Columns copied from each run's Results_FINE_TUNE*.xlsx when present
# (R16 timing columns exist only in runs after the instrumentation commit).
XLSX_COLUMNS = {'#registered_cams_final': 'Nr', 'Rs_ba_final_mean': 'Rot',
                'ts_ba_final_mean': 'Trans', 'our_repro': 'our_repro',
                'Convergence time': 'convergence_seconds',
                'inference_seconds': 'inference_seconds',
                'ba_seconds': 'ba_seconds',
                'peak_gpu_mem_gb': 'peak_gpu_mem_gb',
                'param_count': 'param_count'}


def collect(root_base, seeds, method, loss_variant):
    rows = []
    for s in seeds:
        root = root_base if s == 20 else f'{root_base}_seed{s}'
        for scan in TEST_SCENES_MEGADEPTH:
            run_dir = os.path.join(root, f'{scan}_ba')
            hits = glob.glob(os.path.join(run_dir, 'Results_FINE_TUNE*.xlsx'))
            if not hits:
                continue
            r = pd.read_excel(hits[0]).iloc[0]
            row = {'scene': scan, 'dataset': 'megadepth', 'method': method,
                   'seed': s, 'loss_variant': loss_variant, 'source': 'ours',
                   'git': git_hash(), 'run_dir': run_dir}
            for src, dst in XLSX_COLUMNS.items():
                if src in r.index:
                    row[dst] = r[src]
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uesfm_root', default='results/multiscene/uesfm_stage1_eval')
    parser.add_argument('--uesfm_seeds', default='20,21,22,23,24')
    parser.add_argument('--resfm_root', default='results/multiscene/resfm_repro_env38')
    parser.add_argument('--resfm_seeds', default='20,21,22,23,24')
    parser.add_argument('--out', default='results/multiscene/table1_assembled.csv')
    args = parser.parse_args()

    published = pd.read_csv(PUBLISHED_CSV, comment='#',
                            dtype={'scene': str})
    ours = []
    # Both eval chains fine-tune with plain reprojection (protocol); U-ESFM's
    # method-specific step is the MAD outlier removal before the fine-tune.
    ours += collect(args.uesfm_root, [int(s) for s in args.uesfm_seeds.split(',')],
                    'U-ESFM (ours)', 'mad_prune+ESFMLoss_ft')
    ours += collect(args.resfm_root, [int(s) for s in args.resfm_seeds.split(',')],
                    'RESfM (reproduced)', 'classifier_prune+ESFMLoss_ft')
    ours = pd.DataFrame(ours)
    if ours.empty:
        print('WARNING: no measured rows found — check --uesfm_root/--resfm_root.')
    else:
        # scene metadata (Nc, outlier fraction) comes from the published table
        meta = published[['scene', 'Nc', 'outlier_pct']].drop_duplicates('scene')
        ours = ours.merge(meta, on='scene', how='left')

    table = pd.concat([ours, published], ignore_index=True)
    table.to_csv(args.out, index=False)
    n_ours = 0 if ours.empty else len(ours)
    print(f'{n_ours} measured rows + {len(published)} published rows -> {args.out}')
    if not ours.empty:
        print('\nPer-method scene coverage (ours):')
        print(ours.groupby(['method', 'seed'])['scene'].count().to_string())


if __name__ == '__main__':
    main()
