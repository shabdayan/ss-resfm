"""
Aggregate multi-seed evaluation results: per-scene MEDIAN over seeds of
Nr / Rot / Trans, reported next to the single-seed (seed 20, paper-protocol)
run and the RESfM paper's Table 1.

Usage:
  python aggregate_seed_results.py \
      --root_base results/multiscene/resfm_repro_env38 \
      --seeds 20,21,22,23,24 \
      [--results_file Results_FINE_TUNE_stage_1_esfm_outliers.xlsx]

Roots are resolved as <root_base> for seed 20 and <root_base>_seed<S> otherwise
(the layout produced by run_resfm_repro.sh / run_multiscene_eval.sh --seed).
"""
import argparse
import os
import pandas as pd

from compare_repro_to_paper import PAPER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_base', required=True)
    parser.add_argument('--seeds', default='20,21,22,23,24')
    parser.add_argument('--results_file', default='Results_FINE_TUNE_stage_1_esfm_outliers.xlsx')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    rows = []
    for scan, paper in PAPER.items():
        per_seed = {}
        for s in seeds:
            root = args.root_base if s == 20 else f'{args.root_base}_seed{s}'
            path = os.path.join(root, f'{scan}_ba', args.results_file)
            if os.path.exists(path):
                r = pd.read_excel(path).iloc[0]
                per_seed[s] = (int(r['#registered_cams_final']),
                               float(r['Rs_ba_final_mean']),
                               float(r['ts_ba_final_mean']))
        if not per_seed:
            continue
        df = pd.DataFrame(per_seed.values(), columns=['Nr', 'Rot', 'Trans'])
        row = {'Scene': scan, 'n_seeds': len(per_seed),
               'Nr_paper': paper['Nr'], 'Rot_paper': paper['Rot'], 'Trans_paper': paper['Trans'],
               'Nr_med': df['Nr'].median(), 'Rot_med': round(df['Rot'].median(), 2),
               'Trans_med': round(df['Trans'].median(), 3),
               'Rot_iqr': round(df['Rot'].quantile(0.75) - df['Rot'].quantile(0.25), 2)}
        if 20 in per_seed:
            row['Nr_s20'], row['Rot_s20'], row['Trans_s20'] = per_seed[20][0], round(per_seed[20][1], 2), round(per_seed[20][2], 3)
        rows.append(row)

    out = pd.DataFrame(rows).set_index('Scene')
    print(out.to_string())
    print('\nMeans over scenes:')
    cols = [c for c in ['Nr_paper', 'Nr_s20', 'Nr_med', 'Rot_paper', 'Rot_s20', 'Rot_med',
                        'Trans_paper', 'Trans_s20', 'Trans_med'] if c in out.columns]
    print(out[cols].mean().round(3).to_string())
    dest = f'{args.root_base}_seed_aggregate.xlsx'
    out.to_excel(dest)
    print(f'\nSaved: {dest}')


if __name__ == '__main__':
    main()
