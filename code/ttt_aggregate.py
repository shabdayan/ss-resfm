"""
Aggregate TTT results (SPEC_cvpr_experiments R3) into:
  (a) a Table-2-shaped CSV: scene rows x {frozen, reproj_only, comb} x {Rot, Trans, Nr}
  (b) a tidy step-vs-error CSV for the curve figure, with per-scene outlier%.

Reads ttt_snapshots.csv from results/multiscene/ttt/<variant>_seed<S>/<scan>_ba/.
Frozen = the step-0 row (post-BA) of the reproj_only variant (identical to comb at
step 0 by construction; falls back to comb if reproj_only is absent).

Usage: python ttt_aggregate.py [--root results/multiscene/ttt] [--seed 20]
"""
import argparse
import glob
import os
import pandas as pd

from utils.experiment_guard import git_hash

# Scene outlier fractions (resfm.pdf Table 1, extracted programmatically)
OUTLIER_PCT = {
    "0238": 44.6, "0060": 41.6, "0197": 40.7, "0094": 40.1, "0265": 38.8,
    "0083": 31.3, "0076": 30.5, "0185": 30.0, "0048": 24.2, "0024": 23.0,
    "0223": 17.0, "5016": 0.2, "0046": 14.6, "0099": 47.4, "1001": 43.9,
    "0231": 42.2, "0411": 29.9, "0377": 27.5, "0102": 25.8, "0147": 24.6,
    "0148": 24.6, "0446": 22.1, "0022": 21.2, "0327": 21.0, "0015": 20.6,
    "0455": 19.8, "0496": 19.2, "1589": 17.4, "0012": 16.3, "0104": 16.2,
    "0019": 15.4, "0063": 14.5, "0130": 14.4, "0080": 12.9, "0240": 11.9,
    "0007": 11.7,
}

METRIC_MAP = {'Rot': 'Rs_ba_final_mean', 'Trans': 'ts_ba_final_mean', 'Nr': '#registered_cams_final'}


def load_variant(root, variant, seed):
    rows = []
    for f in glob.glob(os.path.join(root, f'{variant}_seed{seed}', '*_ba', 'ttt_snapshots.csv')):
        scan = os.path.basename(os.path.dirname(f)).replace('_ba', '')
        df = pd.read_csv(f)
        df['scan'] = scan
        df['variant'] = variant
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/multiscene/ttt')
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    frames = {v: load_variant(args.root, v, args.seed) for v in ['reproj_only', 'comb']}
    all_rows = pd.concat([d for d in frames.values() if not d.empty], ignore_index=True)
    if all_rows.empty:
        print('No ttt_snapshots.csv found under', args.root)
        return

    # (b) tidy step-curve CSV
    curve = all_rows.copy()
    curve['outlier_pct'] = curve['scan'].map(OUTLIER_PCT)
    curve['source'] = 'ours'
    curve['git'] = git_hash()
    curve_path = os.path.join(args.root, f'ttt_step_curve_seed{args.seed}.csv')
    curve.to_csv(curve_path, index=False)

    # (a) Table-2-shaped CSV from post-BA rows
    frozen_src = frames['reproj_only'] if not frames['reproj_only'].empty else frames['comb']
    table = {}
    for scan in sorted(all_rows['scan'].unique()):
        row = {'outlier_pct': OUTLIER_PCT.get(scan)}
        fz = frozen_src[(frozen_src['scan'] == scan) & (frozen_src['step'] == 0)]
        for col, key in METRIC_MAP.items():
            row[f'frozen_{col}'] = fz.iloc[0].get(key) if len(fz) else None
        for variant in ['reproj_only', 'comb']:
            d = frames[variant]
            if d.empty:
                continue
            last = d[(d['scan'] == scan) & (d['ba_mode'] == 'post_ba')]
            last = last[last['step'] == last['step'].max()]
            for col, key in METRIC_MAP.items():
                row[f'{variant}_{col}'] = last.iloc[0].get(key) if len(last) else None
        table[scan] = row
    t2 = pd.DataFrame(table).T
    t2.index.name = 'Scene'
    t2['source'] = 'ours'
    t2['git'] = git_hash()
    t2['seed'] = args.seed
    t2_path = os.path.join(args.root, f'ttt_table2_seed{args.seed}.csv')
    t2.to_csv(t2_path)

    print(t2.to_string())
    print('\nWrote:', t2_path, 'and', curve_path)


if __name__ == '__main__':
    main()
