#!/usr/bin/env python
"""
Phase 3 of SPEC_single_scene_experiments.md: aggregate harmonized per-run metrics
(produced by evaluate_single_scene.py) into:

  results/single_scene/summary.csv        method, scene, seed, rot_err_deg, pos_err,
                                          reproj_err_px, n_cams, runtime_s (+ post-BA
                                          columns when present)
  results/single_scene/summary_table.md   one row per scene, method-grouped columns,
                                          mean over seeds, best value per scene bold
  results/single_scene/summary_table.tex  booktabs LaTeX version
  results/single_scene/REPRO.md           reproducibility record (git commits/diffs,
                                          env export, GPU, commands, budgets, seeds,
                                          dataset checksums)

Also prints a short analysis (per-scene wins/losses, overall means, failures).
Run from u-esfm/code:  .venv/bin/python aggregate_single_scene.py
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
UESFM_REPO = os.path.dirname(CODE_DIR)
PROJECT_ROOT = os.path.dirname(UESFM_REPO)
ESFM_REPO = os.path.join(PROJECT_ROOT, 'esfm-baseline')

METHOD_LABELS = {'esfm': 'ESFM (official code)', 'esfm_rc': 'ESFM (RESfM code)',
                 'uesfm': 'U-ESFM', 'uesfm_abl': 'U-ESFM arch + ESFMLoss',
                 'esfm_paper': 'ESFM (paper)',
                 # post-stage completions of the U-ESFM protocol (uesfm_poststage.py)
                 'uesfm_ft_learned_1k': 'U-ESFM +prune(learned)+FT1k',
                 'uesfm_ft_learned_5k': 'U-ESFM +prune(learned)+FT5k',
                 'uesfm_ft_mad_1k': 'U-ESFM +prune(MAD)+FT1k',
                 'uesfm_ft_mad_5k': 'U-ESFM +prune(MAD)+FT5k',
                 'uesfm_ttt_1k': 'U-ESFM +TTT1k',
                 'uesfm_ttt_5k': 'U-ESFM +TTT5k',
                 'uesfm_ttt_comb_1k': 'U-ESFM +TTT(adaptive)1k',
                 'uesfm_ttt_comb_5k': 'U-ESFM +TTT(adaptive)5k',
                 # pruning-threshold sweep (learned scores, FT5k)
                 'uesfm_ft_learned_t3_5k': 'U-ESFM +prune(learned .3)+FT5k',
                 'uesfm_ft_learned_t5_5k': 'U-ESFM +prune(learned .5)+FT5k',
                 'uesfm_ft_learned_t7_5k': 'U-ESFM +prune(learned .7)+FT5k',
                 'uesfm_ft_learned_t9_5k': 'U-ESFM +prune(learned .9)+FT5k',
                 # adaptive-loss percentile probe (6-scene subset)
                 'uesfm_p1090': 'U-ESFM pct 10/90',
                 'uesfm_p3070': 'U-ESFM pct 30/70',
                 'uesfm_p4060': 'U-ESFM pct 40/60',
                 # report-faithful weighted-reprojection CombinedLoss (fresh training)
                 'uesfm_wtd': 'U-ESFM weighted',
                 'uesfm_wtddet': 'U-ESFM weighted(detach)',
                 'uesfm_wtd_smoke': 'U-ESFM weighted [smoke]',
                 'uesfm_wtddet_smoke': 'U-ESFM weighted-detach [smoke]',
                 # second stage on stage-1 detector: remove|weight x continue|scratch
                 'uesfm_remove_continue_100k': 'U-ESFM remove+continue100k',
                 'uesfm_remove_scratch_100k': 'U-ESFM remove+scratch100k',
                 'uesfm_weight_continue_100k': 'U-ESFM weight+continue100k',
                 'uesfm_weight_scratch_100k': 'U-ESFM weight+scratch100k'}

# Published per-scene numbers from Moran et al., ICCV 2021 (supplementary calibrated
# table: "Ours" columns, before-BA and after-BA). Reference only: their post-BA uses
# the paper's own ceres BA, not this benchmark's shared BA, so these columns are shown
# for context and never compete for bold-best.
PUBLISHED_CSV = os.path.join(CODE_DIR, 'data', 'published', 'esfm_olsson_calibrated.csv')
BASE_METRICS = [('rot_err_deg', 'Rot (deg)', True),
                ('pos_err', 'Trans', True),
                ('reproj_err_px', 'Reproj (px)', True),
                ('n_cams', 'Nr', False),
                ('runtime_s', 'Time (s)', True)]
BA_METRICS = [('rot_err_deg_ba', 'Rot-BA (deg)', True),
              ('pos_err_ba', 'Trans-BA', True),
              ('reproj_err_px_ba', 'Reproj-BA (px)', True)]


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                              text=True, timeout=120).stdout.strip()
    except Exception as e:
        return 'ERROR: {}'.format(e)


def collect(results_root):
    rows, failures = [], []
    for meta_path in sorted(glob.glob(os.path.join(results_root, '*', '*', 'seed*', 'run_meta.json'))):
        run_dir = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('status') != 'completed':
            failures.append('{} seed{} {}: status={} exit={}'.format(
                meta.get('method'), meta.get('seed'), meta.get('scene'),
                meta.get('status'), meta.get('exit_code')))
            continue
        hm_path = os.path.join(run_dir, 'harmonized_metrics.json')
        if not os.path.exists(hm_path):
            failures.append('{} seed{} {}: completed but not evaluated '
                            '(run evaluate_single_scene.py)'.format(
                                meta.get('method'), meta.get('seed'), meta.get('scene')))
            continue
        with open(hm_path) as f:
            rec = json.load(f)
        row = {
            'method': rec['method'], 'scene': rec['scene'], 'seed': rec['seed'],
            'rot_err_deg': rec['rot_err_deg'], 'pos_err': rec['pos_err'],
            'reproj_err_px': rec['reproj_err_px'], 'reproj_med_px': rec.get('reproj_med_px'),
            'n_cams': rec['n_cams'], 'runtime_s': rec.get('wall_clock_s'),
            'convergence_time_s': rec.get('convergence_time_s'),
            'best_epoch': rec.get('best_epoch'),
        }
        for k, _, _ in BA_METRICS:
            if k in rec:
                row[k] = rec[k]
        rows.append(row)
    return pd.DataFrame(rows), failures


def build_scene_table(df, metrics):
    """Mean over seeds -> per (scene, method); returns pivoted DataFrame."""
    agg = df.groupby(['scene', 'method']).agg(
        {k: 'mean' for k, _, _ in metrics}).reset_index()
    return agg


def fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1:
        return '{:.0f}'.format(v)
    return '{:.{d}f}'.format(v, d=decimals)


def load_published(scenes):
    """Published ESFM numbers as pseudo-method rows (or empty df if csv absent)."""
    if not os.path.exists(PUBLISHED_CSV):
        return pd.DataFrame()
    pub = pd.read_csv(PUBLISHED_CSV)
    pub = pub[pub['scene'].isin(scenes)]
    rows = []
    for _, r in pub.iterrows():
        rows.append({
            'scene': r['scene'], 'method': 'esfm_paper',
            'rot_err_deg': r['rot_err_deg_paper'], 'pos_err': r['pos_err_paper'],
            'reproj_err_px': r['reproj_err_px_paper'],
            'rot_err_deg_ba': r['rot_err_deg_ba_paper'], 'pos_err_ba': r['pos_err_ba_paper'],
            'reproj_err_px_ba': r['reproj_err_px_ba_paper'],
        })
    return pd.DataFrame(rows)


def write_tables(agg, methods, metrics, out_md, out_tex, seeds_per_cell, bold_methods=None):
    scenes = sorted(agg['scene'].unique())
    label = {m: METHOD_LABELS.get(m, m) for m in methods}
    if bold_methods is None:
        bold_methods = set(methods)

    # ---------- markdown ----------
    md = []
    md.append('# Single-scene comparison — Olsson dataset (calibrated)\n')
    md.append('Mean over seeds per (scene, method); best value per scene in **bold** '
              '(lower is better except Nr).\n')
    header = ['Scene']
    for key, name, _ in metrics:
        for m in methods:
            header.append('{} {}'.format(label[m], name))
    md.append('| ' + ' | '.join(header) + ' |')
    md.append('|' + '---|' * len(header))

    def cell(scene, method, key):
        sel = agg[(agg['scene'] == scene) & (agg['method'] == method)]
        return None if sel.empty else sel.iloc[0][key]

    for scene in scenes:
        row = [scene]
        for key, _, lower_better in metrics:
            vals = {m: cell(scene, m, key) for m in methods}
            valid = {m: v for m, v in vals.items()
                     if m in bold_methods and v is not None and not np.isnan(v)}
            best = None
            if len(valid) > 1:
                best = (min if lower_better else max)(valid, key=lambda m: valid[m])
            for m in methods:
                s = fmt(vals.get(m))
                if best is not None and m == best:
                    s = '**{}**'.format(s)
                row.append(s)
        md.append('| ' + ' | '.join(row) + ' |')

    # mean row
    row = ['**Mean**']
    for key, _, lower_better in metrics:
        means = {m: agg[agg['method'] == m][key].mean() for m in methods}
        valid = {m: v for m, v in means.items()
                 if m in bold_methods and not np.isnan(v)}
        best = (min if lower_better else max)(valid, key=lambda m: valid[m]) if len(valid) > 1 else None
        for m in methods:
            s = fmt(means.get(m))
            if best is not None and m == best:
                s = '**{}**'.format(s)
            row.append(s)
    md.append('| ' + ' | '.join(row) + ' |')
    md.append('\n_Seeds per cell: {}._\n'.format(seeds_per_cell))
    if 'esfm_paper' in methods:
        md.append('_"ESFM (paper)" columns are the published per-scene numbers from '
                  'Moran et al., ICCV 2021 (supplementary calibrated table, "Ours"); '
                  'their post-BA values use the paper\'s own ceres BA, not this '
                  'benchmark\'s shared BA, so they are shown for reference and never '
                  'bolded._\n')
    with open(out_md, 'w') as f:
        f.write('\n'.join(md))

    # ---------- LaTeX (booktabs) ----------
    tex = []
    ncols = 1 + len(metrics) * len(methods)
    tex.append('% Auto-generated by aggregate_single_scene.py')
    tex.append('\\begin{table*}[t]')
    tex.append('\\centering')
    caption = ('Single-scene optimization on the Olsson dataset (calibrated setting). '
               'Mean over ' + str(seeds_per_cell) + ' seeds; best per scene in bold.')
    if 'esfm_paper' in methods:
        caption += (' ESFM (paper) reproduces the published per-scene numbers of '
                    '\\cite{Moran_2021_ICCV} (post-BA values use their BA) and is '
                    'excluded from the bold-best comparison.')
    tex.append('\\caption{' + caption + '}')
    tex.append('\\label{tab:single_scene_olsson}')
    tex.append('\\resizebox{\\textwidth}{!}{')
    tex.append('\\begin{tabular}{l' + 'c' * (ncols - 1) + '}')
    tex.append('\\toprule')
    line1 = [''] + ['\\multicolumn{{{}}}{{c}}{{{}}}'.format(len(methods), name)
                    for _, name, _ in metrics]
    tex.append(' & '.join(line1) + ' \\\\')
    line2 = ['Scene'] + [label[m] for _ in metrics for m in methods]
    tex.append(' & '.join(line2) + ' \\\\')
    tex.append('\\midrule')

    def texfmt(v, bold):
        s = fmt(v)
        return '\\textbf{{{}}}'.format(s) if bold else s

    for scene in scenes:
        row = [scene.replace('&', '\\&')]
        for key, _, lower_better in metrics:
            vals = {m: cell(scene, m, key) for m in methods}
            valid = {m: v for m, v in vals.items()
                     if m in bold_methods and v is not None and not np.isnan(v)}
            best = (min if lower_better else max)(valid, key=lambda m: valid[m]) if len(valid) > 1 else None
            for m in methods:
                row.append(texfmt(vals.get(m), best is not None and m == best))
        tex.append(' & '.join(row) + ' \\\\')
    tex.append('\\midrule')
    row = ['Mean']
    for key, _, lower_better in metrics:
        means = {m: agg[agg['method'] == m][key].mean() for m in methods}
        valid = {m: v for m, v in means.items()
                 if m in bold_methods and not np.isnan(v)}
        best = (min if lower_better else max)(valid, key=lambda m: valid[m]) if len(valid) > 1 else None
        for m in methods:
            row.append(texfmt(means.get(m), best is not None and m == best))
    tex.append(' & '.join(row) + ' \\\\')
    tex.append('\\bottomrule')
    tex.append('\\end{tabular}}')
    tex.append('\\end{table*}')
    with open(out_tex, 'w') as f:
        f.write('\n'.join(tex) + '\n')


def write_repro(df, results_root, out_path):
    py = os.path.join(UESFM_REPO, '.venv', 'bin', 'python')
    manifest = os.path.join(UESFM_REPO, 'datasets', 'MANIFEST_euclidean_sha256.txt')
    lines = []
    lines.append('# Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)')
    lines.append('Generated: {}'.format(datetime.now().isoformat(timespec='seconds')))
    lines.append('')
    lines.append('## Fairness policy (SPEC ground rules)')
    lines.append('- Identical inputs: both methods read the same shared '
                 '`u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).')
    lines.append('- Post-processing: BOTH methods optimized with `ba.run_ba = false`; '
                 'pre-BA numbers come from each method\'s raw cameras. Post-BA numbers '
                 '(when present) come from ONE shared pycolmap BA applied by '
                 '`evaluate_single_scene.py --ba` to both methods identically. '
                 'Neither method\'s own BA was used. Pre- and post-BA are reported separately.')
    lines.append('- Identical budgets: same epochs / eval intervals / lr / schedule per run '
                 '(recorded in each run\'s `run.conf` and below).')
    lines.append('- Same hardware, sequential: for each (scene, seed) the two methods run '
                 'back-to-back in the same LSF job on the same GPU (see per-run '
                 '`run_meta.json` host/gpu fields).')
    lines.append('- Seeds: per-run `random_seed` in the conf; raw per-seed numbers kept in '
                 '`summary.csv`.')
    lines.append('')

    if not df.empty:
        lines.append('## Sweep actually run')
        lines.append('- methods: {}'.format(sorted(df['method'].unique().tolist())))
        lines.append('- scenes ({}): {}'.format(df['scene'].nunique(),
                                                sorted(df['scene'].unique().tolist())))
        lines.append('- seeds: {}'.format(sorted(df['seed'].unique().tolist())))
        lines.append('')

    # budgets from one sample conf per method
    lines.append('## Budgets / exact commands')
    for method in sorted(df['method'].unique()) if not df.empty else []:
        sample = glob.glob(os.path.join(results_root, method, '*', 'seed*', 'run_meta.json'))
        if sample:
            with open(sample[0]) as f:
                meta = json.load(f)
            lines.append('### {}'.format(METHOD_LABELS.get(method, method)))
            lines.append('- epochs: {}  eval_intervals: {}'.format(meta.get('epochs'),
                                                                   meta.get('eval_intervals')))
            lines.append('- example command: `{}` (cwd: `{}`)'.format(
                ' '.join(meta.get('cmd', [])), meta.get('cwd')))
            lines.append('- full config: `<run_dir>/run.conf` in every run directory')
    lines.append('')

    lines.append('## Code versions')
    for name, repo in (('u-esfm (U-ESFM)', UESFM_REPO), ('esfm-baseline (official ESFM)', ESFM_REPO)):
        commit = sh('git rev-parse HEAD', cwd=repo)
        dirty = sh('git status --porcelain', cwd=repo)
        lines.append('### {}'.format(name))
        lines.append('- path: `{}`'.format(repo))
        lines.append('- commit: `{}`'.format(commit))
        if dirty:
            lines.append('- **WARNING: working tree dirty.** Diff:')
            lines.append('```diff')
            lines.append(sh('git diff', cwd=repo) or '(untracked files only)')
            lines.append('```')
            lines.append('- untracked/modified files:')
            lines.append('```')
            lines.append(dirty)
            lines.append('```')
        else:
            lines.append('- working tree clean')
    lines.append('')

    lines.append('## Environment (shared venv used for BOTH methods)')
    lines.append('- python: `{}`'.format(sh('{} --version'.format(py))))
    lines.append('```')
    lines.append(sh('{} -m pip freeze 2>/dev/null | head -200'.format(py)))
    lines.append('```')
    lines.append('')

    lines.append('## Hardware')
    hosts = set()
    for meta_path in glob.glob(os.path.join(results_root, '*', '*', 'seed*', 'run_meta.json')):
        with open(meta_path) as f:
            m = json.load(f)
        if m.get('status') == 'completed':
            hosts.add('{} | {}'.format(m.get('host'), m.get('gpu')))
    for h in sorted(hosts):
        lines.append('- {}'.format(h))
    lines.append('- CUDA (login node view): {}'.format(sh('nvcc --version 2>/dev/null | tail -1') or 'n/a'))
    lines.append('')

    lines.append('## Dataset manifest (sha256)')
    if os.path.exists(manifest):
        lines.append('```')
        with open(manifest) as f:
            lines.append(f.read().strip())
        lines.append('```')
    else:
        lines.append('- MANIFEST file not found at `{}` (checksum job may still be running)'.format(manifest))

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-root', default=os.path.join(CODE_DIR, 'results', 'single_scene'))
    ap.add_argument('--exclude-methods', default='uesfm_abl',
                    help='comma-separated methods to leave out of the tables. Default '
                         'excludes uesfm_abl: the layer-norm/residual/dropout flags only '
                         'exist in the Deep models, so at 1x3 that arm is a bit-exact '
                         'duplicate of esfm_rc (verified per-run); uesfm vs esfm_rc is '
                         'already the pure loss ablation.')
    args = ap.parse_args()

    df, failures = collect(args.results_root)
    excluded = [m.strip() for m in args.exclude_methods.split(',') if m.strip()]
    if excluded and not df.empty:
        n_before = len(df)
        df = df[~df['method'].isin(excluded)]
        if len(df) < n_before:
            print('NOTE: excluded {} runs from methods {} (see --exclude-methods help)'.format(
                n_before - len(df), excluded))
    if df.empty:
        sys.exit('No evaluated runs found under {} — run the sweep and '
                 'evaluate_single_scene.py first.'.format(args.results_root))

    out_csv = os.path.join(args.results_root, 'summary.csv')
    df.sort_values(['scene', 'method', 'seed']).to_csv(out_csv, index=False)

    methods = sorted(df['method'].unique())
    metrics = list(BASE_METRICS)
    if any(k in df.columns for k, _, _ in BA_METRICS):
        metrics += [m for m in BA_METRICS if m[0] in df.columns]

    agg = build_scene_table(df, metrics)
    seeds_per_cell = df.groupby(['scene', 'method'])['seed'].nunique()
    seeds_desc = '{}'.format(sorted(df['seed'].unique().tolist()))

    # Published ESFM numbers (reference columns, never bolded).
    bold_methods = set(methods)
    published = load_published(set(agg['scene'].unique()))
    if not published.empty:
        agg = pd.concat([agg, published], ignore_index=True)
        methods = methods + ['esfm_paper']
        # make sure the BA metric columns are in the table when the csv provides them
        for m in BA_METRICS:
            if m not in metrics and m[0] in published.columns:
                metrics = metrics + [m]

    write_tables(agg, methods, metrics,
                 os.path.join(args.results_root, 'summary_table.md'),
                 os.path.join(args.results_root, 'summary_table.tex'),
                 seeds_desc, bold_methods=bold_methods)
    write_repro(df, args.results_root, os.path.join(args.results_root, 'REPRO.md'))

    # ---------- short analysis (experimental methods only, not the paper column) ----------
    methods = sorted(df['method'].unique())
    print('Wrote: summary.csv, summary_table.md, summary_table.tex, REPRO.md '
          'under {}\n'.format(args.results_root))
    if len(methods) == 2:
        m1, m2 = methods
        wins = {m1: 0, m2: 0}
        for scene in agg['scene'].unique():
            v1 = agg[(agg.scene == scene) & (agg.method == m1)]['rot_err_deg']
            v2 = agg[(agg.scene == scene) & (agg.method == m2)]['rot_err_deg']
            if len(v1) and len(v2):
                wins[m1 if v1.iloc[0] < v2.iloc[0] else m2] += 1
        print('Rotation-error wins per scene: {}={}  {}={}'.format(
            METHOD_LABELS.get(m1, m1), wins[m1], METHOD_LABELS.get(m2, m2), wins[m2]))
    for m in methods:
        sel = df[df.method == m]
        print('{:8s} overall: rot={:.3f}deg pos={:.3f} reproj={:.3f}px time={:.0f}s '
              '({} runs)'.format(METHOD_LABELS.get(m, m), sel.rot_err_deg.mean(),
                                 sel.pos_err.mean(), sel.reproj_err_px.mean(),
                                 sel.runtime_s.mean() if sel.runtime_s.notna().any() else -1,
                                 len(sel)))
    incomplete = seeds_per_cell[seeds_per_cell < df['seed'].nunique()]
    if len(incomplete):
        print('\nCells with missing seeds:')
        print(incomplete.to_string())
    if failures:
        print('\nFailed / unevaluated runs:')
        for f in failures:
            print('  ' + f)


if __name__ == '__main__':
    main()
