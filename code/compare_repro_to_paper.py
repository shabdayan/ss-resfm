"""
Compare RESfM reproduction results (run_resfm_repro.sh) to the paper's Table 1.

Paper numbers ("Ours" columns: Nr, mean Rot [deg], mean Trans) were extracted
programmatically from resfm.pdf page 8 (Table 1). Reproduction numbers are read
from each scene's Results_FINE_TUNE_stage_1_esfm_outliers.xlsx
(#registered_cams_final / Rs_ba_final_mean / ts_ba_final_mean).

Usage: python compare_repro_to_paper.py [--eval_root results/multiscene/resfm_repro]
"""
import argparse
import os
import pandas as pd

# resfm.pdf Table 1, "Ours" (extracted from the PDF; keys = scene ids)
PAPER = {
    "0238": {"Nr": 283, "Rot": 2.61, "Trans": 0.325}, "0060": {"Nr": 503, "Rot": 0.29, "Trans": 0.029},
    "0197": {"Nr": 667, "Rot": 4.22, "Trans": 0.333}, "0094": {"Nr": 537, "Rot": 3.77, "Trans": 0.750},
    "0265": {"Nr": 346, "Rot": 1.25, "Trans": 0.389}, "0083": {"Nr": 596, "Rot": 0.64, "Trans": 0.058},
    "0076": {"Nr": 524, "Rot": 0.37, "Trans": 0.094}, "0185": {"Nr": 350, "Rot": 0.06, "Trans": 0.010},
    "0048": {"Nr": 474, "Rot": 4.69, "Trans": 0.178}, "0024": {"Nr": 309, "Rot": 2.03, "Trans": 0.398},
    "0223": {"Nr": 204, "Rot": 3.76, "Trans": 0.510}, "5016": {"Nr": 28,  "Rot": 0.12, "Trans": 0.016},
    "0046": {"Nr": 399, "Rot": 0.95, "Trans": 0.043}, "0099": {"Nr": 190, "Rot": 3.53, "Trans": 0.709},
    "1001": {"Nr": 251, "Rot": 1.70, "Trans": 0.661}, "0231": {"Nr": 246, "Rot": 0.84, "Trans": 0.065},
    "0411": {"Nr": 273, "Rot": 0.13, "Trans": 0.020}, "0377": {"Nr": 210, "Rot": 0.29, "Trans": 0.018},
    "0102": {"Nr": 284, "Rot": 0.28, "Trans": 0.059}, "0147": {"Nr": 207, "Rot": 4.62, "Trans": 0.325},
    "0148": {"Nr": 197, "Rot": 0.60, "Trans": 0.035}, "0446": {"Nr": 288, "Rot": 0.72, "Trans": 0.046},
    "0022": {"Nr": 274, "Rot": 0.29, "Trans": 0.039}, "0327": {"Nr": 271, "Rot": 0.26, "Trans": 0.090},
    "0015": {"Nr": 215, "Rot": 1.04, "Trans": 0.167}, "0455": {"Nr": 293, "Rot": 0.68, "Trans": 0.105},
    "0496": {"Nr": 281, "Rot": 0.35, "Trans": 0.055}, "1589": {"Nr": 290, "Rot": 0.14, "Trans": 0.019},
    "0012": {"Nr": 287, "Rot": 0.40, "Trans": 0.027}, "0104": {"Nr": 193, "Rot": 0.29, "Trans": 0.029},
    "0019": {"Nr": 250, "Rot": 0.06, "Trans": 0.008}, "0063": {"Nr": 262, "Rot": 0.46, "Trans": 0.048},
    "0130": {"Nr": 192, "Rot": 0.20, "Trans": 0.023}, "0080": {"Nr": 139, "Rot": 0.59, "Trans": 0.096},
    "0240": {"Nr": 275, "Rot": 3.13, "Trans": 0.265}, "0007": {"Nr": 172, "Rot": 0.91, "Trans": 0.041},
}

RESULTS_FILE = "Results_FINE_TUNE_stage_1_esfm_outliers.xlsx"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_root', default='results/multiscene/resfm_repro')
    args = parser.parse_args()

    rows, missing = [], []
    for scan, paper in PAPER.items():
        path = os.path.join(args.eval_root, f'{scan}_ba', RESULTS_FILE)
        if not os.path.exists(path):
            missing.append(scan)
            continue
        df = pd.read_excel(path).set_index('Scene')
        r = df.iloc[0]
        rows.append({
            'Scene': scan,
            'Nr_paper': paper['Nr'], 'Nr_ours': int(r['#registered_cams_final']),
            'Rot_paper': paper['Rot'], 'Rot_ours': round(float(r['Rs_ba_final_mean']), 2),
            'Trans_paper': paper['Trans'], 'Trans_ours': round(float(r['ts_ba_final_mean']), 3),
        })

    if not rows:
        print(f'No results found under {args.eval_root} — run run_resfm_repro.sh first.')
        return

    out = pd.DataFrame(rows).set_index('Scene')
    out['dNr'] = out['Nr_ours'] - out['Nr_paper']
    out['dRot'] = (out['Rot_ours'] - out['Rot_paper']).round(2)
    out['dTrans'] = (out['Trans_ours'] - out['Trans_paper']).round(3)
    print(out.to_string())
    print('\nMeans over available scenes:')
    print(out[['Nr_paper', 'Nr_ours', 'Rot_paper', 'Rot_ours', 'Trans_paper', 'Trans_ours']].mean().round(3).to_string())
    if missing:
        print(f'\nStill missing ({len(missing)}): {" ".join(missing)}')
    out.to_excel(os.path.join(args.eval_root, 'Comparison_to_paper_Table1.xlsx'))
    print(f"\nSaved: {os.path.join(args.eval_root, 'Comparison_to_paper_Table1.xlsx')}")


if __name__ == '__main__':
    main()
