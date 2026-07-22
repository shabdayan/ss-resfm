"""Render the cross-dataset results (CROSSDATASET_RESULTS.md numbers) to PDF.

Page 1: headline 5-seed tables (rotation + translation, all arms + paper).
Page 2: findings summary + scene-matched validation table.
Page 3: per-scene seed-mean rotation table (appendix).

Usage: python make_crossdataset_report.py
       [--csv results/crossdataset/all_arms_seed_means.csv]
       [--out results/crossdataset/crossdataset_report.pdf]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

INK = "#1a1f24"
INK2 = "#5b6570"

ORDER = ["ESFM", "ESFM@1e-4", "RESfM-off", "stage1", "stage1@1e-4",
         "shal-Ep17k", "shal-Ep17k@1e-4", "TTT-comb", "TTT-reproj", "deep-adpt"]
PAPER = {"Rot": {"1dsfm": 3.98, "blendedmvs": 0.011, "strecha": 0.027},
         "Trans": {"1dsfm": 0.427, "blendedmvs": 0.001, "strecha": 0.005}}

FINDINGS = """\
1. A U-ESFM arm is best on every dataset, on both metrics: deep-adpt on 1DSfM
   (5.73 deg / 9.36; 2.86 deg on the healthy-8), stage1 (deep+MAD) on BlendedMVS
   (2.39 / 0.040), stage1@1e-4 on Strecha (0.15 / 0.027). No single arm sweeps;
   deep-adpt has the best worst-case (6.29 deg) of any arm.
2. The released supervised classifier does not transfer across track
   distributions: it trails every U-ESFM variant on 1DSfM and collapses on
   BlendedMVS (31.9 deg) while our label-free MAD arms degrade gracefully.
3. The fine-tune lr is scene-scale-dependent and decisive: protocol 5e-3
   degenerates all methods on Strecha's 8-25-image scenes (14-17 deg); the same
   checkpoints reach 0.14-0.15 deg at 1e-4 (conf-exposed as train.lr_tuning).
4. Ellis_Island and Tower_of_London are pathological in the RELEASED 1DSfM
   data: every arm fails (15-50 deg) and oracle outlier removal does not rescue
   them, while equally contaminated controls become near-perfect.
5. Scene-matched validation: on comparable data our evaluation reproduces the
   paper (table below); the full-dataset gap to the paper is manufactured by
   the three pathological/collapse cases above, i.e. provenance, not bias."""

VALIDATION = [
    ("RESfM, 1DSfM healthy-8 (excl. Ellis/Tower)", "4.79", "4.16"),
    ("ESFM, 1DSfM healthy-8", "8.56", "6.57"),
    ("RESfM, Strecha excl. Herz-Jesu-P8", "0.02-0.03", "0.01-0.02"),
]


def dataset_table(t, metric):
    piv = t.pivot_table(index="ds", columns="arm", values=f"{metric}_agg",
                        aggfunc="mean").reindex(columns=ORDER)
    piv["paper"] = pd.Series(PAPER[metric])
    return piv.round(3 if metric == "Trans" else 2)


def text_page(pdf, title, blocks, fontsize=8):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.05, 0.95, title, fontsize=14, color=INK, weight="bold")
    y = 0.88
    for kind, content in blocks:
        if kind == "sub":
            fig.text(0.05, y, content, fontsize=10, color=INK, weight="bold")
            y -= 0.035
        else:
            fig.text(0.05, y, content, fontsize=fontsize, color=INK,
                     family="monospace", va="top", linespacing=1.5)
            y -= 0.028 * (content.count("\n") + 2)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/crossdataset/all_arms_seed_means.csv")
    ap.add_argument("--out", default="results/crossdataset/crossdataset_report.pdf")
    args = ap.parse_args()
    t = pd.read_csv(args.csv)

    with PdfPages(args.out) as pdf:
        text_page(pdf, "U-ESFM cross-dataset generalization - 5-seed protocol "
                       "(seeds 20-24, per-scene means; our tracks)",
                  [("sub", "Mean rotation error (deg, post-BA); 'paper' = RESfM "
                           "Tables 2-4 on THEIR tracks (reference only)"),
                   ("txt", dataset_table(t, "Rot").to_string()),
                   ("sub", "Mean translation error (1DSfM paper value in their "
                           "GT scale - not comparable)"),
                   ("txt", dataset_table(t, "Trans").to_string()),
                   ("sub", "Arms"),
                   ("txt", "ESFM = deep 2x3, reprojection loss, no outlier removal | "
                           "stage1 = same ckpt + MAD(alpha=2) = U-ESFM\n"
                           "shal-Ep17k / deep-adpt = 1x3 / 2x3 trained with the "
                           "adaptive reproj+outlier CombinedLoss | @1e-4 = fine-tune lr\n"
                           "TTT-* = test-time training from the shallow adaptive ckpt | "
                           "RESfM-off = official released checkpoint + classifier")])
        text_page(pdf, "Findings",
                  [("txt", FINDINGS),
                   ("sub", "Scene-matched validation (mean rotation, deg)"),
                   ("txt", "\n".join(f"{c:48s} paper {p:>10s}   ours {o:>10s}"
                                     for c, p, o in VALIDATION))])
        per_scene = t.pivot_table(index=["ds", "scene"], columns="arm",
                                  values="Rot_agg").reindex(columns=ORDER).round(2)
        text_page(pdf, "Appendix: per-scene seed-mean rotation error (deg)",
                  [("txt", per_scene.to_string())], fontsize=6.5)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
