# SS-RESfM

> **Self-Supervised Robustness in Deep Structure-from-Motion** — what happens when you
> remove the outlier supervision from RESfM, and why that question matters more than it
> sounds.

## What this is

RESfM rejects outlier correspondences with a classifier supervised by COLMAP-derived
labels — supervision that presupposes a trusted reconstruction of the very scenes whose
contamination makes reconstruction hard. **SS-RESfM** replaces that supervision with a
self-supervised loss (confident pseudo-labels from percentiles of the model's own
reprojection error) inside an otherwise byte-identical pipeline, and uses the pair as an
instrument to study *when outlier supervision fails*.

Headline findings (details and exact numbers in the paper drafts under `claude specs/`):

1. **The label ceiling is causal** — COLMAP-regenerated labels collapse (recall
   0.82 → ~0.3) above ~15% contamination, and retraining supervised RESfM *on* a
   60%-contaminated domain with best-case GT-derived labels still loses ~2× to
   label-free removal. At 60%, GT-supervised classification performs identically to no
   outlier mechanism at all.
2. **The optimal mechanism flips with contamination** — hard removal wins the
   high-contamination regime (beating the supervised baseline on every seed pair at
   ~60%), soft reweighting wins clean OOD; supervision keeps only the curated
   25–43% middle band.
3. **Reliability is the qualitative differentiator** — label-free arms land in a narrow
   band on every training seed; supervised recipes are seed-bimodal on clean OOD data
   (and even in-domain), a failure mode invisible to the field's single-seed reporting.
   Our seed-additivity analysis shows single-seed comparisons can reverse the verdict
   on four of six datasets.
4. **The oracle-cleaning ceiling** — at 60% contamination, label-free removal on
   contaminated tracks beats the same architecture on *oracle-cleaned* tracks
   (12.4±2.1 vs. ESFM\* 18.12±4.35; 23 of 25 seed pairs): deleting 60% of
   observations — even correctly — starves the constraint structure. How labels are
   used matters as much as having them.
5. **A reusable robustness benchmark** — rebuilt track suites for five OOD datasets
   (0.5–61% measured contamination) with GT-derived labels and per-domain splits, plus
   5-seed bands for every baseline in RESfM's comparison set (ESFM, ESFM*, GASFM,
   GLOMAP, COLMAP).

## Headline numbers

Post-BA mean translation error, five-training-seed bands (lower is better; best per
column **bold**). Full tables, medians, rotation, and per-scene results in the paper
drafts.

| Method | MegaDepth 25% | 1DSfM 43% | 1DSfM-hard 60% | Strecha 1.7% | BMVS 3.1% | Olsson 0.5% |
|---|---|---|---|---|---|---|
| **SS-RESfM (best arm)**¹ | 0.40±0.11 (wttt) | **8.9±1.2** (rm+TTT) | **12.4±2.1** (remove) | 1.97±0.09 (weight) | **0.14±0.02** (soft) | **2.9±0.3** (weight) |
| — SS madweight | 0.51±0.08 | 11.0±1.7 | 18.9±4.1 | 2.65±0.25 | 0.14±0.02 | 3.0±0.3 |
| — SS weight | 0.51±0.09 | 14.4±3.0 | 22.9±2.8 | 1.97±0.09 | 0.14±0.03 | 2.9±0.3 |
| — SS weight+TTT | 0.40±0.11 | 15.7±2.1 | 22.2±3.2 | 2.00±0.05 | 0.14±0.02 | 8.0±0.3 |
| — SS remove | 0.51±0.08 | 9.1±2.2 | 12.4±2.1 | 3.20±0.11 | 0.24±0.06 | 3.4±0.6 |
| — SS remove+TTT | 0.56±0.07 | 8.9±1.2 | 15.2±1.0 | 3.01±0.07 | 0.22±0.10 | 3.6±0.4 |
| RESfM (scratch, supervised) | **0.37±0.12** | 10.5±1.5 | 19.2±3.4 | **0.39±0.72** | 0.35±0.04 | 7.4±2.6 |
| RESfM (released ckpt) | 0.203 | 10.71 | 15.29 | 0.20 | 0.33 | 9.10 |
| ESFM (same tracks, no mech.) | 0.71±0.11 | 18.42±0.75 | 21.76±0.80 | 2.03±0.14 | 6.45±3.75 | 5.91±3.32 |
| ESFM\* (oracle-clean tracks) | 0.60±0.18 | 13.24±0.52 | 18.12±4.35 | 1.14±0.45 | 3.79±3.51 | — |
| GASFM (released, our BA) | 2.25 | 25.84 | 36.49 | 1.14 | 11.13 | 4.45 |
| GLOMAP (classical) | 3.33 | 27.21 | 35.42 | **0.047** | 0.339 | 3.17 |
| COLMAP (incremental) | — | — | 20.4±1.1 | — | — | — |

¹ *"Best arm" is a per-dataset selection over the five SS-RESfM mechanism variants
listed below it — not a single deployable configuration. The optimal mechanism flips
with contamination (removal high, reweighting low), which is itself the paper's
complementarity finding; no cheap unsupervised selector reliably picks the winner
per scene (see the selector study in the paper).*

In-distribution causal check (train on the domain itself, best-case GT labels):
at 60%, label-free MAD 12.1±6.1 vs. supervised 23.4±7.8 vs. no-mechanism 23.1±5.9 —
supervision with perfect labels adds nothing over no mechanism at either extreme of
the contamination axis (60% and 0.5%), earning its keep only in the curated 25–43%
middle band.

## Repository map

| Path | Contents |
|---|---|
| `code/` | Training/eval pipeline (fork of the official RESfM codebase): `multiple_scenes_learning.py`, `single_scene_optimization.py`, `run_multiscene_eval.sh` |
| `code/confs/` | Every training and evaluation configuration (per arm × dataset × seed) |
| `code/build_tracks/` | Appendix-C track builder (SIFT+RANSAC), GT labeling, frame mapping |
| `code/verify_iclr_tables.py` | Recomputes every printed table cell from saved eval outputs |
| `claude specs/` | Paper drafts (`ICLR_ssresfm.tex` = full-length master; `CVPR_ssresfm.tex` + supp = submission-shaped), figures, the Short Summary, and the resume handbook |
| `papers and code/` | Upstream reference code (RESfM, ESFM, GASFM) with minimal patches |

Results and datasets live outside git (cluster storage); `RUN_MULTISCENE.md` in `code/`
documents the training/eval workflow.

## Lineage & acknowledgements

This project builds directly on the official implementations of
[RESfM](https://github.com/FadiKhatib/resfm) (Khatib, Kasten, Moran, Galun, Basri) and
[ESFM](https://github.com/drormoran/Equivariant-SFM) (Moran et al.) — see their repositories for licenses.
Benchmarks: MegaDepth, 1DSfM, Strecha, BlendedMVS, and Olsson's dataset. Developed at
the Weizmann Institute of Science.
