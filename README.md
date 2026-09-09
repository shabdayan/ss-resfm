# SS-RESfM

> **Self-Supervised Robustness in Deep Structure-from-Motion** — what happens when you
> remove the outlier supervision from RESfM, and why that question matters more than it
> sounds.

**Status**: research code under active development. The accompanying papers are in
submission (3DV 2027 under review; CVPR 2027 version in preparation) — **please keep
this repository private until decisions are out** (double-blind).

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
   contaminated tracks beats the same architecture on *oracle-cleaned* tracks:
   how labels are used matters as much as having them.
5. **A reusable robustness benchmark** — rebuilt track suites for five OOD datasets
   (0.5–61% measured contamination) with GT-derived labels and per-domain splits, plus
   5-seed bands for every baseline in RESfM's comparison set (ESFM, ESFM*, GASFM,
   GLOMAP, COLMAP).

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
[RESfM](https://github.com/) (Khatib, Kasten, Moran, Galun, Basri) and
[ESFM](https://github.com/) (Moran et al.) — see their repositories for licenses.
Benchmarks: MegaDepth, 1DSfM, Strecha, BlendedMVS, and Olsson's dataset. Developed at
the Weizmann Institute of Science.
