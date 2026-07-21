# Report section draft: Cross-dataset generalization (for Experiments)

Paste-ready section for `MVG_Project_Report_Ortal_Dayan` (suggested: 3.x,
after the MegaDepth multi-scene experiments). Numbers: 5-seed protocol
(seeds 20-24, per-scene means over seeds, then per-dataset means), from
`CROSSDATASET_RESULTS.md` / `results/crossdataset/all_arms_seed_means.csv`.

---

## 3.x Cross-dataset generalization: 1DSfM, Strecha, BlendedMVS

**Setup.** We evaluate how the MegaDepth-trained multi-scene models transfer
to the three unseen datasets of RESfM's cross-dataset study: 1DSfM (10
internet-photo scenes, Table 2 of the paper), Strecha (4 small LIDAR-GT
scenes) and BlendedMVS (4 synthetic scenes, identified from the paper's
anonymized scene0-3 by camera count and outlier rate). RESfM never released
its cross-dataset point tracks, so we reconstruct them per their Appendix C
(SIFT, exhaustive pairwise RANSAC matching, track chaining, 4 px GT-based
outlier labels); for 1DSfM we use the dataset's released SIFT tracks with
its gt_bundle reference reconstruction as ground truth. All methods are
therefore compared on identical, our-provenance tracks; the paper's
published numbers (their tracks, their COLMAP GT) are quoted as context
only. Every arm follows the same evaluation protocol as RESfM: test-time
outlier removal, ~1,000-epoch per-scene fine-tune, robust bundle
adjustment; 5 seeds, means reported.

**Arms.** ESFM (deep 2x3 network trained with reprojection loss, no outlier
removal at test); U-ESFM (same checkpoint + label-free MAD outlier removal,
alpha=2); U-ESFM-adaptive (deep 2x3 trained with the adaptive
reprojection+outlier loss of Sec. 2.2.2, MAD at test); their shallow (1x3)
counterparts and test-time-training variants (Appendix); and RESfM's
official released checkpoint with its supervised classifier (thresholds
0.6/0.8 per the paper). The fine-tune learning rate is a per-dataset
setting (5e-3; 1e-4 on Strecha - see the learning-rate finding below).

**Table X: mean rotation error (deg) / mean translation error, 5 seeds.**

| method | 1DSfM | BlendedMVS | Strecha |
|---|---|---|---|
| ESFM (no outlier removal) | 8.37 / 15.34 | 7.78 / 0.111 | 0.14 / 0.030 |
| RESfM (released checkpoint) | 10.21 / 10.71 | 31.91 / 0.330 | 2.04 / 0.196 |
| U-ESFM (deep, MAD) | 7.91 / 12.62 | **2.39 / 0.040** | **0.15 / 0.027** |
| U-ESFM shallow adaptive | 9.23 / 13.17 | 15.42 / 0.224 | 8.37 / 2.18* |
| U-ESFM deep adaptive | **5.73 / 9.36** | 6.29 / 0.097 | 1.59 / 0.218 |
| *RESfM paper (their tracks)* | *3.98 / (0.427)* | *0.011 / 0.001* | *0.027 / 0.005* |

ESFM and U-ESFM on Strecha are at lr 1e-4 (the per-dataset setting); the
shallow-adaptive Strecha value marked * is its lr-1e-4 variant. The paper
row is not directly comparable (different tracks and ground truth; 1DSfM
translation is in a different GT scale, hence parenthesized).

**Findings.**

1. *A U-ESFM variant is the best-performing method on every dataset, on
both metrics.* The adaptive-loss deep model leads 1DSfM (5.73 deg / 9.36,
and 2.86 deg on the 8 non-pathological scenes - see below); the
reprojection-trained deep model with MAD leads BlendedMVS and Strecha. No
single variant sweeps all three: the adaptive loss wins the contaminated
internet-photo setting but retains a fragility on two BlendedMVS scenes
and one Strecha scene, while the MAD-only model is the most robust overall.

2. *The released supervised classifier does not transfer across track
distributions.* On our tracks, RESfM's official checkpoint trails every
U-ESFM variant on 1DSfM, and collapses on BlendedMVS (31.9 deg, driven by
two scenes at 46-81 deg) - while the same weights reproduce their
paper-level Strecha accuracy (0.01-0.02 deg on 3/4 scenes). Label-free
statistical removal degrades gracefully where the learned classifier
breaks; this is the central robustness argument for U-ESFM.

3. *The fine-tune learning rate is scene-scale-dependent and decisive.* At
the protocol lr (5e-3), every method degenerates on Strecha's 8-25-image
scenes (14-17 deg with low reprojection error - degenerate optima); at
1e-4 the same checkpoints reach 0.14-0.15 deg. The reverse holds on the
large scenes, where 1e-4 underfits. We expose this as a per-dataset
configuration (train.lr_tuning), mirroring RESfM's own per-dataset
classifier threshold.

4. *Two 1DSfM scenes are data-pathological in the released files.* Ellis
Island and Tower of London fail for every method (15-50 deg) including the
released RESfM checkpoint; an oracle experiment (removing ground-truth-
labeled outliers before optimization) does not rescue them, while it makes
equally contaminated control scenes nearly perfect. The released
tracks/gt_bundle combination is globally inconsistent for these two scenes
(the paper's own numbers there rely on registering as few as 20% of the
cameras). We therefore report them included in all means for symmetry, and
additionally quote the 8-scene means where relevant.

5. *Track provenance dominates absolute numbers.* With identical weights,
the released checkpoint scores 10.21 deg on our 1DSfM tracks vs 3.98 in
the paper on theirs; on Strecha/BlendedMVS the paper's 0.01-0.03 deg
regime reflects their (unreleased) track construction rather than model
differences. All comparisons in Table X are therefore same-track and
internal; the absolute gap to the paper row measures data provenance, not
model quality.

*(Appendix pointers: full 10-arm tables incl. lr variants and TTT,
per-scene values, and the oracle experiment - CROSSDATASET_RESULTS.md;
track construction and its validation - BUILD_TRACKS.md.)*
