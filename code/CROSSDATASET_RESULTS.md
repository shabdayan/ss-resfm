# Cross-dataset evaluation results (R10) — our tracks

Two protocols below: the seed-20 single-run tables (paper protocol,
2026-07-18) and the **5-seed protocol** (seeds 20-24, 2026-07-20 — the
headline numbers; per-scene MEANS over seeds per the project's reporting
convention). Per-scene values:
`results/crossdataset/all_arms_seed_means.csv`
(built by `aggregate_crossdataset_seeds.py --stat mean`; full 5/5 seed
coverage for every arm x dataset x scene; `--stat median` variant also
available).

## 5-SEED PROTOCOL (per-dataset mean of per-scene seed-means)

Rotation (deg, post-BA). The `paper` column is RESfM's published Tables 2-4
"Ours" (THEIR tracks + COLMAP GT, seed 20 — reference only, see the paper
section below; bold marks the best of OUR arms):

| dataset | ESFM | ESFM@1e-4 | RESfM-off | stage1 | stage1@1e-4 | shal-Ep17k | shal-Ep17k@1e-4 | TTT-comb | TTT-reproj | deep-adpt | paper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1DSfM | 8.37 | 14.38 | 10.21 | 7.91 | 12.08 | 9.23 | 9.85 | 8.93 | 8.83 | **5.73** | 3.98 |
| BlendedMVS | 7.78 | 7.15 | 31.91 | **2.39** | 12.31 | 15.42 | 23.24 | 9.94 | 15.45 | 6.29 | 0.011 |
| Strecha | 16.69 | **0.14** | 2.04 | 15.63 | 0.15 | 14.44 | 8.37 | 12.48 | 14.40 | 1.59 | 0.027 |

Translation (paper 1DSfM value is in THEIR GT's scale — not comparable to
our gt_bundle-scaled column):

| dataset | ESFM | ESFM@1e-4 | RESfM-off | stage1 | stage1@1e-4 | shal-Ep17k | shal-Ep17k@1e-4 | TTT-comb | TTT-reproj | deep-adpt | paper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1DSfM | 15.34 | 16.99 | 10.71 | 12.62 | 14.44 | 13.17 | 12.07 | 13.51 | 14.09 | **9.36** | (0.427) |
| BlendedMVS | 0.111 | 0.113 | 0.330 | **0.040** | 0.176 | 0.224 | 0.299 | 0.152 | 0.224 | 0.097 | 0.001 |
| Strecha | 2.97 | 0.030 | 0.196 | 2.93 | **0.027** | 2.90 | 2.18 | 2.73 | 2.88 | 0.218 | 0.005 |

Seed-robust conclusions (final, incl. the deep-adaptive arm added
2026-07-21 — deep 2x3 + reproj+outlier CombinedLoss, Ep16500, per-dataset
lr_tuning policy): (1) **on every dataset, a U-ESFM arm is the best method
on both metrics** — 1DSfM: deep-adpt 5.73/9.36 (also the 1DSfM-8 rotation
leader at 2.86, and it fixes stage1's Yorkminster failure, 9.7 vs 23.0);
BlendedMVS: stage1 2.39/0.040; Strecha: stage1@1e-4 0.15/0.027 (ESFM@1e-4
rotation 0.14 = same checkpoint, no MAD). RESfM-off's only remaining edge
is 1DSfM-8 translation (3.36 vs deep-adpt 4.60). (2) No single arm sweeps:
deep-adpt is 1st/2nd/2nd across datasets with the best worst-case (6.29)
— the adaptive loss wins internet-photo scenes but keeps a BlendedMVS
scene2/3 fragility and Strecha's entry-P10 (6.14, all seeds); stage1+lr
policy is 1st/1st/4th. The robustness claim ("U-ESFM never collapses;
the released classifier does: 31.9 on BlendedMVS") holds for both.
(3) The Strecha lr finding is unchanged (0.14-0.15 @1e-4 vs 15-17 @5e-3).

## RESfM paper reference (Tables 2-4 "Ours" — THEIR tracks, not comparable head-to-head)

Source: resfm.pdf; RESfM's own tracks (SIFT rebuilt from images) and their
GT (COLMAP for 1DSfM; seed 20, no multi-seed protocol). Rotation (deg) is
unit-free and indicative; translation units follow each GT's scale, so
1DSfM translation is NOT comparable across track provenances (their COLMAP
scale vs our gt_bundle scale).

| scene | Nc | Out% | Nr | Rot | Trans |
|---|---|---|---|---|---|
| Alamo | 573 | 32.6 | 484 | 3.66 | 0.515 |
| Ellis Island | 227 | 25.1 | 214 | 0.82 | 0.122 |
| Madrid Metropolis | 333 | 39.4 | 244 | 8.42 | 0.827 |
| Montreal Notre Dame | 448 | 31.7 | 346 | 2.82 | 0.352 |
| Notre Dame | 549 | 35.6 | 517 | 1.20 | 0.231 |
| NYC Library | 330 | 33.6 | 224 | 3.96 | 0.429 |
| Piazza del Popolo | 336 | 33.1 | 249 | 2.20 | 0.186 |
| Tower of London | 467 | 27.0 | 94 | 0.67 | 0.026 |
| Vienna Cathedral | 824 | 31.4 | 479 | 1.52 | 0.112 |
| Yorkminster | 432 | 29.0 | 331 | 14.54 | 1.468 |
| **1DSfM mean** | | | | **3.98** | 0.427 |
| entry-P10 | 10 | 4.8 | 10 | 0.024 | 0.008 |
| fountain-P11 | 11 | 1.4 | 11 | 0.028 | 0.003 |
| Herz-Jesu-P8 | 8 | 1.8 | 8 | 0.026 | 0.004 |
| Herz-Jesu-P25 | 25 | 2.8 | 24 | 0.030 | 0.006 |
| **Strecha mean** | | | | **0.027** | 0.005 |
| scene0 (75) | 75 | 2.0 | 75 | 0.016 | 0.0007 |
| scene1 (51) | 51 | 1.4 | 51 | 0.011 | 0.0021 |
| scene2 (33) | 33 | 2.2 | 33 | 0.009 | 0.0006 |
| scene3 (66) | 66 | 8.8 | 66 | 0.007 | 0.0007 |
| **BlendedMVS mean** | | | | **0.011** | 0.001 |

Readings against our tables (all-ours, our tracks):
- **Track provenance dominates the gap.** Same released weights: RESfM-off
  on our 1DSfM tracks 10.21 deg vs 3.98 in the paper on theirs. Our best
  arm (stage1 7.91) also sits well above their numbers everywhere —
  consistent with our tracks carrying more contamination (e.g. Ellis 56.3%
  vs their 25.1%) and different GT (gt_bundle vs their COLMAP).
- **Tower_of_London is scene-pathology only on OUR data**: paper 0.67 deg
  (their tracks, though with Nr 94/467 — they register only 20% of the
  cameras) vs 14-50 deg for every arm on ours — supports the oracle verdict
  that the released tracks/gt_bundle are the problem, not the scene.
  Ellis Island likewise: paper 0.82, ours >= 15 for all arms.
- **Their hardest scene is Yorkminster (14.54)** — where several of our
  arms do BETTER on our tracks (shal@1e-4 9.13, stage1@1e-4 11.38).
- On Strecha/BlendedMVS their numbers (0.011-0.03 deg) are 1-2 orders
  below anything on our tracks (best: 0.14 / 2.39) — our SIFT tracks are
  the visible ceiling there, not the models.

---

# Seed-20 tables (paper protocol, 2026-07-18)

All arms evaluated per-scene (RESfM protocol: TEST -> outlier prune -> ~1K-epoch
fine-tune -> robust BA) on the tracks built per `build_tracks/BUILD_TRACKS.md`
(provenance: OUR tracks + for 1DSfM gt_bundle GT — R13 switch on). Per-scene
values: `results/crossdataset/all_arms_per_scene.csv`. Fine-tune lr is the
protocol 5e-3 unless marked `@1e-4` (the `*_ftlr1e4` templates).

Arms: ESFM = stage-1 checkpoint evaluated with NO outlier removal (output_mode=1);
RESfM-off = official released checkpoint, classifier outliers (0.6/0.8);
stage1 = deep 2x3 U-ESFM, MAD(alpha=2); shal-Ep17k = shallow 1x3 adaptive
(reproj+outlier CombinedLoss) best checkpoint Ep17000, MAD; TTT-* = test-time
fine-tune variants from the Ep17k checkpoint (comb = full unsupervised loss,
reproj_only = control; frozen row = step-0 in the ttt result dirs).

## Mean rotation error (deg, post-BA)

| dataset | ESFM | ESFM@1e-4 | RESfM-off | stage1 | stage1@1e-4 | shal-Ep17k | shal-Ep17k@1e-4 | TTT-reproj | TTT-comb |
|---|---|---|---|---|---|---|---|---|---|
| 1DSfM | 10.12 | 14.95 | 10.59 | 8.77 | 11.59 | 10.22 | 9.71 | 8.56 | **7.74** |
| BlendedMVS | 10.63 | 6.96 | 31.91 | **0.79** | 9.54 | 19.27 | 27.85 | 19.28 | 8.46 |
| Strecha | 14.44 | **0.10** | 2.04 | 16.74 | **0.10** | 14.30 | 8.29 | 14.42 | 14.43 |

## Mean translation error (post-BA)

| dataset | ESFM | ESFM@1e-4 | RESfM-off | stage1 | stage1@1e-4 | shal-Ep17k | shal-Ep17k@1e-4 | TTT-reproj | TTT-comb |
|---|---|---|---|---|---|---|---|---|---|
| 1DSfM | 15.87 | 18.12 | 11.12 | 11.96 | 13.63 | 13.01 | 12.33 | 13.17 | **12.04** |
| BlendedMVS | 0.162 | 0.107 | 0.330 | **0.013** | 0.149 | 0.271 | 0.342 | 0.271 | 0.135 |
| Strecha | 2.95 | **0.018** | 0.196 | 3.21 | **0.018** | 2.86 | 2.16 | 2.83 | 3.26 |

## Findings

1. **Fine-tune lr is dataset-dependent, and decisive on Strecha.** The protocol
   lr 5e-3 flatlines every arm on the small (8-25 image) Strecha scenes
   (14-17 deg with LOW BA reprojection = degenerate solutions); at ft-lr 1e-4
   the same checkpoints reach 0.10 deg / 0.018 — better than the official RESfM
   checkpoint on our tracks (2.04, dragged by Herz-Jesu-P8) and in the regime of
   the paper's Table 3. Conversely, 1e-4 HURTS on 1DSfM and BlendedMVS
   (large scenes need the faster 5e-3). Extends the MegaDepth flatline
   diagnosis (multiscene-experiment-state) to scene scale, not just model depth.
2. **Track validation via RESfM-off:** the official checkpoint reproduces its
   paper-level Strecha accuracy on our tracks (0.01-0.02 deg on 3/4 scenes at
   the protocol lr) — the Strecha tracks/GT are sound; failures elsewhere are
   model/protocol-side. Its BlendedMVS failure (31.9 mean, driven by scene2/
   scene3) did not reproduce for our arms.
3. **Best per dataset (mean Rot/Trans):** 1DSfM: TTT-comb 7.74/12.04 (best of
   all arms incl. RESfM-off 10.59/11.12); BlendedMVS: stage1@5e-3 0.79/0.013;
   Strecha: ESFM or stage1 @1e-4 0.10/0.018. No single arm wins everywhere.
4. **TTT:** comb-TTT from the Ep17k checkpoint helps exactly where the frozen
   model is weak (BlendedMVS 19.28 -> 8.46; 1DSfM 8.56 -> 7.74) and is neutral
   on Strecha (the lr, not the loss, is the binding constraint there) —
   consistent with the MegaDepth verdict that TTT rescues mis-tuned configs.
5. **Ellis_Island and Tower_of_London fail for every arm** (17-50 deg incl.
   RESfM-off) — RESOLVED as a data pathology (see "Failure investigation").
6. Shallow-adaptive caveats: best checkpoint Ep17000 (validation plateaued;
   training ran to Ep19999); the earlier Ep3000 rows in the CSV are superseded
   but kept for the training-progress comparison.

## Failure investigation: Ellis_Island + Tower_of_London (2026-07-18)

Ruled out, in order: (a) camera-graph connectivity — Ellis is the
best-connected scene of the set (Fiedler 0.41 vs healthy NYC 0.0099);
(b) GT calibration — gt_bundle focals match coords.txt EXIF (median ratio
1.00) on all scenes; (c) coherent symmetry fold — the misplaced cameras are
internally incoherent (34-60 deg relative errors); (d) contamination
structure — NYC_Library has the same profile (55.6% outlier-dominated
"deceptive" camera pairs, 30% fully-outlier tracks) and evaluates fine.

Decisive **oracle experiment** (GT-labeled outliers removed at load,
remove_outliers_gt=True, plain ESFM protocol, stage-1 ckpt;
confs/crossdataset_oracle_1dsfm.conf.template,
results/crossdataset/oracle_1dsfm_eval/):

| scene | contaminated Rot/Trans | oracle Rot/Trans |
|---|---|---|
| Ellis_Island | 16.96 / 18.8 | 15.65 / 16.1 |
| Tower_of_London | 21.58 / 51.7 | 15.85 / 71.9 |
| NYC_Library (control) | 1.74 / 3.4 | 0.53 / 0.70 |
| Alamo (control) | 1.61 / 1.3 | 2.53 / 1.5 |

Even perfect outlier removal does not rescue Ellis/Tower, while the equally
contaminated NYC control becomes near-perfect. The labeled-INLIER tracks of
these two scenes reconstruct ~15 deg away from gt_bundle's cameras despite
low per-point reprojection (1.4-1.6 px) — the released tracks.txt +
gt_bundle combination is globally inconsistent there (landmark-confusion
scenes; RESfM's paper sidestepped this by rebuilding tracks from images with
COLMAP GT). **Recommendation:** footnote Ellis_Island and Tower_of_London as
excluded-for-data-pathology in the 1DSfM comparison; the remaining 8 scenes
carry the row. The 1DSfM per-dataset means above INCLUDE the two pathological
scenes for all arms symmetrically.
