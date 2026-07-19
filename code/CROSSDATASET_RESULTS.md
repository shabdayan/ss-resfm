# Cross-dataset evaluation results (R10) — our tracks

Two protocols below: the seed-20 single-run tables (paper protocol,
2026-07-18) and the **5-seed median protocol** (seeds 20-24, 2026-07-20 —
the headline numbers). Per-scene seed-medians:
`results/crossdataset/all_arms_seed_medians.csv`
(built by `aggregate_crossdataset_seeds.py`; full 5/5 seed coverage for
every arm x dataset x scene).

## 5-SEED MEDIAN PROTOCOL (per-dataset mean of per-scene seed-medians)

Rotation (deg, post-BA):

| dataset | ESFM | ESFM@1e-4 | RESfM-off | stage1 | stage1@1e-4 | shal-Ep17k | shal-Ep17k@1e-4 | TTT-comb | TTT-reproj |
|---|---|---|---|---|---|---|---|---|---|
| 1DSfM | 8.08 | 14.33 | 10.48 | **7.46** | 11.78 | 8.32 | 9.81 | 8.14 | 8.25 |
| BlendedMVS | 8.20 | 6.97 | 31.91 | **2.25** | 12.26 | 19.27 | 24.71 | 10.32 | 19.28 |
| Strecha | 17.20 | **0.09** | 2.04 | 15.95 | 0.13 | 16.65 | 8.45 | 14.53 | 16.72 |

Translation:

| dataset | ESFM | ESFM@1e-4 | RESfM-off | stage1 | stage1@1e-4 | shal-Ep17k | shal-Ep17k@1e-4 | TTT-comb | TTT-reproj |
|---|---|---|---|---|---|---|---|---|---|
| 1DSfM | 16.21 | 16.66 | 10.60 | 10.94 | 13.48 | 12.08 | 12.10 | 13.14 | 13.75 |
| BlendedMVS | 0.106 | 0.109 | 0.330 | **0.041** | 0.185 | 0.271 | 0.330 | 0.156 | 0.271 |
| Strecha | 3.20 | **0.017** | 0.196 | 2.97 | 0.021 | 3.07 | 2.15 | 2.97 | 2.96 |

Seed-robust conclusions: (1) the Strecha lr finding holds across seeds
(ESFM/stage1 @1e-4: 0.09-0.13 deg vs 16-17 at protocol lr); (2) stage1
(deep 2x3, MAD) is the best arm on both 1DSfM (7.46/10.94) and BlendedMVS
(2.25/0.041) — the seed-20 TTT-comb edge on 1DSfM does not survive the
median protocol (8.14); (3) RESfM-off is seed-invariant (deterministic
checkpoint) and remains the Strecha protocol-lr reference (2.04) and the
BlendedMVS outlier (31.91).

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
