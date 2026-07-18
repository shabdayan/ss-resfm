# Cross-dataset evaluation results (R10) — 2026-07-18, seed 20, our tracks

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
   RESfM-off), pointing at those scenes' tracks/GT (gt_bundle) rather than any
   model. Open item.
6. Shallow-adaptive caveats: best checkpoint Ep17000 (validation plateaued;
   training ran to Ep19999); the earlier Ep3000 rows in the CSV are superseded
   but kept for the training-progress comparison.
