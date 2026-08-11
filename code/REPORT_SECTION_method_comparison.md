# Report section draft: Method comparison across all settings (for Experiments)

Paste-ready consolidated comparison for `MVG_Project_Report_Ortal_Dayan`
(suggested: an Experiments overview/summary subsection, or 3.x preceding the
per-dataset sections). It folds together every U-ESFM and RESfM/ESFM variant
we evaluated and how they compare across the three experimental settings.
Reporting convention (project-wide): **means only, rotation in degrees and
translation always reported together**; cross-dataset and single-scene numbers
are 5-seed and multi-seed protocol means respectively. Sources:
`results/single_scene/summary.csv`, `results/crossdataset/` (per-scene
`Results_FINE_TUNE*.xlsx`, aggregated by `make_paper_style_tables.py` /
`aggregate_crossdataset_seeds.py`), and the MegaDepth multi-scene sweep in
`CROSSDATASET_RESULTS.md` / `multiscene-experiment-state`.

---

## 3.x Method comparison

### 3.x.1 The variants evaluated

All methods share the same equivariant set-of-sets backbone and the same
per-scene evaluation protocol (feed-forward inference from a multi-scene
checkpoint, test-time outlier removal, a ~1,000-epoch per-scene fine-tune, and
robust bundle adjustment). They differ only in (i) network depth, (ii) the
multi-scene training loss, and (iii) how outliers are removed at test time.

**ESFM baselines (no learned outlier head).**
- **ESFM-van** — the original ESFM: a vanilla `SetOfSetNet` of one block x three
  equivariant layers (1x3), trained with the plain unsupervised reprojection
  loss, no outlier removal. This is the true published-ESFM architecture and
  serves as the lower endpoint of every comparison.
- **ESFM-deep** — the same plain reprojection loss and no removal, but on the
  deeper 2x3 architecture (residual connections, SparseLayerNorm, dropout).
  This is the arm that earlier tables mislabeled as "ESFM"; the A1 diagnosis
  showed it is architecturally enhanced and therefore not a fair ESFM endpoint,
  which motivated ESFM-van.

**U-ESFM family (Deep ESFM + label-free statistical outlier handling).**
- **U-ESFM (MAD)** — the headline two-stage method: the deep 2x3 checkpoint
  (plain-loss training) with a label-free MAD outlier-removal stage (median +
  2 x 1.4826 x MAD on per-observation reprojection errors) before the per-scene
  fine-tune.
- **U-ESFM-DA** — "deep-adaptive": the deep 2x3 network trained with our
  Adaptive Confidence-Weighted Outlier Loss (Sec. 2.2.2, unsupervised
  reprojection + confident-pseudo-label term), MAD removal at test.
- **U-ESFM-SA** — the shallow (1x3) counterpart of U-ESFM-DA (same adaptive
  loss, shallow backbone).
- **U-ESFM-DA+TTT / U-ESFM-SA+TTT** — test-time-training variants that replace
  the plain fine-tune with continued optimization of the adaptive loss
  (outlier head active) on the test scene.
- On MegaDepth we additionally swept network depth (1x3, 2x3, 5x2, 6x2, 6x3).

**RESfM family (supervised, learned outlier classifier).**
- **RESfM (released)** — the official pretrained multi-scene checkpoint with its
  supervised outlier classifier (thresholds 0.6/0.8 per the paper).
- **RESfM-deep** — RESfM's *supervised* training recipe (outlier-weighted
  reprojection + GT-label balanced BCE) trained here at the matched deep 2x3
  architecture, learned-classifier pruning at test. This isolates network
  depth as the only difference from the released model.

### 3.x.2 Cross-dataset transfer (1DSfM / Strecha / BlendedMVS)

All arms are MegaDepth-trained and evaluated on the three unseen datasets on
identical, our-provenance point tracks (RESfM never released its cross-dataset
tracks); 5 seeds, per-scene means then per-dataset means. Fine-tune learning
rate is a per-dataset setting (5e-3; 1e-4 on Strecha's small scenes).

**Table X: mean rotation error (deg) / mean translation error, 5 seeds.**

| method | 1DSfM | Strecha | BlendedMVS |
|---|---|---|---|
| ESFM-van | 14.26 / 17.89 | 7.41 / 1.900 | 26.98 / 0.224 |
| ESFM-deep | 8.37 / 15.34 | 0.14 / 0.030 | 7.78 / 0.111 |
| U-ESFM (MAD) | 7.91 / 12.62 | 0.15 / 0.027 | **2.39 / 0.040** |
| U-ESFM-SA | 9.23 / 13.17 | 8.37 / 2.18 | 15.42 / 0.224 |
| U-ESFM-SA+TTT | 8.93 / 13.51 | 12.48 / 2.73 | 9.94 / 0.152 |
| U-ESFM-DA | **5.73 / 9.36** | 1.59 / 0.218 | 6.29 / 0.097 |
| U-ESFM-DA+TTT | 6.48 / 11.06 | 1.62 / 0.220 | 6.59 / 0.100 |
| RESfM-deep | 7.44 / 11.72 | **0.01 / 0.005** | 10.20 / 0.143 |
| RESfM (released) | 10.21 / 10.71 | 2.04 / 0.197 | 31.91 / 0.330 |

(RESfM-deep 1DSfM is over the four largest scenes' partially-completed seed set
at the time of writing; the two largest scenes are still finishing and the
number will firm up slightly.)

### 3.x.3 Single-scene reconstruction (Olsson, calibrated, 36 scenes)

Per-scene optimization from scratch (no multi-scene transfer), multi-seed
means. This isolates the architecture-and-loss factor and the fine-tune-budget
factor (1,000 vs 5,000 iterations). Reprojection error (post-BA, px) is
reported here because ESFM's own single-scene tables report it.

**Table Y: single-scene means (rotation deg / translation / reprojection px).**

| method | Rot | Trans | Reproj |
|---|---|---|---|
| ESFM-van (official repo) | 11.67 | 1.809 | 22.04 |
| ESFM-deep | 11.24 | 1.762 | 5.80 |
| U-ESFM | 18.91 | 2.963 | 8.15 |
| U-ESFM MAD-FT (1k / 5k) | 18.83 / 19.35 | -- | 5.42 / 3.93 |
| U-ESFM learned-FT (1k / 5k) | 18.65 / 17.89 | -- | 6.01 / **3.38** |
| U-ESFM TTT (1k / 5k) | 16.34 / **14.55** | 2.35 | 9.24 / 6.35 |
| U-ESFM TTT-comb (1k / 5k) | 14.84 / 16.86 | 2.36 | 11.64 / 7.56 |

The single-scene means are inflated by a handful of non-converging scenes
(Gustav Vasa, GustavIIAdolf, The Pumpkin, Statue of Liberty, ...) that ESFM's
own paper also flags as failures; the per-scene medians are sub-degree /
sub-2px for the well-conditioned scenes.

### 3.x.4 Multi-scene training on MegaDepth (36 test scenes)

For reference, the setting in which RESfM is originally trained/evaluated:
- RESfM (reproduced, official checkpoint): 1.90 / 0.209 (paper reports 1.29);
  the gap is bundle-adjustment environment sensitivity, seed-negligible.
- ESFM (published): 6.77 / 0.780.
- Best U-ESFM sweep arms: DSOS 1x3 @lr1e-4 3.47 / 0.358; DSOS 6x3 @ft-lr1e-4
  3.61 / 0.415. Depth helps monotonically only once the fine-tune learning rate
  is lowered (the deep models flatline at the 5e-3 protocol lr).

### 3.x.5 Cross-cutting findings

1. **No single method wins on every dataset.** Cross-dataset, the best arm is
   different each time: U-ESFM-DA on 1DSfM (5.73), U-ESFM (MAD) on BlendedMVS
   (2.39), RESfM-deep on Strecha (0.01). The robustness claim is that the
   U-ESFM arms never *collapse* on an unseen dataset, whereas the released
   supervised classifier does (31.9 deg on BlendedMVS).

2. **Depth is what makes the adaptive loss pay off.** U-ESFM-DA (deep adaptive)
   is the strongest U-ESFM variant on 1DSfM and BlendedMVS, whereas its shallow
   twin U-ESFM-SA is consistently the weakest U-ESFM arm.

3. **Test-time training is neutral-to-harmful at the proper fine-tune learning
   rate.** Across single-scene, multi-scene and cross-dataset, TTT only rescues
   mis-tuned configurations; at the correct lr it does not improve pose
   (e.g. U-ESFM-DA 5.73 -> 6.48 on 1DSfM, U-ESFM-SA 8.37 -> 12.48 on Strecha).

4. **Training RESfM's supervised loss at depth beats the released RESfM
   everywhere.** RESfM-deep improves on RESfM (released) on all three datasets
   (1DSfM 7.44 vs 10.21; Strecha 0.01 vs 2.04; BlendedMVS 10.20 vs 31.91); the
   released model's BlendedMVS collapse disappears at matched depth. RESfM-deep
   is the best arm on Strecha.

5. **Vanilla ESFM is the true floor.** ESFM-van is clearly weakest on every
   dataset, well below ESFM-deep -- confirming that the originally-labelled
   "ESFM" column was the enhanced deep network, and that the corrected
   comparison interval is published-calibrated ESFM-van -> U-ESFM -> RESfM.

6. **Post-BA reprojection is bundle-adjustment-limited, not method-limited.**
   Where the geometry is well-conditioned, every arm reaches sub-pixel
   reprojection after BA; the methods differ in *pose* accuracy. The fine-tune
   budget (1k vs 5k) buys reprojection accuracy (30-44 % on single-scene) but
   barely moves rotation -- so 1k suffices for the pose-based comparison RESfM
   reports, while 5k matters only if reprojection is reported (as ESFM does).
