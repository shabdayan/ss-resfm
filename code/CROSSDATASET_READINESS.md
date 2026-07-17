# Cross-Dataset Readiness Report — 1DSfM / Strecha / BlendedMVS

Per `claude specs/TASK_crossdataset_readiness.md` (discovery + smoke only).
Delivered 2026-07-17. Smoke budget: **50 fine-tune epochs** (config 3: frozen,
0 steps) — all metric values below are **sanity indicators only, NOT results**.

## 1. Dataset table

| Dataset | Tracks present | GT present / format | GT readable by eval | Loader/conf | Blockers |
|---|---|---|---|---|---|
| 1DSfM | 10/10 target scenes (`code/datasets/1dsfm/*.npz`) | gt_bundle.out (dataset's reference *Bundler* reconstruction, not COLMAP) encoded into `Ps_gt` at track build | ✅ generic arrays | ✅ `dataset.dataset="1dsfm"`, zero code changes | none |
| Strecha | 4/4 (`entry-P10, fountain-P11, Herz-Jesu-P8/P25`) + 2 extra scenes in raw | LIDAR-derived released `.P`/`.camera` → `Ps_gt` (non-COLMAP ✅) | ✅ | ✅ | none |
| BlendedMVS | 9 scenes (hash-named low-res, e.g. `5a48ba95…` — NOT "scene0-3" as task assumed) | synthetic MVSNet `cams/*_cam.txt` → `Ps_gt` (non-COLMAP ✅) | ✅ | ✅ | scene-selection decision (which 4 of 9) |

- npz schema is identical to MegaDepth **including `outliers2` GT masks and
  `outlier_pct`** (Ellis Island 56.3%, Strecha ~2.7%, BlendedMVS ~1.5%).
- Raw data provenance + checksums: `datasets/MANIFEST.md`; builders in
  `code/build_tracks/` (Appendix-C-style construction).
- **R13 provenance note:** these are OUR tracks (RESfM never released theirs
  for these datasets) ⇒ RESfM comparison rows here are
  "RESfM (released checkpoint, our tracks)".

## 2. Checkpoint inventory

| Row | Checkpoint | Loadable | Provenance |
|---|---|---|---|
| RESfM (released) | `pretrained/pretrained_model.pt` — wrapped dict, SetOfSetOutliersNet 1×3 + outlier head | ✅ (used by the MegaDepth reproduction) | pre-provenance (authors'); guard-clean |
| ESFM (ours) | `results/multiscene/uesfm_27scenes_shallow_reproj/models/Model_Ep19999.pt` | ✅ | clean 27-scene training. **Caveat:** Deep-*class* at 1×3 (enhanced blocks), not byte-original ESFM; no released ESFM multi-scene ckpt exists |
| U-ESFM (shallow 1×3 CombLoss) | `results/multiscene/uesfm_27scenes_sos_adaptive_1gpu_any80g/models/Model_Ep16500.pt` | ✅ | clean 27-scene training, collapse-checked |

## 3. Readiness matrix — ALL 15 CELLS PASS (plumbing-wise)

Rot° / Trans / Nr at smoke budget (one scene each: Ellis_Island 227 cams,
entry-P10 10 cams, BlendedMVS `5a48ba95…` 33 cams). Removal threshold 0.6
(1DSfM) / 0.8 (Strecha, BlendedMVS) per RESfM convention, recorded in each
generated conf (`confs/crossdataset_smoke_generated/`). Every run dir carries
conf snapshot + git hash + seed; configs 4/5 logged collapse stats.

| Config | 1DSfM | Strecha | BlendedMVS |
|---|---|---|---|
| 1 ESFM plain | PASS 7.6/17.6/124 | PASS 18.8/7.5/10 | PASS 36.1/0.54/26 |
| 2 RESfM protocol | PASS 18.2/14.7/189 | PASS **0.01/0.007/10** | PASS 81.2/0.66/19 |
| 3 U-ESFM frozen | PASS 17.5/17.1/189 | PASS 28.4/8.5/10 | PASS 80.1/0.61/21 |
| 4 U-ESFM protocol | PASS 14.1/18.1/170 | PASS 31.0/8.5/10 | PASS 51.1/0.65/18 |
| 5 U-ESFM full TTT | PASS 17.7/16.0/187 | PASS 27.4/7.4/9 | PASS 83.2/0.51/21 |

Sanity notes: numbers are rough as expected at 1/20th of the protocol budget
under dataset shift; RESfM's near-perfect entry-P10 (0.01°) confirms the
end-to-end path (their checkpoint generalizes to the easy 10-camera scene).
1DSfM translation values are in the dataset's own (bundler) scale.

## 4. Gap list — to turn readiness into results

1. **Full-budget runs** (1001-epoch FT): ~18 scenes × 5 configs ≈ 90 GPU-jobs
   (~30-60 min each). Runner exists (`run_crossdataset_smoke.sh` — set
   `num_epochs=1001`, eval_intervals=250; or fold into run_multiscene_eval
   per-dataset). NOTE: the other session already queued a `repro_s20_*`
   cross-dataset fan-out — coordinate to avoid duplication.
2. **BlendedMVS scene selection**: pick 4 of the 9 built scenes (RESfM's exact
   4 unknown — their paper lists "scene0-3"; decide + document).
3. **Threshold calibration** (parent-spec R12): our head flags ~0.35-0.4 at
   0.6 on MegaDepth; on low-outlier Strecha/BlendedMVS the 0.8 convention is
   untested at full budget — the removal_threshold sweep applies unchanged.
4. **Optional ESFM purity**: an original-class ESFM baseline would need
   training SetOfSetNet 1×3 multi-scene (~1 day GPU); current ESFM row uses
   our Deep-class 1×3 reproj arm.

## 5. Code changes made for the smoke tests

- `run_crossdataset_smoke.sh` (new): 5-config × 3-dataset smoke fan-out, all
  through the existing conf-driven `single_scene_optimization.py --phase
  FINE_TUNE` pipeline. No pipeline-core changes were needed — the readiness
  matrix ran on conf generation alone (dataset key, thresholds, budgets,
  output modes, loss selection).
