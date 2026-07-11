# Cross-dataset discovery report (SPEC_uesfm_combined §2.B R9 — before any dataset code)

Delivered 2026-07-11, per acceptance criterion 8 ("discovery report delivered
before any new dataset code is written"). Searched: all of
`MVG/final-project/` (u-esfm + 7 sibling RESfM checkouts), `MVG/` one level up,
all `code/datasets/` trees, all confs, and RESfM's released `datasets.zip`
(Dropbox link from `resfm-main-orig/README.md`, 487 MB, inventoried in full).

## 1. What exists for 1DSfM / Strecha / BlendedMVS: NOTHING

- **No scene data** (npz/tracks/mat) for any of the three datasets anywhere in
  the project tree or siblings. The only populated dataset folder anywhere is
  `u-esfm/code/datasets/megadepth/` (68 npz).
- **No loaders/confs** mention 1dsfm/strecha/blended/olsson; no conf sets
  `dataset.dataset` to anything but `"megadepth"`.
- **RESfM's release does not ship them**: `datasets.zip` contains ONLY
  `datasets/megadepth/` (68 npz — see §3), and the README checklist items
  "Release all the datasets" and "Release the code for creating tracks from
  images" are both unchecked. No Strecha/BlendedMVS/1DSfM download URLs exist.

**Consequence:** R10 (cross-dataset TTT matrix) and acceptance 9 (one full
cross-dataset variant with real-GT eval) are **blocked on data**, not on code.
Options, in order of preference:
1. Ask the RESfM authors for their preprocessed 1DSfM/Strecha/BlendedMVS
   tracks (their Appendix C construction; keeps the comparison on their data).
2. Build tracks ourselves from the raw datasets with Appendix C parameters —
   NOTE: per R13's track-provenance switch this converts the RESfM comparison
   row on those datasets to "RESfM (retrained, our tracks)".

## 2. What the pipeline needs for a new dataset: data only, no code

- `path_utils.path_to_datasets(dataset)` resolves scenes as
  `code/datasets/<dataset.dataset>/<scan>.npz`; `Euclidean.get_raw_data` is
  fully dataset-agnostic (no per-dataset branching anywhere).
- Required npz schema (verified on `megadepth/0007.npz`): `M` (2m x n tracks),
  `Ns` (inverse calibrations), `Ps_gt` (GT projections), `namesList`; optional
  `outliers2` (GT outlier mask), `outlier_pct`. GT rotations/translations are
  generic arrays (`R_gt`, `T_gt`, `K_gt` also present) — **nothing
  COLMAP-specific at load time**, so Strecha/BlendedMVS real (non-COLMAP) GT is
  an npz-creation concern, not an eval-path concern (R9's real-GT requirement
  is satisfied by encoding their GT into `Ps_gt`/`R_gt`/`T_gt` when building
  the npz).
- So the R9 "plumbing" = drop npz files into `code/datasets/<name>/` + a conf
  with `dataset.dataset = "<name>"` and the scene list. Zero loader changes.
- Per-dataset removal threshold (R9): already conf-exposed —
  `test.outliers_threshold` (classifier path, RESfM: 0.6 MegaDepth/1DSfM, 0.8
  low-outlier datasets) and `test.mad_alpha` (the active MAD path's analogous
  knob). Every run's generated conf is kept (`confs/*_generated/`) and
  snapshotted into checkpoints, so the value used is logged per run.

## 3. Bonus finding: MegaDepth track provenance CONFIRMED (S1.A item 7 closed)

The 68 local `datasets/megadepth/*.npz` are **byte-size-identical, scene for
scene**, to the 68 files in RESfM's released `datasets.zip` — the local copies
are the official files with `" new"` / `"_300"` stripped from the names (e.g.
local `0007.npz` IS the official `0007_300 new.npz` Group-2 subsample; local
`0176.npz` IS `0176_300 new.npz`, closing MULTISCENE_MAP's "missing _300
validation subsets" caveat — they were here all along under plain names).

Consequences:
- We evaluate on RESfM's released tracks ⇒ R13 track-provenance switch does
  NOT trigger; the primary comparison row remains "RESfM (reproduced)" from
  their released `.pth`. No retraining required.
- Group-2 "subsampled to 300" protocol is already satisfied by the data files
  themselves (no local subsampling was silently substituted).
- The reproduction gap on record (Rot 1.90 vs paper 1.29,
  `results/multiscene/resfm_repro_env38*`) cannot be a tracks/subsampling
  difference — data is identical; the BA-environment explanation stands.
