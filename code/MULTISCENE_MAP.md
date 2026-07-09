# MULTISCENE_MAP — How RESfM's multi-scene pipeline maps onto this repo (u-resfm)

Phase A deliverable for `claude specs/SPEC_multi_scene.md`. Reference for the original
pipeline: sibling checkout `../../resfm-main-orig/code/` (RESfM as released).

## 1. Where is the multi-scene path?

**The training loop is here; the entry script is not.**

- `train.py:train()` (line 189) fully supports multi-scene learning: it takes
  `train_data`, `validation_data`, `test_data` loaders plus `phase=Phases.TRAINING`,
  runs `epoch_train` over batches of scenes, evaluates per-scene, tracks a best model,
  and finally evaluates train/validation/test. `epoch_train`/`epoch_evaluation` iterate
  `for batch in loader: for scene in batch:` — a "batch" is a *list of SceneData*, one
  per sampled scene (`ScenesDataSet.collate_fn` is identity).
- `datasets/SceneData.py:create_scene_data_from_list()` and
  `datasets/ScenesDataSet.py` (with `return_all=False`, `min/max_sample_size` camera
  subsampling) are the multi-scene data machinery, carried over intact.
- **Missing:** RESfM's entry point `multiple_scenes_learning.py` was not carried into
  this repo (it exists at `../../resfm-main-orig/code/multiple_scenes_learning.py`;
  `utils/general_utils.py:log_code()` line 46 still lists it, so `log_code` would crash
  here). ⚠️ **Contradiction with the spec** ("the codebase already contains a
  multi-scene path"): the path exists *minus its ~90-line launcher*. The reuse-not-
  rebuild move is to port that launcher, adapting it to this repo's drifted API:
  - `train.train(..., fabric=fabric)` here vs `fabri=fabric` in the original.
  - conf key `train.num_epochs` here vs `train.num_of_epochs` in the original.
  - model construction here needs the `phase` arg for `*OutliersNet` classes,
    Kaiming init, and `torch.compile` (see `single_scene_optimization.py:108-134`).
- Contrast with `single_scene_optimization.py` (this repo's equivalent of
  `single_scene_optimization_merged.py`): it wraps ONE scene in
  `ScenesDataSet([scene_data], return_all=True)` and calls the *same* `train.train()`
  with `phase=OPTIMIZATION` (or `FINE_TUNE`) and no validation/test loaders.

## 2. How does it select model and loss?

Both are conf-driven; **no code change needed to swap either**.

- Model: `general_utils.get_class("models." + conf["model.type"])`. U-RESfM's class
  `SetOfSet.DeepSetOfSetOutliersNet` **exists and is live** (`models/SetOfSet.py:111`;
  the commented-out duplicates start at line 560 — the live copy is above them).
  `*OutliersNet` classes take `(conf, phase)`; the others take `(conf)`.
- Loss: `train.py:207` — `loss_func = getattr(loss_functions, conf["loss.func"])(conf)`.
  For `FINE_TUNE`/`SHORT_OPTIMIZATION` it uses `loss.func_tuning` instead (falls back
  to `loss.func` with a warning).
- Conf keys for the U-RESfM multi-scene run:
  - `model.type = "SetOfSet.DeepSetOfSetOutliersNet"` (+ `num_blocks`, `block_size`, …)
  - `train.output_mode = 3` (both camera and outlier heads → `epoch_train` takes the
    `loss_func(pred_cam, pred_outliers, curr_data, epoch)` branch, `train.py:155`)
  - `loss.func = CombinedLoss` — **in this repo `CombinedLoss` IS the U-RESfM
    unsupervised loss** (`loss_functions.py:182`): `ESFMLoss_weighted` (outlier-score-
    weighted reprojection) + `AdaptiveConfidenceWeightedOutliersLoss` (adaptive
    percentile pseudo-label BCE, `loss_functions.py:456`). No GT labels anywhere.
  - Multi-scene dataset keys the loop reads: `dataset.train_set/validation_set/test_set`,
    `dataset.batch_size`, `dataset.min_sample_size`, `dataset.max_sample_size`.

⚠️ Name collision to be aware of: original RESfM's `CombinedLoss` was *supervised*
(GT-BCE via `OutliersLoss`, 5-arg `forward(pred_cam, pred_outliers, pred_weights_M,
data, epoch)`). This repo repurposed the name for the unsupervised loss (4-arg). The
original supervised combined loss does not exist here, and this repo's `train.py`
calls the 4-arg form. See §"Supervised sanity conf" below.

## 3. What does the multi-scene loop assume about the loss / GT labels?

Label touchpoints on the training path:

- **Loss:** none, if `loss.func = CombinedLoss` (unsupervised). The supervised classes
  `GT_Loss_Outliers`/`OutliersLoss` (`loss_functions.py:157-178`) read
  `data.outlier_indices` (GT) but are only used if the conf selects them.
- **Train metrics:** `epoch_train` (`train.py:161`) computes
  `OutliersMetrics(pred_outliers, curr_data)` every step — this reads GT
  `data.outlier_indices`. Metric-only, never touches the loss. The MegaDepth npz files
  here contain GT `outliers2` (verified on `0012.npz`), so it works; if a scene lacked
  labels it degrades to zeros → some NaN sub-metrics, no crash.
- **Validation / best-model selection — the real integration point.** Original RESfM
  validated multi-scene training with `validation_metric = ["Accuracy"]` (GT-label
  outlier-classification accuracy), which its `epoch_evaluation` merged in via
  `OutliersMetrics`. **This repo's `epoch_evaluation` no longer computes
  `OutliersMetrics`** — only geometric metrics from `evaluation.compute_errors`
  (`our_repro`, `ts_mean`, `Rs_mean`, BA variants…). So for TRAINING phase the
  validation metric must be geometric — which is also exactly what an unsupervised run
  wants. BUT `train.py:389-391` hardcodes the improvement direction by phase:
  TRAINING = higher-is-better (built for Accuracy). Reprojection/rotation/translation
  errors are lower-is-better → **one small code change needed**: a conf-controlled
  direction (e.g. `train.validation_metric_higher_is_better`, default `true` to keep
  existing behavior), used only in the TRAINING branch of `is_better`.
- `evaluation.prepare_predictions`/`prepare_outliers_predictions` copy
  `data.outlier_indices` into outputs — always present (zeros fallback in
  `Euclidean.get_raw_data:114`), no crash path.

## 4. Where is the train/val/test split defined?

**Found — but not in this repo.** It lives in the original checkout:
`../../resfm-main-orig/code/confs/RESFM_Learning.conf` lines 18-20:

- 27 training scenes: `0156, 0860, 0412, 0217, 0181, 0285, 0214, 0041, 0275, 0186,
  0474, 0476, 0768, 0204, 0733, 0047, 0229, 0175, 0349, 0107, 0303, 0058, 0286, 0062,
  0478, 0271, 0360` (listed there with a `" new"` suffix, e.g. `"0156 new"`).
- validation: `5015, 0176_300, 0299_300, 0290`; test (in that conf): `5015`.
- Beware: lines 23-25 of that conf are a **debug override** (`0012_300` for all three
  sets) that HOCON-shadows the real lists — don't copy them blindly.

Mapping to this repo's data (`datasets/megadepth/`, 69 scene files, resolved by
`path_utils.path_to_datasets` → `<scan>.npz`):

- Scene files here are named **without** `" new"` (e.g. `0156.npz`). All 27 training
  scenes exist. ✅
- The `_300` validation subsets (`0176_300.npz`, `0299_300.npz`, `0012_300.npz`) do
  **not** exist here; the full `0176.npz`/`0299.npz`/`0012.npz` do. The new conf must
  either use the full scenes for validation or the `_300` files must be copied from
  wherever RESfM's data drop keeps them. Not a blocker for training.

## 5. What shapes flow through the loss in multi-scene mode?

- A dataloader batch is a `list` of `SceneData` (batch_size scenes, each a random
  camera subsample per `ScenesDataSet.__getitem__`). `epoch_train` calls the loss
  **once per scene**, then sums into `batch_loss` before one optimizer step — so
  heterogeneous camera/point counts never meet in one tensor.
- Inside one scene the alignment invariant is: `pred_outliers` has one entry per
  sparse observation (`data.x.indices`, built by `M2sparse` from
  `get_M_valid_points(M)`), and `reproj_err[data.valid_pts]` uses the *same* mask, so
  `(1 - pred_outliers) * reproj_err` shapes match — the historical 61354-vs-370987
  mismatch can only recur if `x` and `valid_pts` are built from different `M`s
  (they're both built in `SceneData.__init__` from one `M`, including for sampled
  subsets — safe).
- `AdaptiveConfidenceWeightedOutliersLoss` is **stateless**: thresholds are
  `torch.quantile` over the current scene's errors each forward. Per-scene-per-forward
  statistics (spec R2) hold by construction; nothing is pooled or cached across scenes.
- ⚠️ Landmine found: `ScenesDataSet.__getitem__` (lines 29-31) *mutates*
  `self.min/max_sample_size` when it encounters a scene with <50 cameras — a side
  effect that changes sampling for **all subsequent scenes** in a mixed-size training
  set. Pre-existing behavior; noted, not fixed (spec: minimal diff).
- ⚠️ Throughput note: `CombinedLoss` appends a row to a pandas DataFrame and
  periodically writes CSV **every forward** (`loss_functions.py:266`), and
  `train.py:156-157` calls `finalize_and_save()` (full CSV dump + console summary)
  after *every scene*. Tolerable for a toy run; for the 27-scene run this is O(steps²)
  DataFrame growth and heavy log spam — flagged for Phase B as the one place a
  guarded, conf-off switch may be justified.

## 6. Where is the test-time fine-tune path? (TTT vehicle)

- Original protocol: `multiple_scenes_learning.py:70-90` — after training, rank 0
  loops over `dataset.test_set`, sets `conf.dataset.scan`, and calls
  `train_single_model(conf, device, Phases.FINE_TUNE)` per scene with
  `train.optimization_num_of_epochs = 1001` (the "1K fine-tune"),
  `optimization_eval_intervals = 250`, `optimization_lr = 5e-3` (all conf keys in
  `RESFM_Learning.conf`).
- In this repo the same vehicle exists: `single_scene_optimization.py:train_single_model`
  with `phase=Phases.FINE_TUNE`:
  - loads the best TRAINING checkpoint: `path_utils.path_to_model(conf,
    Phases.TRAINING, best=True)` → `<results_path>/models/Model_Ep<latest>.pt`
    (line 140), with `_orig_mod.` prefix reconciliation for `torch.compile`.
  - `Euclidean.get_raw_data` (line 134) additionally loads *predicted* outliers from
    the TEST phase and prunes M + keeps the largest connected component when
    `output_mode == 3`.
  - **Loss is conf-selectable** ✅ (R4): `train.py:197-203` uses `loss.func_tuning`
    for FINE_TUNE — nothing hardcoded. RESfM set `func_tuning = ESFMLoss` (plain
    unsupervised reprojection, as the spec expected). The later TTT experiment is
    literally `func_tuning = CombinedLoss` in a conf.
  - Parameters updated: the `*OutliersNet` classes set `mode = 1` when
    `phase is FINE_TUNE` (`models/SetOfSet.py:211-212`) — outlier head frozen,
    camera/point heads + trunk train.
  - Uses `validation_metric_fine_tuning` (lower-is-better branch — correct already).
- Checkpoint contents today: `{'epoch', 'model_state_dict', 'optimizer_state_dict'}`
  (`train.py:378-382`). **No conf snapshot** — spec R4 asks for one; adding a conf
  dump next to the checkpoint (or into the dict) is a Phase B item. Nothing in the
  state dict is scene-count-specific (fully equivariant architecture), so a
  multi-scene checkpoint loads into a single-scene fine-tune unchanged.

## Supervised sanity conf (spec acceptance #3) — contradiction to flag

The spec assumes "RESfM's own supervised confs" exist here and must keep running.
**They don't exist in this repo**: no `RESFM_Learning.conf`, no supervised
`CombinedLoss` (the name now denotes the unsupervised loss), and this repo's
`epoch_train` never passes `pred_weights_M`. What still runs supervised here is the
GT-BCE building blocks (`OutliersLoss`, usable with `output_mode = 2`). To satisfy
acceptance #3 verbatim we must *add* a supervised multi-scene conf that maps onto this
repo's surviving API — e.g. `loss.func = CombinedLossSupervised` (a thin port of the
original class under a non-colliding name) or a `output_mode = 2` + `OutliersLoss`
conf. Decision recorded in Phase B; the original untouched conf cannot "still run"
because it never existed here.

## Other landmines confirmed (spec R3)

- `is_better` on a Series: already mitigated in this repo — `train.py:366-368`
  reduces to `…mean(axis=0).values.tolist()` and takes `metric[0]`; multi-scene
  validation keeps the `"Mean"` row (`evaluation.organize_errors`), so the TRAINING
  branch `validation_metrics.loc[["Mean"], …]` works. Single-item-list metric only.
- `train_errors["Scene"]`: `epoch_evaluation` always writes a `Scene` key per scene
  and `organize_errors` indexes on it; `single_scene_optimization.py:202` drops
  `"Mean"` — both shapes intact as long as we don't touch `organize_errors`.
- Fabric: this repo's `train()` takes `fabric=` (keyword renamed from RESfM's
  `fabri=`); `initialize_fabric()` in `single_scene_optimization.py` builds the
  DDP fabric; LSF launch goes through `bsub -q waic-<queue> … uv run <script>`
  (`run_single_scene_optimization.sh:1798`). The ported multi-scene launcher must use
  this repo's fabric setup, not the module-level `fabric.launch()` of the original.
- `datasets/ScenesDataSet.py` had a broken import (`import utils.dataset_utils` but
  bare `dataset_utils` used at line 41) that only the multi-scene sampling path
  (`return_all=False`) executes — NameError on the first sampled batch. Fixed
  (surfaced in the toy run's CPU smoke test).
- BA at evaluation crashes (`ba_advanced.pycolmap_to_batch_matrix`: `max()` of empty
  `point3D_ids()`) when the model is so poor that colmap keeps zero 3D points —
  surfaced when the 2-epoch toy model hit the final BA evaluation. Not fixed in code
  (only affects evaluating near-untrained models); the toy conf sets
  `ba.run_ba = False`, the 27-scene conf evaluates with BA only after training
  (`ba.only_last_eval = True`), matching RESfM's own protocol.
