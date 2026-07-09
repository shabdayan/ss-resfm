# RUN_MULTISCENE — Launching U-RESfM multi-scene training

Companion to `MULTISCENE_MAP.md` (discovery) and `claude specs/SPEC_multi_scene.md`.

## What runs

`multiple_scenes_learning.py` (ported from RESfM, see MULTISCENE_MAP.md §1) drives the
existing `train.py:train()` loop with `phase=TRAINING`: samples camera subsets from the
training scenes each epoch, validates per-scene on the validation set at
`train.eval_intervals`, checkpoints every eval, tracks the best model by
`train.validation_metric`, and finally evaluates train/validation/test with BA.

## Conf files

| Conf | Purpose |
|---|---|
| `confs/multiscene_uresfm.conf` | The real run: 27 MegaDepth training scenes, unsupervised `CombinedLoss` |
| `confs/multiscene_uresfm_toy.conf` | 2-scene / 2-epoch smoke test (acceptance #2); `ba.run_ba = False` because BA on a near-untrained model can produce an empty reconstruction and crash |
| `confs/multiscene_resfm_supervised.conf` | Supervised sanity run via `CombinedLossSupervised` (acceptance #3) |

Conf keys that make a run "U-RESfM multi-scene" (vs. the single-scene confs):

- `dataset.train_set` / `validation_set` / `test_set` — scene lists (names match
  `datasets/megadepth/<scan>.npz`; RESfM's `" new"` suffix stripped).
- `dataset.batch_size`, `dataset.min_sample_size`, `dataset.max_sample_size` — scenes
  per optimizer step and the per-scene camera-subsample range.
- `model.type = "SetOfSet.DeepSetOfSetOutliersNet"`, `train.output_mode = 3` — both
  heads active.
- `loss.func = CombinedLoss` — the unsupervised loss (weighted reprojection + adaptive
  confidence-weighted pseudo-label BCE; per-scene statistics each forward). No GT
  labels are touched on the training path.
- `train.validation_metric = ["our_repro"]` with
  `train.validation_metric_higher_is_better = False` — TRAINING-phase best-model
  selection historically maximized (RESfM's label-based "Accuracy"); this key flips it
  for geometric (lower-is-better) metrics. Omitting the key preserves the old behavior.
- `loss.enable_csv_logging = False` — disables CombinedLoss's per-forward pandas/CSV
  tracking (O(steps²); fine for single-scene, not for 27 scenes).
- `loss.func_tuning = ESFMLoss` — what the per-test-scene fine-tune stage will use;
  the later TTT experiment changes only this key (to `CombinedLoss`).
- `train.fine_tune_after_training = False` — the post-training per-test-scene
  fine-tune fan-out stays off (separate TTT task).

## Launch (LSF / Fabric)

Toy smoke test (single GPU, interactive-queue friendly):

```bash
cd /home/projects/bagon/ortalda/MVG/final-project/u-resfm/code
mkdir -p lsf_output/multiscene
bsub -q waic-short \
  -J uresfm_ms_toy \
  -oo lsf_output/multiscene/toy_%J.out \
  -eo lsf_output/multiscene/toy_%J.err \
  -gpu "num=1:j_exclusive=yes:gmem=40G" \
  -R "rusage[mem=50000]" \
  "cd /home/projects/bagon/ortalda/MVG/final-project/u-resfm/code; \
   uv run multiple_scenes_learning.py \
     --conf confs/multiscene_uresfm_toy.conf \
     --phase TRAINING \
     --exp_version multiscene_toy \
     --wandb 0"
```

The 27-scene training run:

```bash
cd /home/projects/bagon/ortalda/MVG/final-project/u-resfm/code
mkdir -p lsf_output/multiscene
bsub -q waic-long \
  -J uresfm_ms_27 \
  -oo lsf_output/multiscene/uresfm27_%J.out \
  -eo lsf_output/multiscene/uresfm27_%J.err \
  -gpu "num=1:j_exclusive=yes:gmem=80G" \
  -R "rusage[mem=64000]" \
  "cd /home/projects/bagon/ortalda/MVG/final-project/u-resfm/code; \
   uv run multiple_scenes_learning.py \
     --conf confs/multiscene_uresfm.conf \
     --phase TRAINING \
     --exp_version multiscene_uresfm \
     --wandb 1"
```

Notes:
- Fabric is initialized exactly as in single-scene runs
  (`single_scene_optimization.initialize_fabric`: CUDA, `devices="auto"`, DDP). To use
  N GPUs on one node, change `-gpu "num=N:..."` — Fabric picks them all up and
  `train.py` already all-reduces losses and all-gathers metrics.
- `--wandb 1` logs to the `RESfM` wandb project under `exp_name`; use `--wandb 0` to
  disable.
- The supervised sanity conf launches the same way with
  `--conf confs/multiscene_resfm_supervised.conf`.

## Where results land

Everything is rooted at the conf's `results_path`
(e.g. `code/results/multiscene/uresfm_27scenes/`):

- `models/Model_Ep<E>.pt` — best checkpoints (`models_all/` = every eval).
  Checkpoints contain `epoch`, `model_state_dict`, `optimizer_state_dict`, and a
  `conf` HOCON snapshot (added for TTT compatibility, R4).
- `Validation_over_epochs.xlsx` — per-scene + Mean validation metrics per eval; the
  same table is printed to the job log each eval.
- `Train_Stats.xlsx`, `Validation.xlsx`, `Test.xlsx`, `myTest.xlsx` — final
  evaluations with the best model.
- `VALIDATION/<scene>/metrics/metrics_Ep<E>.xlsx` — per-eval metric excel.
- `code/` — snapshot of the source tree (`log_code`).
- `wandb/` — wandb run files.

## TTT compatibility (R4 — confirmed, not implemented)

The per-test-scene fine-tune stage is `single_scene_optimization.py:train_single_model`
with `--phase FINE_TUNE`: it loads the best TRAINING checkpoint from
`<results_path>/models/` (`path_utils.path_to_model(conf, Phases.TRAINING, best=True)`,
`single_scene_optimization.py:140`) and selects its loss from `loss.func_tuning`
(`train.py`, FINE_TUNE branch) — fully conf-selectable, nothing hardcoded. The TTT
experiment is therefore: same conf + `loss.func_tuning = CombinedLoss`, run the
fine-tune stage per test scene against the multi-scene `results_path`.

⚠️ Known gap for the TTT task (pre-existing, out of scope here): with
`output_mode = 3`, `Euclidean.get_raw_data` in FINE_TUNE wants *predicted outliers*
from `<results_path>/TEST/<scan>/outliers_results/Final_outliers.npz`, but this repo's
`epoch_evaluation` no longer saves them (the original RESfM one did, via
`dataset_utils.save_outliers`). The TTT task must either restore that save in the test
evaluation or feed the fine-tune stage differently.
