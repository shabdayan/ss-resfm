# SPEC: Run U-RESfM Through the Existing RESfM Multi-Scene Pipeline

**Repo:** `deep-unsupervised-outlier-detection-esfm/` (built on RESfM/ESFM code)
**Priority:** P0 — blocks Table 1 of the paper. Must be usable today.
**Author of spec:** Ortal (via Claude). **Implementer:** Claude Code.

---

## 1. Goal

The codebase already contains a **multi-scene training configuration/path
inherited from RESfM** (ESFM/RESfM both support "learning from multiple scenes").
Our unsupervised variant, **U-RESfM** (Deep ESFM-style equivariant architecture +
adaptive MAD-based confidence-weighted outlier loss), was so far developed and run
mainly through the single-scene path (`single_scene_optimization_merged.py`).

The task: **understand the existing multi-scene pipeline and make U-RESfM's model
and loss run through it**, so we can train jointly on RESfM's 27 MegaDepth training
scenes and evaluate on their test split with their protocol. Do NOT build a new
multi-scene system — reuse what's there.

A later phase (out of scope, but design for it) loads the multi-scene checkpoint
and fine-tunes per test scene with the same unsupervised loss (TTT). Nothing you
do may block that.

## 2. Phase A — Discovery (do this before touching any code)

Produce a short written map (`MULTISCENE_MAP.md`, ~1 page) answering:

1. **Where is the multi-scene path?** Find the multi-scene entry point / conf files
   inherited from RESfM (look for a multi-scene optimization script, conf keys like
   dataset/scene lists, phases, and how RESfM's own experiments were launched).
   Contrast it with `single_scene_optimization_merged.py`.
2. **How does it select model and loss?** Model classes load via
   `general_utils.get_class("models." + model_type)` from the conf; confirm the
   same conf-driven pattern selects the loss. List exactly which conf keys must
   change to swap in U-RESfM's model class (in `models/SetOfSet.py`, e.g. the
   Deep/Outliers variant — verify the class exists and is not commented out; this
   has bitten us before) and U-RESfM's loss (in `loss_functions.py`).
3. **What does the multi-scene loop assume about the loss?** RESfM's multi-scene
   path was built for a *supervised* BCE outlier loss with GT labels. Identify
   every place the pipeline passes labels, expects a labels tensor, or computes
   label-dependent metrics — these are the integration points where the
   unsupervised loss differs. U-RESfM's loss needs no labels; it needs per-scene
   reprojection errors and computes MAD statistics from them.
4. **Where is the train/val/test split defined?** Locate RESfM's 27-scene training
   split and test scene lists in the repo/confs. Record the exact scene names and
   where they live. If the split is NOT in the repo, stop and report — this is a
   go/no-go fact we've never verified.
5. **What shapes flow through the loss in multi-scene mode?** Scenes have
   heterogeneous camera/point counts. Confirm how per-scene batches reach the loss
   (past bug: `(1 - pred_outliers) * reproj_err` shape mismatch, 61354 vs 370987).
6. **Where is the test-time fine-tune path?** RESfM's evaluation protocol includes
   a per-test-scene "1K fine-tune" stage (inherited from ESFM; see also
   `single_scene_optimization_2nd_stage.py`-style scripts in this lineage). Locate
   that stage, document which loss it uses (expected: plain unsupervised
   reprojection loss, since the supervised BCE outlier loss has no labels at test
   time), which parameters it updates, how many steps/LR, and how it's launched.
   This stage is the intended vehicle for the later TTT phase — the TTT experiment
   should amount to running this stage with U-RESfM's full adaptive outlier loss.

Deliver Phase A findings before starting Phase B. If discovery contradicts this
spec anywhere, report the contradiction — don't silently improvise.

## 3. Phase B — Integration requirements

### R1 — Conf-driven swap, minimal diff
- Create new conf file(s) (e.g. `confs/multiscene_uresfm.conf`) that run the
  EXISTING multi-scene pipeline with U-RESfM's model class and unsupervised loss.
- Prefer conf changes over code changes. Code changes only where the pipeline
  hard-assumes the supervised loss (Phase A, item 3).

### R2 — Unsupervised loss in the multi-scene loop
- The adaptive MAD-based loss must compute its statistics **per scene per forward
  pass** — never pooled or cached across scenes.
- Remove/bypass any GT-outlier-label plumbing on the training path without breaking
  RESfM's own supervised confs (they should still run — we may want their numbers
  as a sanity reference).
- Any label-dependent *metric* (e.g. outlier-classification accuracy) should be
  skipped gracefully when labels aren't used, not crash.

### R3 — Known landmines (from past debugging of this repo)
- `train.py` best-model check: `is_better` once crashed on a pandas Series
  ("truth value of a Series is ambiguous"). Multi-scene validation yields per-scene
  DataFrames; the best-model criterion must reduce to an explicit scalar.
- `train_errors` has a `"Scene"` column only in multi-scene mode (single-scene
  KeyError historically). Don't break either shape.
- Fabric (`fabri=fabric`) is threaded through training; the multi-scene run must
  still launch under the existing Fabric/LSF setup.

### R4 — Checkpointing (TTT-compatible)
- Multi-scene checkpoints must save model state_dict + conf snapshot + epoch and be
  loadable by the EXISTING test-time fine-tune stage (Phase A, item 6). Nothing
  scene-count-specific baked in.
- Confirm (don't implement) that the fine-tune stage's loss is conf-selectable, so
  the later TTT experiment is just: load multi-scene checkpoint → run fine-tune
  stage with U-RESfM's adaptive loss instead of plain reprojection. If the loss is
  hardcoded there, note the exact line(s) in MULTISCENE_MAP.md but leave the change
  for the TTT task.

### R5 — Backward compatibility
- Single-scene U-RESfM confs and RESfM's original multi-scene confs must both still
  run unchanged. New behavior only via the new conf files.

## 4. Non-goals
- No TTT implementation (separate task; only R4 must stay compatible).
- No new multi-scene infrastructure, no refactors, no LoRA, no architecture or loss
  changes beyond what R2 strictly requires.
- No new dependencies (Python 3.9 venv as-is).

## 5. Acceptance criteria
1. `MULTISCENE_MAP.md` exists and answers all five Phase A questions.
2. A 2-scene toy run with `multiscene_uresfm.conf` completes 2 epochs: no label
   tensors touched by the loss, per-scene + mean validation metrics printed, best
   checkpoint written.
3. RESfM's original supervised multi-scene conf still starts and runs a few steps.
4. An existing single-scene U-RESfM conf still runs identically (fixed-seed smoke
   test, first few losses match pre-change).
5. `RUN_MULTISCENE.md`: conf keys used, exact Fabric/LSF launch command for the
   27-scene training run, where results land.

## 6. Order of work
1. Phase A discovery → write MULTISCENE_MAP.md → report findings.
2. New conf + minimal loss-plumbing changes (R1–R2).
3. Toy-split run; fix landmines as they surface (R3).
4. Checkpoint/compat checks (R4–R5), write RUN_MULTISCENE.md.

Keep the diff minimal — this is a wiring task, not a rewrite.
