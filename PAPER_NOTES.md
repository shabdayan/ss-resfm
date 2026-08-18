# Paper notes — single-scene ESFM vs U-ESFM (Olsson, calibrated)

Working notes for the ICLR write-up: the methodology decisions in the single-scene
benchmark, and **why each is fair when comparing against ESFM**. The guiding rule
throughout: any change to inputs, training budget, or post-processing is applied
**identically to every method**, or reported separately — so the head-to-head stays
clean. Where a change is applied to only one method it is called out explicitly.

Results/commits referenced live in `u-esfm/code/results/single_scene/`
(`summary.csv`, `summary_table.{md,tex}`, `REPRO.md`).

---

## 1. Benchmark setup (fairness ground rules)

- **Identical inputs.** All methods read the same shared `datasets/Euclidean/<scene>.npz`
  point tracks (symlinked into both repos). No method re-generates or filters tracks.
- **Identical training budget.** 100k epochs, lr 1e-4, MultiStepLR milestones
  [50k,70k,90k], γ=0.1, Adam, best-checkpoint by `our_repro` (reprojection error),
  full scene per epoch. Verified against the ESFM paper and its released code (§4).
- **Identical post-processing.** All methods evaluated by ONE shared harmonized
  evaluator (`evaluate_single_scene.py`): same Sim(3) alignment to GT, same
  rotation/position/reprojection metrics, and — when reported — the same shared
  pycolmap bundle adjustment applied to every method's raw cameras. No method uses
  its own BA in the reported numbers.
- **Arms.** `esfm` (official Equivariant-SFM code), `esfm_rc` (plain ESFM run through
  the RESfM-lineage code — controls for codebase drift), `uesfm` (our method),
  plus reference column `esfm_paper` (published Table-2 numbers).

## 2. Report-faithful adaptive loss (applied to U-ESFM only — a method fix, not an advantage)

- **What.** The benchmarked `CombinedLoss` had silently used *plain* `ESFMLoss` for
  the geometry term, so the outlier detector's scores never weighted the reprojection
  loss — contradicting the report (sec 2.2.2: "the outlier predictions are then used
  to weight the reprojection loss") and the report's own appendix code
  (`ESFMLoss_weighted`). Fixed: `loss.reproj_weighting ∈ {none, weighted,
  weighted_detach}`; `weighted` restores `(1−score)·err` per the report.
- **Fairness.** This only makes U-ESFM *match its own paper's method*; it does not
  touch ESFM. It is a correctness fix to our method, disclosed as such.
- **Also fixed (scale-aware threshold):** `min_threshold_separation` defaulted to
  0.5 (pixel scale), but reprojection errors here are in normalized image coords
  (~0.01–0.2), so 0.5 exceeded the whole error range and eliminated all confident
  inliers. Now expressed as a fraction of the median error (scale-aware).

## 3. Epoch-bookkeeping alignment with ESFM (applied to U-ESFM — removes a confound)

- **What.** Fresh U-ESFM runs did epochs 1..99999 (99999 iters, first eval at 5000);
  ESFM does `range(num_of_epochs)` = 0..99999 (100000 iters, eval at epoch 0). Set
  `resuming_epoch=-1` for fresh runs so both match exactly (100000 iters, eval at 0).
- **Fairness.** Makes U-ESFM's budget/eval schedule *identical* to ESFM's. Forward-
  looking; the frozen results used the old off-by-one, which never changed model
  selection (0.001% of the budget).

## 4. Sequential-optimization fallback (applied to BOTH arms — symmetric)

- **What.** The ESFM paper (Table 8) reruns 10 hard calibrated scenes with a
  sequential schedule: greedily grow the image subset by shared tracks, 1000 epochs
  per added image, then a full-scene stage. The official ESFM code has this
  (`train.sequential`); U-ESFM did not — we ported it (same `get_subset` greedy
  ordering) and ran it on **both** arms (`esfm_seq`, `uesfm_seq`).
- **Fairness.** Applied identically to both methods; it is a general optimization
  schedule, not method-specific. Reported as its own arm.
- **Finding (important, and it corrects an earlier 2-scene read).** Sequential is
  **high-variance per scene** for both methods: it rescues catastrophic failures
  (U-ESFM Nijo 7.64°→0.08°, Smolny 21.45°→0.03°, Some Cathedral 1.37°→0.03° post-BA)
  but *induces* new ones (U-ESFM Alcatraz Courtyard 0.04°→28°; ESFM Water Tower
  0.23°→20.69°). A 100k-epoch-final diagnostic proved the regressions are a
  **bad-basin trap**, not an under-training budget issue (32.3° with 100k final ≈
  32.5° with 20k final). Blanket sequential is ~neutral on the mean.
- **Correct usage (as the paper does): per-scene fallback, best-of(standard,
  sequential).** Mean post-BA rotation over the 10 scenes:
  U-ESFM 5.85°→**2.80°**, ESFM 4.65°→**2.51°**.
- **Recommended paper framing.** Use sequential as a *selective fallback* (report
  best-of), applied uniformly to all methods, and state that the reference ESFM
  code applies it selectively too.

## 5. Drinking Fountain — reflection degeneracy (EXCLUDED, not "fixed")

- **Symptom.** Both ESFM and U-ESFM converge to ~23° post-BA rotation on Drinking
  Fountain, where the paper reports 0.007°. It is the only scene neither standard
  nor sequential reproduces, and both arms fail *identically* (shared init/env).
- **Diagnosis (verified).** Our reconstruction is a **global reflection** of the true
  geometry: the Sim(3) alignment to GT needs a negative scale (det<0), and mirroring
  the reconstruction drops the rotation error 25.8°→6.0°. The scene is
  near-degenerate — its 14 camera centers have singular values [18.9, 2.2, **0.08**],
  i.e. nearly co-planar. Small camera count + near-planar baseline ⇒ the reflected
  solution is an equally-valid local minimum of the unsupervised reprojection loss.
- **Why there is no fair fix.** A cheirality (points-in-front) check — the standard
  tool — **cannot** resolve this: our reconstruction is itself internally
  cheirality-valid (100% of visible points project in front of their cameras), so a
  cheirality check finds nothing to flag and has no signal to trigger a flip. The
  reflection is a *global* gauge relationship to GT (a global reflection of the whole
  scene preserves cheirality — every point stays in front of every camera), invisible
  without GT. Cheirality only breaks *local* pose ambiguities, not a global Necker
  reversal. For a near-planar
  degenerate configuration this ambiguity is fundamental to correspondence-only SfM;
  the paper resolved it by luck of initialization, not by an explicit mechanism (the
  ESFM code has no calibrated cheirality/reflection fix — its only "Chirality" option
  is projective-only; its ceres BA has no cheirality constraint). Any GT-based pick
  would be cheating.
- **Fair handling: exclude Drinking Fountain** as a documented reflection degeneracy
  (standard practice for degenerate scenes), applied symmetrically to all methods.
  With it excluded, ESFM best-of = **0.19°** ≈ paper **0.20°** (our baseline
  reproduces the paper), and U-ESFM best-of = 0.53° (close, still trailing ESFM).
- **NOT DONE:** a cheirality reflection-fix was investigated and rejected because it
  provably does not resolve this global-reflection case. Do not claim it in the paper.

## 6. One-line takeaways for the paper

- Our re-run ESFM baseline **reproduces the published ESFM** on the calibrated Olsson
  scenes (best-of, excluding the one reflection-degenerate scene).
- **U-ESFM does not beat plain ESFM in single-scene optimization on this (clean,
  low-outlier) dataset** — even with the report-faithful weighted loss, the best
  percentiles (30/70), and the sequential fallback. This is consistent with the
  report's own observation that the adaptive outlier loss helps on high-outlier
  scenes and degrades on low-outlier ones; Olsson is essentially outlier-free.
- Every cross-method change (budget, sequential, BA, alignment, scene exclusion) is
  applied **identically to all arms** and disclosed, so the comparison is fair.
