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

## 5. Drinking Fountain — restart sensitivity, fixed by best-of-seeds (NOT a degeneracy, NOT excluded)

- **Symptom (as first observed).** With **mean-over-seeds** aggregation, both arms
  report ~15.6° post-BA rotation on Drinking Fountain (paper: 0.007°). This first
  looked like a global-reflection degeneracy.
- **Corrected diagnosis (verified per-seed).** It is **not** a degeneracy — it is
  **restart sensitivity**. Of 3 seeds, **seed 2 solves the scene for BOTH arms**
  (post-BA rot **0.005°** ≈ paper 0.007°, reproj **0.31px**); seeds 0 and 1 land in a
  bad local minimum (rot 23°, reproj **7.6px**). The good and bad basins are cleanly
  separated by **reprojection error alone** — a 24× gap (0.31 vs 7.6 px) — with **no
  ground truth**.

  | arm | seed0 rot/reproj | seed1 | seed2 |
  |-----|-----|-----|-----|
  | esfm  | 23.4° / 7.65px | 23.4° / 7.59px | **0.005° / 0.31px** |
  | uesfm | 23.4° / 7.67px | 23.5° / 7.59px | **0.005° / 0.31px** |

- **Fair fix: best-of-seeds selected by reprojection error** (GT-free), applied
  **uniformly to every scene and both arms**. This recovers Drinking Fountain for
  both methods and needs no scene exclusion. It is standard multi-restart practice
  and, because the selection metric is reprojection (not GT), it is not cherry-picking.
  Recommendation for the paper: report best-of-seeds (and/or median-of-seeds), not
  mean-of-seeds, so a single unlucky restart on one scene doesn't dominate the mean.
- **What did NOT work (investigated, rejected — do not claim in the paper):**
  - A **cheirality reflection-fix**: our reconstructions are internally
    cheirality-valid (points in front), so a cheirality check has no signal to flag.
  - A **BA-from-both-mirror-hypotheses** fix (run BA from the reconstruction *and*
    from a global-reflection init, keep lower reprojection): the mirror init always
    converges to a *worse* reprojection (37–315px) and a different bad basin (~19–21°),
    so it never rescues the failed seeds. Reported here for completeness; not used.
  - Both are consistent with the geometry: a proper-rotation reflection twin that
    reprojects to the observations does not exist, so neither trick can manufacture
    the correct solution — but a good *restart* finds it directly.

## 6. One-line takeaways for the paper

- Our re-run ESFM baseline **reproduces the published ESFM** on the calibrated Olsson
  scenes under best-of-seeds selection (Drinking Fountain included — see §5; it is a
  restart-sensitivity scene that a good seed solves, not a degeneracy to exclude).
- **U-ESFM does not beat plain ESFM in single-scene optimization on this (clean,
  low-outlier) dataset** — even with the report-faithful weighted loss, the best
  percentiles (30/70), and the sequential fallback. This is consistent with the
  report's own observation that the adaptive outlier loss helps on high-outlier
  scenes and degrades on low-outlier ones; Olsson is essentially outlier-free.
- Every cross-method change (budget, sequential, BA, alignment, scene exclusion) is
  applied **identically to all arms** and disclosed, so the comparison is fair.
