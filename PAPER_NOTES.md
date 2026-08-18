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

## 5. Seed protocol — single random run, matching ESFM (Drinking Fountain reported as-is)

- **ESFM's protocol (verified in paper + code).** Single-scene optimization is a
  **single unseeded random-init run**, one number per scene (paper Sec 3.4; the
  released `Optimization_Euc.conf` sets **no** `random_seed`; `single_scene_optimization.py`
  makes one `train.train` call with no restart/selection loop). There is no seed to
  copy and no best-of-seeds.
- **Our protocol (matched).** We report a **single random run at a fixed seed (seed 0)**
  for every scene and every arm — `aggregate_single_scene.py --seed 0`. Both arms use
  the *same* injected `random_seed`, so the two methods share an identical
  initialization and differ only by method. No mean-over-seeds, no best-of-seeds: a
  strict like-for-like against ESFM. (We still *run* seeds 0/1/2 and keep them all in
  `summary.csv` for provenance, but only the reported seed enters the tables.)
- **Drinking Fountain is reported as-is, NOT excluded and NOT rescued.** It is a
  restart-sensitive scene: at seed 0 all three arms share the same init and all land
  in the same bad basin (~23.4° post-BA, reproj 7.6px); at seed 2 the same code solves
  it (0.005°, reproj 0.31px). ESFM's own unseeded run carries the identical lottery
  risk — its published 0.007° is simply a run that landed good. Since our fixed seed
  is applied identically to all arms, no method is advantaged, and the comparison stays
  fair. Report the across-scene **median** alongside the mean so one unlucky scene does
  not dominate (at seed 0: post-BA rotation median esfm 0.19° / esfm_rc 0.24° / uesfm 1.73°;
  mean 12.5° / 11.8° / 18.7°).
- **What we did NOT do (investigated, rejected — do not claim in the paper):**
  - **best-of-seeds** — would beat ESFM's literal single-run protocol; ruled out for
    fairness (we match ESFM's procedure, not just its reported outcome).
  - A **cheirality reflection-fix** — our reconstructions are internally
    cheirality-valid (points in front), so a cheirality check has no signal to flag.
  - **BA-from-both-mirror-hypotheses** (BA from the reconstruction *and* from a
    global-reflection init, keep lower reprojection): the mirror init always converges
    to a *worse* reprojection (37–315px) and a different bad basin (~19–21°), so it
    never rescues the failed seeds. Consistent with the geometry — a proper-rotation
    reflection twin that reprojects to the observations does not exist.

## 6. One-line takeaways for the paper

- Our re-run ESFM baseline **reproduces the published ESFM** on the calibrated Olsson
  scenes (single random run, seed 0), except on restart-sensitive scenes where the
  fixed seed lands in a bad basin for *all* arms (e.g. Drinking Fountain — see §5;
  reported as-is, since ESFM's unseeded run carries the same lottery risk).
- **U-ESFM does not beat plain ESFM in single-scene optimization on this (clean,
  low-outlier) dataset** — even with the report-faithful weighted loss, the best
  percentiles (30/70), and the sequential fallback. This is consistent with the
  report's own observation that the adaptive outlier loss helps on high-outlier
  scenes and degrades on low-outlier ones; Olsson is essentially outlier-free.
- Every cross-method change (budget, sequential, BA, alignment, scene exclusion) is
  applied **identically to all arms** and disclosed, so the comparison is fair.
