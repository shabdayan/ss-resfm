# TASK: Diagnose the "U-ESFM+TTT lands between ESFM and RESfM" Result

**Repo:** `deep-unsupervised-outlier-detection-esfm/`
**Parent spec:** SPEC_uesfm_combined.md — Ground rules (§1) apply throughout
(leakage guard, provenance, traceability, minimal diffs). Reuse the existing
runners (R6 TTT runner, R7 snapshot logging, R8 aggregation); build nothing
new unless a required knob is not yet conf-exposed.
**Context:** Multi-scene U-ESFM with full TTT currently evaluates between the
ESFM and RESfM baselines on the MegaDepth test protocol. Before this result is
accepted as the method's ceiling, three diagnostic questions (Part A) and two
structural analyses (Part B) must be answered. The output of this task decides
the paper's claim and venue — treat report accuracy as more important than
speed, and flag anomalies rather than smoothing over them.
**Priorities:** A1 → A3 → A2 → B1 → B2. (A1 and A3 are cheap and gate the
interpretation of everything else; B2 is the most consequential but also the
most expensive.)

---

## Part A — Diagnostics: is the current number real or a configuration artifact?

### A1. Reproduction-gate audit (calibration check)

Question: does our pipeline reproduce RESfM's published results, or is part of
the U-ESFM-to-RESfM gap a pipeline calibration difference?

1. Locate the existing reproduction-gate report (parent spec R13; produced
   during Stage 1). If it does not exist or covered fewer than 3 scenes, run
   it now: released RESfM `.pth` → their inference protocol (threshold 0.6
   removal → 1K reprojection-only fine-tune → robust BA) → our eval, on at
   least 5 test scenes spanning both Group 1 and Group 2.
2. Produce a per-scene table: reproduced vs published (rot, trans, Nr),
   absolute and relative gaps, plus the mean over the covered scenes.
3. Classify the outcome:
   - **CLEAN:** gaps small and unsystematic (sign varies by scene, magnitudes
     within run-to-run noise). Record achieved tolerance; the RESfM column is
     trustworthy.
   - **SYSTEMATIC:** reproduced consistently better or worse than published.
     This means part of the U-ESFM gap is measurement, not method. Diagnose:
     diff our robust-BA invocation/parameters, track handling, Group-2
     subsample selection, metric alignment (similarity alignment step), and
     eval-time track-matrix completeness against RESfM's described protocol.
     Report the suspected cause; do NOT silently "fix" numbers.
4. Same audit for the ESFM baseline column (published vs ours) — the interval
   "between ESFM and RESfM" has two endpoints and both must be calibrated.

### A2. TTT hyperparameter sweeps (inherited-settings check)

Question: is the TTT ceiling a property of the method or of the first
configuration? All current TTT hyperparameters were inherited from regimes
they were not tuned for. Run the following on **2 pilot scenes** chosen to
span contamination (one low-outlier ~12-17%, one high-outlier >30%; prefer
scenes where full per-variant results already exist for comparison).

Budget note: this is roughly a day of GPU compute total; run sweeps
sequentially, reuse the R6 runner, one conf per cell, results per parent-spec
traceability rules.

1. **Protocol-mode removal threshold** (calibrated for RESfM's densely
   supervised head, not our tail-trained head): sweep
   {0.4, 0.5, 0.6, 0.7, 0.8} in the protocol-mode variant
   (`ttt_loss = reproj_only` from OUR checkpoint). Also log, per threshold:
   fraction of observations removed and resulting outlier% (vs the scene's
   known contamination), and whether the viewing graph stays connected.
   Additionally record the score-distribution histogram of our outlier head
   on each pilot scene (20 bins) — if mass clusters near 0/1 with a sparse
   middle (expected from tail-only training), note the implied calibration
   difference vs a 0.6 cutoff.
2. **Warmup** (sized for from-scratch training; at TTT the head is already
   trained): full TTT with `warmup_epochs` in {0, 10 (current), 50}.
3. **TTT learning rate** (inherited from RESfM's reprojection-only fine-tune):
   full TTT with LR in {0.25x, 0.5x, 1x (current), 2x} of the current value.
   Keep the scheduler consistent: verify the TTT conf's scheduler milestones
   are scaled to the TTT budget (a from-scratch schedule with milestones at
   5000/7000/9000 inside a 1K run means LR never decays — if found, flag it
   as a bug-level finding and add scaled milestones as a swept variant).
4. **Budget** (1K inherited from RESfM, never verified for the combined
   loss): extend full TTT to 5K epochs on the 2 pilot scenes with R7
   snapshots at {0, 10, 100, 500, 1000, 2000, 5000}, using the best warmup/LR
   found above. Deliver the error-vs-step curve and identify where it
   flattens. Fairness rule (parent spec): if any extended budget is later
   adopted for results, it applies to ALL variants; for this pilot, also run
   the reproj-only control to 5K on the same 2 scenes so the "does the
   control saturate while TTT keeps improving?" question is answered.
5. Summarize: best-found configuration per scene, delta vs the current
   configuration, and an estimate of how much of the U-ESFM-to-RESfM gap the
   tuning headroom closes on these 2 scenes.

### A3. Internal-delta health check (does the mechanism work at all?)

Question: does full TTT clearly beat our OWN frozen and protocol-mode
variants? This is cheap (the numbers should already exist from the Table-2
runs) and gates interpretation:
- If YES (TTT > protocol-mode > / ≈ frozen, consistently): the mechanism
  delivers; only the absolute ceiling vs RESfM disappoints → Part B decides
  the paper.
- If NO (TTT ≈ or < protocol-mode): the adaptation mechanism itself is not
  delivering — a different and more concerning finding. In that case check,
  per scene, the R7 collapse/bootstrap diagnostics from the existing runs:
  mean predicted outlier fraction over TTT steps (drifting toward 1.0 =
  collapse; frozen at initialization = no learning signal reaching the head),
  percentile-threshold trajectories, and confident-set sizes
  (min_confident_samples gate constantly returning 0-loss = BCE silently
  inactive — check for this explicitly). Report which failure signature, if
  any, appears.

Deliverable for A3: a three-column per-scene table (frozen / protocol-mode /
full TTT) over all scenes already run, with per-scene deltas and a verdict
line: MECHANISM-HEALTHY / MECHANISM-WEAK / MECHANISM-BROKEN(+signature).

## Part B — Structure: where does U-ESFM+TTT win, if anywhere?

### B1. Per-scene analysis vs contamination (in-distribution)

Using the existing full-protocol MegaDepth results (all scenes run so far):

1. Produce a per-scene table sorted by outlier%: columns = outlier%, ESFM,
   RESfM (reproduced), U-ESFM frozen, U-ESFM protocol-mode, U-ESFM full TTT
   (rot and trans each), plus per-scene winner.
2. Compute the TTT-gain (full TTT minus protocol-mode, and full TTT minus
   RESfM) as a function of scene outlier%, and emit a scatter CSV for the
   gain-vs-contamination figure. Test the boundary hypothesis: do U-ESFM
   wins/losses vs RESfM split around the ~20-25% contamination boundary
   observed in the single-scene study? Report the split explicitly (scenes
   above boundary: won X of Y; below: won X of Y), for rotation and
   translation separately (the single-scene study found different hierarchies
   per task — preserve that distinction).
3. Flag any scene where U-ESFM+TTT beats reproduced RESfM outright; these are
   the candidate headline scenes.

### B2. Cross-dataset / distribution-shift runs (the consequential experiment)

Hypothesis to test: RESfM's supervised head is specialized to MegaDepth's
contamination statistics and must generalize zero-shot; our percentile
mechanism recalibrates per scene by construction. If U-ESFM+TTT overtakes
anywhere, it is under shift.

1. Precondition: consult CROSSDATASET_READINESS.md (readiness task). Run only
   PASS cells; list BLOCKED cells with unblock cost and wait for approval on
   anything requiring downloads/track regeneration.
2. Order of execution (by expected shift magnitude and cost):
   a. **1DSfM** — largest realistic shift, RESfM's own Table 2 setting, 10
      scenes. This is the priority dataset.
   b. **Strecha** (4 scenes) and **BlendedMVS** (4 scenes) — real
      (non-COLMAP) GT; low-outlier regime, so expect parity-at-best and use
      the per-dataset removal-threshold convention (conf-exposed, logged).
3. For each dataset, run the three-variant matrix (frozen / protocol-mode /
   full TTT) from the MegaDepth multi-scene checkpoint, PLUS the two baseline
   columns: reproduced RESfM (their `.pth`, their protocol) and ESFM if a
   checkpoint exists (else mark blocked). Use the best TTT configuration
   found in A2 AND the current default configuration (two TTT columns) so
   tuning and shift effects are separable.
4. Analysis per dataset: same per-scene structure as B1, plus the headline
   comparison — does U-ESFM full TTT beat reproduced RESfM on the dataset
   mean and/or on a majority of scenes? Compare each method's IN-distribution
   → OUT-of-distribution degradation (MegaDepth mean vs external mean):
   the crossover claim requires RESfM degrading more than U-ESFM+TTT.
5. Publish-or-not signal to report (not decide): dataset-level verdict per
   external dataset — CROSSOVER (U-ESFM+TTT ≥ RESfM), PARTIAL (wins on
   high-contamination subset), NONE.

## Deliverable

One report, `DIAGNOSIS_REPORT.md`, with sections A1/A2/A3/B1/B2, each ending
in its verdict line, plus a front-page summary table:

| Check | Verdict | Key number |
|---|---|---|
| A1 calibration | CLEAN / SYSTEMATIC(+cause) | mean reproduced-vs-published gap |
| A2 tuning headroom | headroom estimate on 2 scenes | best-config delta vs current |
| A3 mechanism | HEALTHY / WEAK / BROKEN(+signature) | mean TTT minus protocol-mode |
| B1 in-distribution structure | boundary CONFIRMED / ABSENT | wins above vs below ~20-25% |
| B2 shift | CROSSOVER / PARTIAL / NONE per dataset | U-ESFM+TTT vs RESfM per dataset mean |

All backing CSVs under the report's results directory, parent-spec
traceability fields in every file. No paper text, no venue recommendation in
the report — verdicts and numbers only.

## Out of scope
- Any retraining of multi-scene models (use existing checkpoints only).
- Architecture or loss changes; sweeps touch conf-exposed values only.
- Adopting an extended TTT budget for production runs (report the curve; the
  decision is made outside this task).
- Dataset downloads or track regeneration without approval (B2.1).
