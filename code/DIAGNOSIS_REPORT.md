# DIAGNOSIS REPORT — "U-ESFM+TTT lands between ESFM and RESfM"

Per `claude specs/TASK_diagnosis_gap_to_resfm.md` (updated version incl. A2.5).
Assembled 2026-08-02. All numbers seed 20, 36 MegaDepth test scenes unless
noted; per-scene MEANS (project reporting convention). Backing CSVs:
`results/diagnosis/` (git hash + seed columns inside each file).

## Front-page verdict table

| Check | Verdict | Key number |
|---|---|---|
| A1 calibration | SYSTEMATIC, both endpoints, opposite signs | RESfM repro +0.60° vs published (BA env); enhanced-"ESFM" −1.71° vs published (wrong architecture — see esfm-vanilla task) |
| A2 tuning headroom | REAL, scene-dependent, opposing directions | 0104: 0.21→0.06° (thr 0.8) / TTT 0.17→0.07° (pct 10/90); 0099: defaults already ≈optimal (best gain 0.68→0.59°) |
| A2.5.0 unit bug | CONFIRMED in code default; NEUTRALIZED in all existing results | separation floor 0.5 vs real thresholds 0.007–0.15; all result confs use 0.0 (fe55bf0); BCE empirically active (confident_frac=0.40 every snapshot) |
| A3 mechanism | MECHANISM-WEAK (no failure signature) | TTT−protocol = +1.34° mean; TTT wins 19/36 scenes; ≫ frozen (5.74→4.82) |
| B1 in-distribution structure | boundary ABSENT | TTT beats RESfM(repro) 18/36 — 10/18 above 22.5% contamination, 8/18 below; gain-contamination corr +0.14 |
| B2 shift | 1DSfM: CROSSOVER (protocol-mode, NOT TTT); Strecha: NONE; BlendedMVS: NONE for learned heads (statistical mode wins) | 1DSfM: U-ESFM protocol 6.93° vs RESfM 10.59°; TTT worst row on all 3 datasets |

## A1 — Calibration audit

- **RESfM endpoint** (36 scenes × 5 seeds, `resfm_repro_env38*`): reproduced
  1.89/0.209/304 vs published 1.29/0.169/299. SYSTEMATIC (consistently worse).
  Cause: robust-BA environment sensitivity (pycolmap/ceres build); data ruled
  out — local MegaDepth npz are byte-identical to RESfM's released datasets.zip
  (incl. Group-2 300-subsamples); seeds negligible.
- **ESFM endpoint** (`A1_esfm_endpoint.csv`): our then-"ESFM" run measured
  5.05/0.663/287 vs published 6.77/0.780/240 — SYSTEMATIC, better, on 24/36
  scenes. Cause: mislabeled architecture (DeepSetOfSetOutliersNet 1×3 with
  residual/LN/dropout-enhanced blocks + Kaiming) — i.e. "Deep-ESFM plain loss",
  not vanilla ESFM. Handled by TASK_esfm_vanilla_baseline: run relabeled
  `deep_esfm_plain_loss`; true vanilla (SetOfSetNet, 663,562 params ≈ the
  published 0.66M) trained to Ep19999; its eval + A1-addendum follow.
- **Calibrated interval (single ruler, current)**: enhanced-ESFM 5.05 →
  U-ESFM 3.47 → RESfM(repro) 1.89 (Rot); vanilla-ESFM endpoint pending.

**Verdict: SYSTEMATIC (+0.60° / −1.71°, causes identified). Use reproduced
rows only for deltas; never published numbers.**

## A2 — TTT hyperparameter sweeps (pilots: 0104 @16.2%, 0099 @47.4%)

Base: best arm (DSOS 1×3 lr1e-4, ckpt Ep19000). References: protocol
0.21/0.66 (Rot, 0104/0099), default-config TTT 0.17/0.68.

1. **Removal threshold** (protocol-mode): 0104 improves monotonically to
   thr 0.8 → 0.06° (3.5× better than 0.6); 0099 is best at 0.6 (0.4 → 2.87°).
   Score histograms (`A2_score_hist_*.csv`): 0099 bimodal (threshold-robust);
   0104 soft-mid-mass — every fixed threshold over-prunes (0.6 removes 39% vs
   16% true contamination). The 0.6 convention is calibrated for RESfM's
   supervised head, not ours.
2. **Warmup** {0,10,50}: minor; 50 slightly better on 0104 (0.14), worse-mixed
   on 0099.
3. **TTT lr** {0.25×,0.5×,1×,2×}: 0104 wants 0.25× (0.09); 0099 wants 2×
   (0.59). **Scheduler bug (bug-level finding): fine-tune confs carry
   scheduler_milestone=[10000] inside 1001-epoch runs — LR never decays in ANY
   fine-tune, ours or RESfM's protocol.** Scaled milestones (500/700/900) did
   not help (0104: 0.17 =default; 0099: 2.03 — worse); flagged, not adopted.
4. **Budget** (5K, best warmup/lr, + reproj control): curves flatten at
   ~500–1000 steps; TTT does NOT keep improving where control saturates.
   0099 degrades badly past ~1–2K (post-BA 2.16 vs 0.68 at 1K). 1K vindicated.
5. **A2.5.0 unit check**: CONFIRMED-BUT-NEUTRALIZED (see verdict table).
   The single-scene report era (min_threshold_separation=0.5 in pixel-scale
   confs) plausibly ran with silently-inactive BCE; every multi-scene/TTT
   result in this repo used 0.0 and logged active confident sets. No result
   invalidated; no rerun required.
   **A2.5.1 percentiles**: 0104: (10,90) → 0.07°; contamination-linked
   (new conf-gated option, commit 21e3ce6) → 0.08° with auto-picked pct 92 —
   near-optimal without tuning. 0099: (30,70) → 0.62°; contamination-linked
   mid-pack (0.73; MAD underestimates at 47% contamination — known breakdown).
   **A2.5.2 gate**: 50 hurts 0104 (0.24), helps 0099 (0.59). Gate confirmed
   active (never returned empty confident sets in any logged run).

**Headroom summary: on the low-contamination pilot the gap to RESfM(repro)
inverts under tuning (best cell 0.06–0.07 vs RESfM 0.27); on the
high-contamination pilot U-ESFM already beats RESfM and tuning adds ~13%.
No single fixed configuration captures both — the consistent winner-direction
is contamination-adaptive settings, which is the method's own philosophy
applied to its inference knobs.**

## A3 — Mechanism health (`A3_mechanism_table_bestarm.csv`)

Frozen 5.74 → protocol 3.47 → full TTT 4.82 (Rot means, best arm, 36 scenes).
TTT ≫ frozen (26/36 scenes); TTT vs protocol: wins 19/36 but mean +1.34°
worse — losses are heavy-tailed, wins small. No failure signature: no collapse
(pred fraction 0.36–0.50 at final step, never →1), head never frozen (1/36
scenes with |Δfrac|<0.02), BCE never inactive (confident_frac ≥ 0.40 on every
logged snapshot).

**Verdict: MECHANISM-WEAK — adaptation delivers vs frozen and on half the
scenes vs protocol, but harms a tail of scenes; not BROKEN.**

## B1 — In-distribution structure (`B1_per_scene_vs_contamination.csv`)

- TTT beats RESfM(repro) on 18/36 scenes (protocol: 17/36) — dead-even
  per-scene split with the supervised method; the mean gap is magnitude
  asymmetry on losses, not win-rate.
- Contamination boundary (single-scene study's 20–25%): ABSENT — wins split
  10/18 above / 8/18 below 22.5%; TTT-gain vs contamination corr +0.14.
- 18 headline scenes where TTT beats RESfM outright, e.g. 0046 (2.49 vs
  12.82), 0099 (0.68 vs 3.39), 0240 (0.86 vs 3.40).

**Verdict: boundary ABSENT; per-scene parity with supervised RESfM; the gap
is a tail problem.**

## B2 — Distribution shift (full-budget matrix, results/crossdataset*)

Rot/Trans means; U-ESFM = SOS 1×3 arm; RESfM = released .pth, our tracks.

| Row | 1DSfM (10) | Strecha (4) | BlendedMVS (4) |
|---|---|---|---|
| ESFM (enhanced baseline) | 10.12 / 15.87 | 14.44 / 2.95 | 10.63 / 0.16 |
| RESfM (repro) | 10.59 / 11.12 | **2.04 / 0.20** | 31.91 / 0.33 |
| U-ESFM MAD-shallow | 7.76 / 11.89 | 12.97 / 2.55 | **6.56 / 0.11** |
| U-ESFM frozen | 14.77 / 14.96 | 16.08 / 2.81 | 21.10 / 0.32 |
| U-ESFM protocol | **6.93 / 9.22** | 5.86 / 1.37 | 31.86 / 0.41 |
| U-ESFM full TTT | 16.67 / 20.00 | 9.53 / 0.76 | 34.17 / 0.28 |

- Degradation in→out (Rot mean, MegaDepth→dataset): RESfM 1.89→10.59 (×5.6)
  on 1DSfM; U-ESFM protocol 4.31→6.93 (×1.6). The crossover condition (RESfM
  degrades more) HOLDS on 1DSfM — for protocol-mode, not TTT.
- **Per-dataset verdicts: 1DSfM CROSSOVER (protocol-mode beats RESfM by 35%
  Rot; caveat: registers fewer cameras, 288 vs 346); Strecha NONE (RESfM
  best; U-ESFM protocol 2nd); BlendedMVS NONE for every learned head incl.
  RESfM's (31.9°) — only statistical treatments survive (MAD 6.56°).**
- Full TTT is the worst U-ESFM row on all three datasets: the "TTT gains are
  larger under shift" hypothesis is REFUTED in its current configuration.
  (A2's best-config TTT columns under shift were not run — headroom exists
  per A2 but the default-config verdict is unambiguous.)

## Failure-tail note (cross-cutting)

The single recurring failure mode across A3/B1/B2 is a small set of scenes
where comb-TTT degrades a good protocol solution. Collapse diagnostics rule
out head pathology; A2 shows per-scene-adaptive inference settings fix the
low-contamination side of the tail. The actionable candidates are
contamination-linked percentiles (implemented, conf-gated) and
threshold-from-score-distribution removal.

## A1-addendum (TASK_esfm_vanilla_baseline Step 5) — vanilla ESFM calibration

`esfm_vanilla` (SetOfSetNet, 663,562 params, no outlier handling; best ckpt
Ep8000/20k) on all 36 scenes, matched environment:
**Rot 7.32 / Trans 0.809 / Nr 276** vs published ESFM 6.77 / 0.780 / 240.

Classification: **SYSTEMATIC, +0.55° — consistent in sign and magnitude with
the RESfM endpoint's +0.60°.** Both endpoints now show the same uniform
BA-environment offset; the previous −1.71° anomaly is fully explained by the
mislabeled architecture (that run is `deep_esfm_plain_loss`, R11 ablation
only). The pipeline is uniformly calibrated.

**Corrected calibrated interval (single ruler, Rot / Trans means):**
vanilla ESFM 7.32 / 0.809 → **U-ESFM 3.47 / 0.358** → RESfM(repro) 1.89 / 0.209.
U-ESFM removes 53% of vanilla ESFM's rotation error (72% of the ESFM→RESfM
interval measured from the ESFM end: 3.85° of 5.43°) without labels.
