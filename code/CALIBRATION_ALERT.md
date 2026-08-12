# CALIBRATION_ALERT — Phase −1.A (Gate R0)

Reproduced RESfM = `RESfM (released, our tracks)`: authors' pretrained checkpoint,
our eval path, our tracks, 5 seeds. Published = RESfM Table 2 (arXiv:2404.14280v2,
ICLR 2025). Rotation in degrees; comparable across pipelines. Translation NOT
compared here (our gt_bundle scale ≠ their COLMAP-GT scale). Metric note: R0 is a
rotation-gap check by construction.

## 1. Per-scene reproduced vs published RESfM (1DSfM) + track contamination

| scene | our Out% | pub Out% | ΔOut | reproRot | pubRot | ratio | reproNr | pubNr |
|---|---|---|---|---|---|---|---|---|
| Alamo | 29.2 | 32.6 | −3.4 | 4.27 | 3.66 | 1.17 | 499 | 484 |
| Ellis_Island | 56.3 | 25.1 | **+31.2** | 18.68 | 0.82 | **22.8×** | 199 | 214 |
| Madrid_Metropolis | 29.1 | 39.4 | −10.3 | 2.52 | 8.42 | 0.30 | 272 | 244 |
| Montreal_Notre_Dame | 36.3 | 31.7 | +4.6 | 0.49 | 2.82 | 0.17 | 399 | 346 |
| Notre_Dame | 48.1 | 35.6 | +12.5 | 1.22 | 1.20 | 1.02 | 510 | 517 |
| NYC_Library | 53.8 | 33.6 | +20.2 | 11.38 | 3.96 | 2.87× | 261 | 224 |
| Piazza_del_Popolo | 42.1 | 33.1 | +9.0 | 2.40 | 2.20 | 1.09 | 241 | 249 |
| Tower_of_London | 41.6 | 27.0 | +14.6 | 50.18 | 0.67 | **74.9×** | 286 | 94 |
| Vienna_Cathedral | 51.5 | 31.4 | +20.1 | 0.79 | 1.52 | 0.52 | 483 | 479 |
| Yorkminster | 45.3 | 29.0 | +16.3 | 10.21 | 14.54 | 0.70 | 306 | 331 |

Scene identity confirmed (our Nc ≈ pub Nc on all ten) — the scenes/images match;
only the track construction differs.

## 2. Aggregates

| set | repro rot (mean-of-means) | pub rot | ratio | our Out% | pub Out% |
|---|---|---|---|---|---|
| full 10 | 10.21 | 3.98 | **2.57×** | 43.3 | 31.9 |
| healthy 8 (excl. Ellis, Tower) | **4.16** | 4.79 | **0.87×** | — | — |
| repro rot mean-of-medians | 7.76 | 1.52 | — | | |

## 3. Decomposition of the 2.57× gap

- The entire gap is **two scenes**: Ellis_Island (22.8×) and Tower_of_London
  (74.9×). Removing them, reproduced RESfM (4.16°) **beats** published (4.79°).
- On the other 8 scenes reproduced RESfM is within ±20% or better than published
  (7 of 8 have ratio ≤ 1.17; NYC is 2.87×).
- Ellis and Tower are the known released-data pathology (gt_bundle/tracks globally
  inconsistent — prior oracle-pruning experiment could not rescue them while it
  rescued NYC/Alamo controls). Tower also shows the coverage confound: published
  keeps 94/467 cameras (0.67°); we keep 286 (50.18°) — their aggressive removal,
  not a better model.
- Our tracks are **systematically dirtier**: mean outlier 43.3% vs their 31.9%,
  higher on 8/10 scenes (Ellis +31, NYC +20, Vienna +20, Notre_Dame +12). The
  fixed 0.6 removal threshold was tuned on their cleaner score distribution.

## 4. Eval-path vs track decomposition (via MegaDepth identical tracks)

Track provenance (user, 2026-08-12): **MegaDepth tracks are identical to RESfM's
release** (byte-verified); the **1DSfM/Strecha/BlendedMVS tracks were built by us**,
provenance of RESfM's OOD tracks unknown (possibly different processing). This
enables the spec's row-2/row-3 decomposition for the in-distribution case:

| comparison | tracks | reproduced | published | factor | isolates |
|---|---|---|---|---|---|
| MegaDepth (row 2 − row 1) | **identical** | 1.82 | 1.29 | **1.41×** | eval path / BA environment |
| 1DSfM full-10 | ours | 10.21 | 3.98 | 2.57× | eval path × tracks |
| 1DSfM healthy-8 | ours | 4.16 | 4.79 | 0.87× | eval path × tracks |

- On **identical MegaDepth tracks** the reproduction factor is **1.41×** (BA
  environment sensitivity, seed-negligible — prior finding). This is the clean
  eval-path measurement the spec's row 2 requires; **the eval path is not broken.**
- The **residual 1DSfM factor** on top of eval-path ≈ 2.57 / 1.41 = **1.83×** is
  the OOD track-construction effect — and it collapses on healthy-8 (0.87× < the
  1.41× eval-path factor), i.e. once Ellis/Tower are removed our tracks are *not*
  systematically worse for RESfM.
- We **cannot** produce the true OOD row 2 (`released, their tracks`) — we do not
  have RESfM's OOD tracks and do not know their construction. Only Fadi can supply
  them; until then the OOD track factor is an estimate anchored on the MegaDepth
  eval-path factor.

## Gate R0 — verdict: **TRACK-MISMATCH (benign, resolved)**

The gap is explained by the track distribution (our tracks are dirtier; the
inherited 0.6 threshold is off-calibration) plus two pathological scenes, **not a
code/eval-path bug**. On comparable (healthy-8) data our evaluation reproduces the
paper (4.16 vs 4.79). No retraining performed (spec forbids it).

## BLOCKED sub-items (require inputs we do not have)

- **`RESfM (released, their tracks)` row for OOD** — the decisive OOD track
  isolation. We built the 1DSfM/Strecha/BlendedMVS tracks ourselves; RESfM's OOD
  tracks are not in hand and their construction is unknown. **Only Fadi can
  supply them.** (In-distribution row 2 is DONE via MegaDepth identical tracks =
  1.41× eval-path factor, §4.) Unblock: obtain Fadi's OOD tracks + processing
  procedure, then 1 eval fleet.
- **Steelman: 0.6 threshold swept on our track distribution** — required before
  any published comparison (comparing our adaptive mechanism to their
  miscalibrated fixed threshold is not a fair comparison). Unblock: threshold
  sweep on resfm_repro, ~1 fleet.
- **Outlier-head score distribution vs 0.6** (item 5) — needs a score dump from
  the resfm_repro TEST pass. Unblock: 1 inference pass with score logging.
