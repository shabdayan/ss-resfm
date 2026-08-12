# TRANSLATION_CHECK — Phase −1.C (Gate T1)

Metric priority (Fadi): **translation → rotation → Nr**. All comparisons are
U-ESFM vs **reproduced** RESfM (`released, our tracks`) — same pipeline, same
gt_bundle scale, so valid. Published translation (0.427) is a different scale and
is NOT compared. U-ESFM = adaptive-loss method; protocol tag shown. Two-stage MAD
and ESFM are baselines. 5-seed per-scene means; "med" = median over scenes.

## 1DSfM — translation (primary) and rotation (secondary)

| arm (tag) | T mean10 | T med10 | T mean8 | T med8 | R mean10 | R mean8 |
|---|---|---|---|---|---|---|
| reproRESfM | 10.71 | **3.61** | **3.36** | **1.99** | 10.21 | 4.16 |
| U-ESFM (protocol-mode) | **9.35** | 4.47 | 4.60 | 3.50 | **5.73** | **2.86** |
| U-ESFM (ttt) | 11.06 | 5.38 | 6.01 | 4.71 | 6.48 | 3.58 |
| two-stage MAD | 12.62 | 4.92 | 6.15 | 3.77 | 7.91 | 4.87 |
| ESFM-deep | 15.34 | 10.14 | 9.68 | 7.34 | 8.37 | 6.57 |
| ESFM-van | 17.89 | 13.99 | 12.07 | 8.89 | 14.26 | 11.06 |

(mean10/med10 = over all ten scenes; mean8/med8 = healthy-8, excl. Ellis+Tower.)

## Per-scene translation: U-ESFM (protocol-mode) vs reproRESfM

Wins for U-ESFM: **3/10** (Alamo, NYC_Library, Tower_of_London).

| scene | U-ESFM T | RESfM T | winner |
|---|---|---|---|
| Alamo | 1.94 | 2.20 | U-ESFM |
| Ellis_Island | 17.95 | 15.55 | RESfM |
| Madrid_Metropolis | 6.85 | 5.02 | RESfM |
| Montreal_Notre_Dame | 5.17 | 0.91 | RESfM |
| Notre_Dame | 2.29 | 0.56 | RESfM |
| NYC_Library | 3.76 | 5.89 | U-ESFM |
| Piazza_del_Popolo | 3.24 | 1.76 | RESfM |
| Tower_of_London | 38.78 | 64.69 | U-ESFM |
| Vienna_Cathedral | 3.02 | 1.78 | RESfM |
| Yorkminster | 10.54 | 8.79 | RESfM |

## Reading

- **Translation (primary):** U-ESFM's only full-10 mean "win" (9.35 vs 10.71) is
  carried entirely by Tower_of_London (38.78 vs 64.69), a pathological scene where
  both fail. On the **median** (3.61 vs 4.47), on **healthy-8 mean** (3.36 vs
  4.60), on **healthy-8 median** (1.99 vs 3.50), and on **per-scene wins (3/10)**,
  reproduced RESfM beats U-ESFM. **U-ESFM does not win 1DSfM translation.**
- **Rotation (secondary):** U-ESFM wins clearly — full-10 5.73 vs 10.21, healthy-8
  2.86 vs 4.16 (and vs published 4.79). The crossover is real on rotation.
- **TTT** is worse than protocol-mode on both metrics → drop from the headline.
- U-ESFM beats **two-stage MAD** on both metrics (mechanism story intact; that is
  Phase 1B, not the RESfM comparison).

## 5.73° vs 6.93° reconciliation

- **5.73°** = the paper_style table value: U-ESFM (adaptive), checkpoint
  `uesfm_27scenes_adaptive` Ep16500, MAD test-time prune, **protocol-mode**,
  per-dataset ft-lr 5e-3, fixed 20/80 percentiles (pre-A2.5), 5-seed per-scene
  mean of `Rs_ba_final_mean`. Reproduced exactly (seed-20 alone 5.35; per-seed
  means 4.16–6.69; scene-median aggregate 3.55; 1DSfM-8 2.86).
- **6.93°** is **not reproducible** from these runs — no aggregation of the
  deep-adaptive 1DSfM cells yields it. The only literal 6.93 in committed data is
  (a) ESFM-deep on BlendedMVS scene2, and (b) MegaDepth scene 0060 rotation in
  `B1_per_scene_vs_contamination.csv` — different arm/dataset. Treat the B2
  "protocol-mode 6.93°" as a stale/mis-transcribed interim figure superseded by
  5.73°; they are **not the same run**.

## Gate T1 — verdict: **CROSSOVER-ROTATION-ONLY**

The 1DSfM crossover holds on rotation but **not** on translation. Under the
translation-first priority, the headline OOD claim as currently written (rotation-
led) is on the secondary metric and must be restated. Maps to publication
**Scenario B** (narrow), not Scenario A.

**Provisional per Gate R0:** R0 = TRACK-MISMATCH (not CALIBRATION-OK), so this
verdict is measured against a baseline whose full-10 aggregate runs hot on two
pathological scenes. On healthy-8 the picture is unchanged and stronger for the
verdict (reproRESfM translation 3.36/1.99 vs U-ESFM 4.60/3.50), so
CROSSOVER-ROTATION-ONLY does not depend on the pathological scenes.

## Pending

- **MegaDepth** per-scene translation for all arms (in-distribution reference) —
  not yet aggregated here; 1DSfM is the OOD gate and is decisive on its own.
