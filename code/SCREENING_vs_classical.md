# SCREENING_vs_classical — Phase −1.B

Free screening (no new runs) of whether U-ESFM is competitive with the non-deep
methods, via the mandatory chaining rule. Published 1DSfM aggregates below are
**our averages over their ten Table 2 scenes** (RESfM prints no 1DSfM Mean row) —
not attributable to the paper as a Mean.

## Published 1DSfM aggregates (RESfM Table 2)

| method | trans mean / med | rot mean / med | coverage |
|---|---|---|---|
| GLOMAP | 0.576 / 0.112 | 1.27 / 0.55 | 0.98 |
| RESfM | 0.427 / 0.078 | 3.98 / 1.52 | 0.72 |
| Theia | 1.337 / 0.579 | 5.80 / 3.32 | 0.98 |
| ESFM | 1.119 / 0.542 | 11.17 / 7.09 | 0.64 |

## Chaining (rotation; translation cannot be chained — scale differs)

1. **U-ESFM vs our reproduced RESfM** (same pipeline): U-ESFM 2.86° vs reproRESfM
   4.16° on healthy-8 rotation → U-ESFM better by 1.45×.
2. **Calibration factor** (reproRESfM vs published RESfM): healthy-8 4.16 vs 4.79
   → ≈ 0.87× (near-calibrated on healthy-8; full-10 is 2.57× but that is the two
   pathological scenes, see CALIBRATION_ALERT / Gate R0).
3. **Read across (healthy-8, factor ≈ 1):** U-ESFM ≈ 2.86° estimated on the
   published scale would sit **between GLOMAP (1.27°) and Theia (5.80°)** on
   rotation, and ahead of published ESFM (11.17°) — but **behind GLOMAP**, a
   training-free classical method.

## Translation (the primary metric) — cannot chain, but the direction is clear

- Within our pipeline U-ESFM **loses translation to reproduced RESfM** (see
  TRANSLATION_CHECK / Gate T1: 3/10 scene wins; healthy-8 4.60/3.50 vs 3.36/1.99).
- On the published scale RESfM (0.427) already **beats** GLOMAP (0.576) on
  translation mean. Since U-ESFM trails reproduced RESfM on translation, it is
  very unlikely to beat the best classical method (GLOMAP/RESfM) on translation.

## Verdicts

- **Translation (primary): `NOT-COMPETITIVE`** with the best classical/deep
  method on the primary metric, on current evidence. U-ESFM does not beat
  reproduced RESfM on 1DSfM translation, and RESfM already beats GLOMAP there.
- **Rotation (secondary): `COMPETITIVE-UNCONFIRMED`** — U-ESFM is plausibly
  between GLOMAP and Theia, ahead of ESFM/Theia but behind GLOMAP. A "competitive"
  rotation read is **not trustworthy** until GLOMAP is run on our tracks (Phase
  1A) — coverage (GLOMAP 0.98 vs RESfM 0.72) and the intersection comparison
  cannot be settled from published aggregates.

## Caveats (mandatory)

- Coverage is unsettled here by construction; Phase 1A intersection comparison is
  required before any competitiveness claim on either metric.
- The rotation read leans on the healthy-8 calibration factor; on full-10 the
  factor is 2.57× and the chain is invalid (pathological scenes).
