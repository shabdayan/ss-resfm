# U-ESFM vs RESfM — consolidated results (as of Aug 28 2026)

All numbers: **post-BA translation error (°), mean / median over scenes**, seed 20 unless noted.
Contamination axis: Strecha ~2% · BMVS ~3% · MegaDepth ~26% · 1DSfM 43% · 1DSfM-hard 61%.
OOD tracks are our own rebuilds (NOT comparable to the RESfM paper's OOD numbers; Fadi's author tracks expected ~Sep 4).

## A. PRIMARY — config-consistent twin pair (finelr = authors' 11-milestone schedule)

U-ESFM-finelr and RESfM-scratch-finelr share EVERYTHING (authors' schedule, margin 1e-4, our_repro
selection, tracks, env, seed 20) except the loss (self-sup vs COLMAP-sup) and base lr (1e-4 vs 1e-3,
each method's preferred; RESfM@1e-4 symmetry probe training).

| Model | MegaDepth (36) | 1DSfM (10) | 1DSfM-hard (5) | Strecha (4) | BMVS (4) |
|---|---|---|---|---|---|
| U-ESFM madweight | 0.42 / 0.15 | 12.39 / 4.37 | **12.21 / 8.83** | 2.50 / 0.97 | 0.15 / **0.004** |
| U-ESFM weight | 0.42 / 0.18 | 12.79 / 11.01 | 21.94 | 2.00 / 0.026 | **0.10** / 0.036 |
| U-ESFM weight+TTT | **0.316** / 0.19 | 16.13 / 7.16 | 26.44 | 1.98 / **0.005** | 0.14 / 0.011 |
| RESfM-finelr | 0.344 / **0.07** | **11.03 / 2.93** | 16.22 / 13.08 | **0.006 / 0.005** | 0.38 / 0.35 |

Seed-20 readings: U-ESFM wins MegaDepth mean (0.316 vs 0.344), 1DSfM-hard (12.2 vs 16.2), BMVS
(0.10–0.15 vs 0.38); ties Strecha on median (0.005 — mean gap is one bad scene); loses plain 1DSfM
(11.0/2.9 RESfM vs 12.4/4.4). Caveat: 1DSfM has the largest seed variance of any dataset.

## A2. Original 5-milestone schedule (port-drift recipe — superseded as primary)

| Model | MegaDepth (36) | 1DSfM (10) | 1DSfM-hard (5) | Strecha (4) | BMVS |
|---|---|---|---|---|---|
| **U-ESFM madweight** | 0.49 / 0.19 | **5.50 / 2.32** | **12.21 / 8.83** † | 2.81 / — (n4) | 0.21 / 0.18 (n4) |
| U-ESFM weight | 0.46 / — | 15.88 | 21.94 † | 1.95 | 0.11 (n4) |
| U-ESFM weight+TTT | 0.53 / — | 15.64 | 26.44 † | 2.02 | 0.13 (n4) |
| RESfM-scratch (protocol twin) | 0.33 / — | 11.39 / 9.88 | 16.22 / 13.08 † | **0.006** | 0.39 (n4) |
| RESfM authors-code, our tracks | 0.39 / 0.14 | 13.03 / 5.60 (n8‡) | 20.84 (n2‡) | **0.005** | 2.95 / 0.22 (n9) |
| RESfM released ckpt | **0.209–0.217** | 10.71 / 10.21 | — | 0.20 | 0.33 (n4) |
| RESfM paper (their tracks) | 0.169 | n/c | — | n/c | n/c |

† 1DSfM-hard rows use finelr checkpoints (the only ckpts evaluated there).
‡ authors-code OOD partially failed (2 of 10 1DSfM, 3 of 5 hard scenes crashed in their pipeline) — mean not comparable, per-scene comparison only.

**U-ESFM finelr (authors' 11-milestone schedule), seed 20:** MD 0.316 (weight+TTT) · 1DSfM madweight ~14.5. All other U-ESFM rows above = original 5-milestone schedule, seed 20.

## B. Multi-dataset trained (shallow, v1 = 39 scenes MD+ETH3D; v2 = 52 scenes +VGG+T&T)

NOTE for combined reading with Table A: multids arms trained on the 5-milestone recipe (not finelr);
U-ESFM-52 best-val Ep8500 (early) vs RESfM-52 Ep19500 (late) = checkpoint-regime confound;
BMVS n=9 here vs n=4 in Table A (compare medians, not means).

| Model | MegaDepth (36) | 1DSfM (10) | 1DSfM-hard (5) | Strecha (4) | BMVS (9) |
|---|---|---|---|---|---|
| **U-ESFM-52 madweight** | — | **6.43 / 3.53** | 20.57 | 2.88 | **0.22 / 0.07** |
| U-ESFM-52 weight | 0.66 / 0.13 | 14.10 | — | 2.03 | 3.97 / 0.19 |
| U-ESFM-52 weight+TTT | 0.58 / 0.14 | — | — | 2.05 | 2.89 / 0.11 |
| U-ESFM-39 madweight | — | 11.42 / 2.68 | 19.03 | 2.99 | **0.30 / 0.27** |
| U-ESFM-39 weight+TTT | 0.51 / 0.15 | — | — | 2.08 | 2.87 / 0.29 |
| RESfM-52 | 0.44 / 0.18 | **8.84 / 3.30** | **13.96** | 2.83 | 2.84 / 0.25 |

## Readings

1. **High contamination: schedule-dependent on 1DSfM-43** (original: madweight 5.50 vs RESfM 11.39; finelr twins: 12.39 vs 11.03 — within 1DSfM seed noise), but **consistent at 61%**: madweight 12.21 beats RESfM-finelr 16.22, and in the 52-pool 6.43 vs 8.84.
2. **Clean OOD (BMVS): madweight is the only mechanism that survives multi-dataset training** — 0.22–0.30 vs 2.8–4.0 for weight arms and RESfM-52. RESfM's clean-OOD strength (Strecha 0.005–0.006) collapses to 2.83 the moment its training pool diversifies.
3. **In-distribution: RESfM stays ahead** (0.33 vs 0.46–0.49 MD-only) — the honest tradeoff the paper already frames.
4. Diversity reallocates rather than adds: RESfM-52 gains 1DSfM (11.4→8.8) but loses Strecha (0.006→2.83); no across-the-board multids win → hold multids for ICLR.

## Still training (not in table)
RESfM menv (Ep8000/20k) · faithful, faithful_menv, lr1e4 (queued) · RESfM s21/s22 (~Ep9000) · RESfM-39 · deep multids pair · 100k arms (confirmatory).
