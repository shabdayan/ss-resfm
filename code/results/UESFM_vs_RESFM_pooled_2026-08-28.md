# Pooled config-matched table (finelr twin config; env-agnostic replicates)

Rule: all configs match (11-milestone schedule, margin 1e-4, our_repro selection, tracks, arch);
environment (train or eval) treated as a replicate axis alongside seed. U-ESFM lr = 1e-4;
RESfM lr = best of {1e-3, 1e-4} — twin-config 1e-4 run launched Aug 28 (job 402931), so
best-of currently = 1e-3. Cells: post-BA translation mean/median (deg), ±std where r>=2.

| Arm (lr) | MegaDepth (36) | 1DSfM (10) | 1DSfM-hard (5) | Strecha (4) | BMVS (4) | Olsson (39) |
|---|---|---|---|---|---|---|
| U-ESFM madweight (1e-4) | 0.42 / 0.15 (r1) | 11.5±2.3 / 6.0 (r3) | 12.21 / 8.83 (r1) | 2.50 / 0.97 | 0.146 / 0.004 | -- |
| U-ESFM weight (1e-4) | 0.42 / 0.18 (r1) | 12.79 / 11.01 (r1) | 21.94 / 13.13 | 2.00 / 0.026 | 0.100 / 0.036 | -- |
| U-ESFM weight+TTT (1e-4) | 0.41±0.09 / 0.17 (r4: 2env x 3seed) | 16.13 / 7.16 (r1) | 26.44 / 17.67 | 1.98 / 0.005 | 0.136 / 0.011 | -- |
| RESfM (best-of = 1e-3) | 0.347±0.004 / 0.08 (r2) | 11.03 / 2.93 (r1) | 16.22 / 13.08 | 0.006 / 0.005 | 0.379 / 0.348 | 8.47 / 0.17 (r1) |

Replicates used:
- U-wttt MD: s20-ourenv 0.316, s20-authorsenv 0.453, s21 0.370, s22 0.518
- U-mad 1DSfM: s20 12.39, s21 13.26, s22 8.88
- RESfM MD: ourenv 0.344, authorsenv 0.350

Pooling consequences vs seed-20-only: MD mean flips to RESfM (U-wttt indistinguishable at ±0.09);
1DSfM means tie (11.5±2.3 vs 11.0), RESfM keeps median; surviving structural wins = 1DSfM-hard
(12.2 vs 16.2) and BMVS (0.10-0.15 vs 0.38) — the two claims the paper leads with.

Gaps to grow the table: RESfM finelr seeds (training: 400449/400451 + menv trio 400497-501);
U finelr s21/s22 OOD fleets (checkpoints ready, not evaluated); U Olsson fleets (empty roots);
RESfM twin-config lr1e4 (402931); TTT same-env repeat (mvp fleet, running).
