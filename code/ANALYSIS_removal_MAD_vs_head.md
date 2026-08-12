# MAD vs learned head vs GT outliers (1DSfM, frozen-checkpoint TEST predictions)

Predictions from the SAME adaptive checkpoint (uesfm_27scenes_adaptive Ep16500),
TEST pass, before fine-tune. GT = npz `outlier_indices` at observed track points.
HEAD = adaptive outlier-head score ≥ 0.6 (advhead eval). MAD = `outlier_source=mad`
statistical rule (median + 2·1.4826·MAD on reprojection errors; the removal used
by ALL standard U-ESFM / two-stage-MAD arms). Metrics over observed track points.

| scene | GT out% | HEAD flag% | HEAD prec | HEAD recall | MAD flag% | MAD prec | MAD recall |
|---|---|---|---|---|---|---|---|
| NYC_Library | 54 | 34.2 | 74.2 | 47.2 | 1.2 | 59.7 | **1.3** |
| Alamo | 29 | 36.5 | 53.9 | 67.3 | 2.9 | 42.6 | **4.2** |
| Madrid_Metropolis | 29 | 32.1 | 47.5 | 52.4 | 2.2 | 31.3 | **2.4** |

## Finding

- **MAD collapses on high-contamination OOD data**: recall 1.3–4.2%. With 29–54%
  true outliers it flags ~1–3% of points and catches almost none. MAD's ~50%
  breakdown point plus a contaminated reprojection-error distribution (the frozen
  checkpoint reconstructs poorly on OOD, inflating all errors) push the threshold
  so high that almost nothing exceeds it. (On low-contamination sets — BlendedMVS
  ~1% — MAD flags ~0.7%, which is appropriate; the failure is OOD-specific.)
- **The learned self-supervised head works**: flags ≈ the true rate (32–37%),
  recall 47–67%, precision 48–74%. Imperfect (over-flags some inliers) but does
  real detection where MAD does none.

## Implication

Every standard U-ESFM arm and the two-stage-MAD baseline prune with MAD, so on
1DSfM they run with effectively **no outlier removal** (≈ ESFM + a better
checkpoint) — explaining the modest 1DSfM edge over ESFM and the translation loss
to RESfM. The pipeline **trains a learned head and then ignores it at test time**;
that head is the mechanism that generalizes to OOD contamination. Directly
motivates the learned-head arm (advhead PoC) as the fix.

## Caveats

- These are per-observation classification metrics on the FROZEN checkpoint; final
  pose accuracy depends on the precision/recall tradeoff after fine-tune + robust
  BA. The PoC (advhead vs MAD-based U-ESFM) measures the downstream effect.
- MAD camera set (328) aligned to GT (332) by first-N match (97.7%); ~2% noise.
- 3 scenes, seed 20. Extend to all 10 × 5 seeds for the paper figure.
