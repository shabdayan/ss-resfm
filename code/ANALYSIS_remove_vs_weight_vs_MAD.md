# Outlier handling: remove vs weight vs MAD (seed-20 PoC)

Same deep adaptive checkpoint (`uesfm_27scenes_adaptive` Ep16500). All three
arms run the identical protocol (per-scene inference -> outlier handling ->
1000-epoch fine-tune -> robust BA); the ONLY thing that varies is how the
outlier scores are used at test time:

- **MAD** — statistical MAD rule on reprojection errors (median + 2*1.4826*MAD)
  removes outliers. This is what all standard U-ESFM arms ship with.
  (`uesfm_deep_adaptive_{ds}_eval`)
- **REMOVE** (advhead) — the trained adaptive outlier head prunes points with
  score >= 0.6, then plain fine-tune. (`uesfm_deep_advhead_{ds}_eval`)
- **WEIGHT** (advweight) — no removal; the frozen head score soft-weights each
  point's reprojection loss by (1-score) during fine-tune
  (ESFMLoss_weighted_by_rep_err). (`uesfm_deep_advweight_{ds}_eval`)

## Results (seed 20; Rot deg / Trans / Nr)

| dataset | MAD | REMOVE (advhead) | WEIGHT (advweight) |
|---|---|---|---|
| 1DSfM (n=8, provisional*) | 6.16 / 12.26 / 292 | 7.44 / 8.83 / 236 | **8.95 / 8.15 / 316** |
| Strecha (n=4) | 1.61 / 0.220 / 13 | 6.87 / 1.90 / 13 | **0.05 / 0.011 / 13** |
| BlendedMVS (n=4) | 14.36 / 0.224 / 52 | 4.06 / 0.063 / 51 | **0.54 / 0.007 / 47** |

*1DSfM over 8/10 scenes (Notre_Dame, Vienna_Cathedral pending). Strecha/BMVS complete.

## Finding — soft-weighting is the best mechanism of the three

- **WEIGHT wins both Strecha and BlendedMVS outright**, by large margins, on both
  metrics. It is the only arm that neither under-removes nor over-removes:
  - **Strecha (~1-3% outliers):** WEIGHT 0.05 deg vs REMOVE 6.87 (advhead's hard
    threshold discards clean points -> disaster) and MAD 1.61. Soft-weighting
    down-weights instead of deleting, so nearly-clean scenes stay near-perfect.
  - **BlendedMVS:** WEIGHT 0.54 deg vs MAD 14.36 (MAD barely removes -> collapse)
    and REMOVE 4.06. Keeps/exceeds the OOD gain.
- **1DSfM (provisional):** WEIGHT has the **best translation** (8.15, the primary
  metric) and keeps the **most cameras** (Nr 316 vs REMOVE's 236), but the
  **worst rotation** (8.95). Mixed; the two pending big scenes may shift it.

## Why this is the mechanism story

Ties together the earlier findings:
- MAD collapses on OOD contamination (recall 1-4% on 1DSfM; ANALYSIS_removal_MAD_vs_head).
- The learned head detects real outliers but at only 48-74% precision, so HARD
  removal discards many inliers -> hurts on clean data (Strecha).
- SOFT weighting by the head score sidesteps both: no threshold to miscalibrate,
  nothing discarded; a misclassified inlier is only slightly down-weighted, a
  real outlier heavily down-weighted. Robust BA still does a final hard cleanup.
Net: probabilistic (soft) outlier handling > hard remove/keep AND > statistical
MAD. This is a candidate for the paper's main mechanism claim.

## Caveats / next steps

- Seed 20 only; 1DSfM missing 2 scenes. Lock in with the 2 pending scenes + full
  5 seeds before any paper claim.
- Under Fadi's translation-first priority, report translation mean AND median;
  1DSfM rotation regression needs the full set to interpret.
- Sharper weight functions ((1-score)^2 or a temperature) are an untested lever.
- Compare against RESfM (released / -deep / -shallow) on the same tracks once the
  from-scratch RESfM arms finish.
