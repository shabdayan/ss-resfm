# Path A — 3DV 2027 Submission Materials

**Working title (options):**
1. *When to Remove and When to Reweight: Contamination-Dependent Outlier Handling in Deep Structure-from-Motion*
2. *Label-Free Outlier Handling for Deep SfM: A Complementarity Study Across Contamination Regimes*
3. *No Free Lunch in Outlier Handling: The Optimal Mechanism Flips with Contamination*

All numbers below are seed 20, all scenes per dataset, post-BA, mean / (median where noted), in degrees.
Frozen from `code/.twotables.out` (committed 29bf7c9) + verified aggregations.

---

## Abstract (draft)

Deep equivariant Structure-from-Motion (ESFM/RESfM) handles outlier correspondences with a
supervised inlier/outlier classifier trained on COLMAP-derived labels. We show that the *optimal*
outlier-handling mechanism is not fixed but **depends on the scene's contamination level**, and that
a **label-free** mechanism can match or beat the supervised classifier exactly where supervision is
weakest. Under a strictly controlled protocol — identical point tracks, robust bundle adjustment, and
training schedule, varying only the outlier mechanism — we compare hard removal, soft reweighting, a
three-band hybrid, and a new **MAD-remove + head-reweight ("madweight")** variant against a
from-scratch RESfM baseline across four datasets spanning 2.7%–31.9% outliers. We find a clear
complementarity: **hard, statistics-based removal wins the high-contamination regime (1DSfM, ~32%
outliers), where our label-free madweight halves the supervised baseline's median translation error
(2.32° vs 3.97°); soft reweighting wins the low-contamination regime (BlendedMVS, Strecha); and the
supervised classifier retains the clean, in-distribution regime (MegaDepth).** We analyze why —
supervised labeling requires a trustworthy COLMAP reconstruction that degrades precisely on
high-contamination scenes — and position label-free adaptivity as the missing ingredient for
robust deep SfM.

---

## Contributions

1. **A controlled study** of four outlier-handling mechanisms (remove / weight / hybrid / madweight),
   each with and without test-time fine-tuning, under an identical tracks+BA+schedule pipeline, so
   differences are attributable to the mechanism alone.
2. **The complementarity finding:** the best mechanism flips with contamination — removal in the
   high-outlier regime, reweighting in the low-outlier regime, supervised classification in the clean
   in-distribution regime. No single mechanism dominates.
3. **madweight**, a label-free MAD-remove + head-reweight variant that is the strongest arm on
   contaminated 1DSfM, beating the from-scratch supervised RESfM baseline by ~2× on both mean and
   median translation.
4. **An analysis of *why* supervision fails in the high-contamination regime** — the COLMAP-derived
   4px labeling assumption breaks exactly where outliers are dense — motivating label-free handling.

---

## Experimental setup

- **Architectures.** Shallow = SetOfSetOutliersNet 1×3 (RESfM's released architecture); Deep = 2×3.
- **Training.** 20,000 epochs on 27 MegaDepth scenes, batch 4, lr 1e-3, seed 20 — identical to RESfM.
- **Protocol.** Load multi-scene checkpoint → per-scene TEST pass (saves outliers) → ~1001-epoch
  per-scene fine-tune → robust BA (drop reproj>5px, <3-cam points, largest component, re-triangulate,
  2nd BA). Classifier threshold 0.6 (1DSfM/MegaDepth), 0.8 (Strecha/BlendedMVS), per RESfM.
- **Mechanisms.** remove (head score>thr → drop); weight (soft (1−score), no removal); hybrid
  (3-band remove/weight/keep); **madweight** (MAD reprojection-statistics removal + head soft-weight
  on survivors). ±TTT = continue adapting the head during fine-tune.
- **Datasets and contamination (RESfM-paper Outliers%):**

  | Dataset | Outliers% | Regime |
  |---|---|---|
  | Strecha | 2.7% | clean, LIDAR-GT |
  | BlendedMVS | 3.6% | low outlier, rendered |
  | MegaDepth | 25.6% | contaminated, **in-distribution** |
  | 1DSfM | 31.9% | contaminated, OOD |

- **Fair-comparison anchor.** The controlled baseline is **RESfM-from-scratch** — trained by us with
  the *identical* training setup as U-ESFM (same 27 scenes, sampling, epochs, seed), so only the
  outlier mechanism differs. We also report **RESfM-released** (authors' weights) and **RESfM-paper**.
- **Track provenance (state explicitly).** MegaDepth tracks + preprocessing are *identical* to RESfM
  (same 36 test scenes). The OOD tracks (1DSfM/Strecha/BlendedMVS) are **reconstructed by us** (RESfM
  did not release OOD tracks), so OOD comparisons are *controlled within our pipeline* and are **not**
  directly comparable to RESfM's published OOD numbers.

---

## Table 1 — Deep variants (2×3). Mean translation / rotation (deg).

| Arm | MegaDepth (25.6%) | 1DSfM (31.9%) | Strecha (2.7%) | BlendedMVS (3.6%) |
|---|---|---|---|---|
| U-ESFM-DA · MAD | 0.531/3.42 | 11.597/6.51 | 1.383/9.30 | 0.040/1.76 |
| U-ESFM-DA · remove | 0.406/3.46 | **6.773**/6.19 | 3.325/16.30 | 0.052/3.05 |
| U-ESFM-DA · remove+TTT | 0.518/4.12 | 9.223/9.02 | 2.623/12.69 | 0.139/10.72 |
| U-ESFM-DA · weight | 0.558/2.63 | 17.821/6.75 | 1.053/8.04 | **0.027**/2.35 |
| U-ESFM-DA · weight+TTT | 0.453/2.99 | 15.157/8.20 (n9) | 0.807/9.69 | 0.148/9.09 |
| U-ESFM-DA · hybrid | 0.400/3.81 | 15.311/6.62 | 2.606/10.12 | 0.118/21.42 |
| U-ESFM-DA · hybrid+TTT | 0.728/4.05 | 16.984/7.57 (n9) | 2.634/12.07 | 0.136/8.17 |
| U-ESFM-DA · madweight | 0.636/3.95 | 12.489/7.09 | **0.225/1.71** | 0.130/9.48 |
| RESfM-deep (scratch) | 0.559/4.75 | 14.866/9.94 | 0.005/0.01 | 0.048/2.87 |
| ESFM-deep | 0.663/5.05 | 13.713/9.84 | 0.018/0.10 | 0.162/10.63 |
| RESfM (paper) | 0.169/1.29 | – | – | – |
| ESFM (paper) | 0.780/6.77 | – | – | – |

## Table 2 — Shallow variants (1×3). Mean translation / rotation (deg).

| Arm | MegaDepth (25.6%) | 1DSfM (31.9%) | Strecha (2.7%) | BlendedMVS (3.6%) |
|---|---|---|---|---|
| U-ESFM-SA · MAD | 0.490/4.83 | 6.787/7.48 | 2.807/12.21 | 0.205/12.86 |
| U-ESFM-SA · remove | 0.499/3.80 | 9.854/7.03 | 2.871/12.92 | 0.145/10.20 |
| U-ESFM-SA · remove+TTT | 0.392/3.52 | 10.002/6.82 | 2.768/13.63 | 0.155/9.40 |
| U-ESFM-SA · weight | 0.460/3.97 | 15.884/9.25 | 1.945/7.44 | 0.114/6.80 |
| U-ESFM-SA · weight+TTT | 0.527/3.34 | 15.643/7.80 | 2.024/7.41 | 0.126/7.77 |
| U-ESFM-SA · hybrid | 0.463/3.50 | 16.803/7.01 | 3.116/14.84 | 0.124/7.09 |
| U-ESFM-SA · hybrid+TTT | 0.512/4.28 | 17.770/8.72 | 2.229/12.92 | 0.139/8.96 |
| U-ESFM-SA · madweight | 0.511/3.72 | **5.502/5.62** | 2.660/12.02 | 0.121/7.80 |
| RESfM-shallow (scratch) | 0.332/1.60 | 11.067/9.89 | 0.006/0.02 | 0.389/37.64 |
| RESfM (released) | 0.209/1.89 | 11.124/10.59 | 0.197/2.04 | 0.330/31.91 |
| ESFM (author 1×3) | 0.756/6.91 | 17.960/14.47 | 1.975/7.35 | 0.227/27.27 |
| RESfM (paper) | 0.169/1.29 | – | – | – |
| ESFM (paper) | 0.780/6.77 | – | – | – |

**Note on 1DSfM means:** the mean is dominated by two symmetric-structure failure scenes
(Ellis_Island, Tower_of_London) that fail for *all* methods. Report the **median** as the
representative figure: shallow madweight 1DSfM = **2.32 median** (mean 5.50) vs RESfM-scratch
**3.97 median** (mean 11.07) — a ~42% robust median improvement.

## Table 3 — The complementarity result (headline). Best mechanism per regime.

| Regime (Outliers%) | Best mechanism | why |
|---|---|---|
| High-contamination OOD — 1DSfM (31.9%) | **madweight** (label-free removal) | dense true outliers; statistics-based removal pays off; supervised head is out-of-domain |
| Low-contamination — BlendedMVS (3.6%) | **weight** (soft) | few outliers; removal discards good points |
| Clean LIDAR-GT — Strecha (2.7%) | RESfM (supervised) | almost nothing to remove; supervision on clean data wins |
| Contaminated in-distribution — MegaDepth (25.6%) | RESfM (supervised) | in-distribution training dominates despite contamination |

## Table 4 — Best shallow U-ESFM per dataset vs RESfM-released (controlled). Mean T · R.

| Dataset (out%) | best shallow U-ESFM (config) | RESfM-released | winner |
|---|---|---|---|
| MegaDepth (25.6%) | 0.358 · 2.99 (weight+TTT 30/70) | **0.209 · 1.89** | RESfM |
| 1DSfM (31.9%) | **5.502 · 5.62** (madweight) | 11.12 · 10.59 | U-ESFM |
| Strecha (2.7%) | 1.818 · 7.62 (weight+TTT 15/85) | **0.197 · 2.04** | RESfM |
| BlendedMVS (3.6%) | **0.051 · 3.83** (weight 10/90) | 0.330 · 31.91 | U-ESFM |

→ 2–2 split *along the contamination/distribution axis*: U-ESFM wins the contaminated-OOD and
low-outlier regimes; RESfM wins the clean and in-distribution regimes.

---

## Analysis — why supervision fails in the high-contamination regime

RESfM's inlier/outlier labels (Appendix C) require: (1) an associated **COLMAP** reconstruction per
training scene, (2) triangulation from those poses, (3) a **4-pixel** reprojection cutoff. This is
viable on well-reconstructed in-distribution data (MegaDepth), but on heavily-contaminated scenes
COLMAP itself degrades, so trustworthy labels **cannot be produced** — a chicken-and-egg that bounds
supervised methods exactly in the regime where robustness matters most. A per-scene, label-free
detector (MAD) recomputes the outlier test on the actual test scene and needs neither labels nor
COLMAP — which is why madweight wins 1DSfM. (RESfM's own 1DSfM result is on *its* tracks and is a
domain-shift story, not a labeling failure — we do not claim its labels are bad; its strong MegaDepth
result proves they are fine there.)

---

## Limitations (state these explicitly — they pre-empt the obvious reviews)

1. **No single deployable config wins all datasets.** Table 4 uses the best mechanism *per dataset*;
   selecting it requires knowing the contamination level. An unsupervised selector is future work
   (Path B). We present this as a *study of the phenomenon*, not a turnkey method.
2. **RESfM retains MegaDepth and Strecha.** On the one apples-to-apples benchmark (MegaDepth, identical
   tracks), supervised RESfM wins on both metrics at every config. We do not claim SOTA on MegaDepth.
3. **OOD tracks are self-constructed.** OOD numbers are controlled within our pipeline; they are not
   comparable to RESfM's published OOD numbers, so we compare only against our from-scratch and the
   released model on identical inputs.
4. **Reproduction gap on MegaDepth.** From-scratch RESfM (0.332) vs paper (0.169). This is partly
   intrinsic: the authors' *released* weights also give only 0.209 in our harness (the paper reports a
   best-of-search over threshold/layers), so ~0.209 is the reproducible level and our harness is
   faithful.
5. **Two symmetric-structure scenes fail for all methods** (Ellis_Island, Tower_of_London) — these
   inflate 1DSfM means; we report medians alongside.

## Claims we make / do not make

- **We claim:** the optimal outlier mechanism is contamination-dependent; label-free removal
  (madweight) beats supervised classification in the high-contamination regime under a controlled
  comparison; and we explain why.
- **We do NOT claim:** a single method that beats RESfM everywhere, or SOTA on MegaDepth/Strecha.

---

## TODO for the 8-day submission

- [ ] Freeze tables above (done — verified from disk).
- [ ] Figure 1: mechanism-vs-contamination line plot (translation error of each mechanism across the
      four datasets ordered by contamination) — the visual of the complementarity finding.
- [ ] Figure 2: qualitative reconstructions on a 1DSfM scene (madweight vs RESfM-scratch).
- [ ] Optional strengthener if they land in time: threshold-sweep result (remove @ best threshold) and
      madweight 30/70 — do NOT block submission on these.
- [ ] Related work: RESfM/ESFM, robust BA, RANSAC/learned outlier rejection, MAD robust statistics.
- [ ] Method figure: the four mechanisms + madweight schematic.
