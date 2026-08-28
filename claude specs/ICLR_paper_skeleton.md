# ICLR 2027 paper skeleton (working title)
**"Learning When to Remove: Contamination-Adaptive Outlier Handling for Self-Supervised SfM"**

Status: SKELETON (Aug 28 2026). Decision gate: multids verdict (~Sep 2–5).
Dual-submission constraint: <20% overlap with the 3DV submission; cite it as
concurrent anonymous work + include anonymized copy in supp; the 3DV content
appears here only as compressed, re-written motivation.

---

## 1. Introduction
- Hook: deep SfM outlier handling is a *fixed* mechanism chosen at design time,
  but the optimal mechanism is contamination-dependent (cite 3DV submission as
  concurrent; 1–2 sentences, no reuse of text).
- Gap: no deployable way to *choose* the mechanism per scene without labels.
  Cheap unsupervised selectors (contamination estimate, reproj stats, held-out
  CV, cycle consistency) fail to reach the oracle [Path B NO-GO — this negative
  result becomes motivation].
- Contribution 1: **learned selector** — a lightweight head trained
  self-supervised to pick remove/weight/hybrid per scene from scene statistics
  (embedding of track stats + model confidence distribution + partial-BA
  residuals). [TO BUILD — core work item]
- Contribution 2: **training-distribution study** — self- vs supervised outlier
  handling as data diversity (27→52 scenes, 4 domains) and schedule (20k→100k)
  scale. [IN FLIGHT — multids + 100k matrices]
- Contribution 3: first **author-track OOD evaluation** — directly comparable
  numbers on RESfM's own preprocessed OOD tracks (obtained from the authors)
  + track-provenance sensitivity analysis (their 31.9% vs our 43.3% rebuilds).
  [ARRIVES next week]

## 2. Related work
Deep equivariant SfM (ESFM/RESfM/GASFM); modern feed-forward SfM (VGGSfM,
DUSt3R/MASt3R, GLOMAP) — MUST add at least one as baseline [open item];
robust estimation / adaptive loss selection; self-supervised geometry.

## 3. Method: the learned selector
- Inputs (label-free, per scene): contamination proxies (MAD stats of reproj
  errors at a probe checkpoint), track-graph features (density, cams/pt
  distribution), head-confidence histogram, cheap partial-BA residual.
- Output: mechanism choice (or mixture weights) over {remove, weight, hybrid,
  madweight}; optionally also the removal threshold (percentile regression).
- Training signal WITHOUT test labels: self-supervised ranking on *training*
  scenes where all mechanisms are run and post-BA reprojection (not GT!) ranks
  them; or meta-split of training pool. Key novelty claim: selector trained
  with zero GT/COLMAP labels end-to-end.
- Baselines for the selector: oracle (upper bound), per-dataset fixed choice,
  the failed cheap selectors (from Path B), always-madweight, always-weight.

## 4. Experiments
### 4.1 Setup
Environment-controlled protocol (1 sentence + cite concurrent submission);
datasets: MegaDepth-27/52-pool training; OOD: authors' tracks AND our rebuilds
(provenance ablation); contamination axis 1.5–61%.
### 4.2 Selector results  [core table]
Selector vs oracle vs fixed mechanisms across all datasets × both track
provenances. Success bar: recover ≥80% of the oracle-over-best-fixed margin.
### 4.3 Data-diversity scaling  [multids matrix — lands Sep]
27 vs 52 scenes × shallow vs deep × U-ESFM vs RESfM: does self-supervision
close the in-distribution gap with diversity? Does OOD improve?
### 4.4 Schedule scaling  [100k matrix — partial by Sep deadline]
Convergence curves; preliminary finding to verify: supervised keeps improving
past 20k (best-val Ep22.5k) while self-supervised plateaus by Ep8.5k at
constant LR — interacts with milestone decay at 50k. Report whatever region
is complete by submission; full curves camera-ready.
### 4.5 Track provenance
Same arms on authors' vs our OOD tracks: do complementarity + madweight
conclusions replicate? Contamination re-measured uniformly on both.
### 4.6 Ablations
Selector input groups; selector generalization to unseen datasets (train
selector on MegaDepth+ETH3D scenes, test on 1DSfM/hard).

## 5. Analysis
- When does the selector fail? (per-scene keying was the Path B failure mode —
  does learned signal fix it?)
- Label-ceiling connection (1 para, cite concurrent).

## Risk register
- Selector doesn't beat per-dataset fixed choice → fallback framing: "scaling
  study + author-track comparability" (weaker; consider CVPR'28 instead).
- Multids results murky → drop 4.3 to ablation, lean on selector.
- ICLR deadline ~late Sept: selector needs first results by ~Sep 12 to be
  viable. Fadi tracks needed by ~Sep 8 for 4.5.

## Overlap budget vs 3DV paper (<20%)
Reused as re-written motivation only: complementarity finding (2 sentences),
label ceiling (2 sentences), protocol description (2 sentences). All tables,
figures, methods, and experiments here are NEW. The 3DV paper contains no
selector, no multids/100k results, no author tracks.
