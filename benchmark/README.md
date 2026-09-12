# RUC-SfM: Robustness Under Contamination for Structure from Motion

*(working name — final name set at release)*

A paradigm-agnostic benchmark asking one question: **how does a camera-pose
estimation method behave as track/match contamination rises from 0.5% to 61% —
and how reliably, across training seeds?**

## Why this benchmark exists

Every SfM paper reports accuracy; almost none report *reliability*. This
benchmark family's prior work fixes a single seed and compares means computed
over each method's own registered cameras. We found (see the paper) that on
these benchmarks: published numbers can sit in unreproducible
environment/checkpoint draws; training is a seed lottery with scene-level
catastrophic failures; supervised outlier labels collapse above ~15%
contamination; and metric-family choice (raw error vs. coverage-priced AUC)
can invert verdicts. The benchmark packages the data, protocol, and reference
numbers that make these failure modes measurable for any method.

## What a participant provides (the paradigm-agnostic contract)

Your method may consume **anything** derived from the scene inputs — our
standardized track tensors, the underlying images (public sources, build
scripts included), view graphs, or pointmaps. It must output, per scene:

- camera rotations `R [n,3,3]` and translations `t [n,3]` (world-to-cam),
- the set of registered camera indices (unregistered cameras are scored as
  failures by coverage-priced metrics, not dropped silently),
- and this **five times, from five independent training runs (seeds)** for any
  learned component. Single-run submissions are marked *unbanded* on the
  leaderboard.

`evaluate.py` scores a directory of such predictions against the GT.

## Datasets (all derived from public sources — see ATTRIBUTION)

| Suite | Scenes | Measured contamination | Regime |
|---|---|---|---|
| MegaDepth (RESfM split) | 36 test / 27 train | 25.4% | in-distribution |
| Olsson | 39 | 0.5% | clean OOD |
| Strecha | 4 | 1.7% | clean OOD (54 cams) |
| BlendedMVS | 4 | 3.1% | clean OOD |
| 1DSfM | 10 | 43.3% | heavy OOD |
| 1DSfM-hard-300 | 5 | 60–61% | extreme OOD |
| + 6 in-distribution training pools | | 0.5/3.1/43/60% | causal axis |

Contamination is measured uniformly (robust GT-pose triangulation, 4px cutoff).
Track tensors ship as npz (`M [2m,n]`, `Ps_gt`, `Ns`, `outliers2`,
`namesList`); builder scripts regenerate everything from the public sources.

## Metrics (all reported; no single family carries a verdict)

1. Translation/rotation error over registered cameras (mean and median,
   similarity-aligned) — the field's convention.
2. Registration coverage `Nr/Nc`.
3. Pairwise pose AUC@{1,5,30}° and registration-aware **RA-AUC** (unregistered
   pairs = 180°) — coverage-priced, alignment-free.
4. Per-scene win counts vs. a declared baseline.
5. **Reliability census**: seed-band width and catastrophic scene×seed cells
   (error >30 / >100). Reliability is the one verdict all metric families
   agree on.

## Reference results

`reference_results.json` carries five-seed bands for nine methods
(self-supervised arms, supervised RESfM, ESFM, ESFM*, no-mechanism, GASFM,
COLMAP, GLOMAP) — every number machine-verified against raw evaluation
outputs (`code/verify_iclr_tables.py`, 322 checks).

## Files

- `evaluate.py` — score a predictions directory against a suite.
- `export_reference.py` — regenerates `reference_results.json`.
- `../code/build_tracks/` — suite builders (from public sources).
- `ATTRIBUTION.md` — per-dataset licenses, sources, and citations.

Release: data DOI (Zenodo) and public repository at publication.
