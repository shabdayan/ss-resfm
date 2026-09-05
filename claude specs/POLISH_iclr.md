# ICLR Submission Polish List (SS-RESfM)

Status legend: [ ] open · [~] in progress / blocked on data · (→ who)

## A. Data completions (must land before freeze)
- [~] **43% in-distribution cells** (1dsfmid + 1donly, both recipes × s20/s21; training
  ~Ep16k–19.5k) → fire held-out evals on completion, add the 43% column to the
  in-dist table, state where the supervised advantage inverts. Escalate to 5 seeds
  if the screen shows a claimable margin. (→ me, automatic)
- [~] **Olsson@0.5% clean control** (olssonid, ~Ep7k) → same; expected supervised win
  completes the "curated middle band" story. (→ me, automatic)
- [ ] **Re-run one Roman_Forum in-dist eval with `save_eval_diagnostics=True`** for the
  qualitative SS-vs-supervised figure (strongest closure of the "errors too high"
  objection). ~1 h. (→ me)

## B. Figures
- [ ] **Qualitative Roman_Forum figure** (from A3): side-by-side BA-aligned cameras,
  SS vs supervised, in the in-dist subsection.
- [ ] **Contamination-curve figure**: supervised-minus-SS in-distribution gap vs
  training-domain contamination (0.5 / 1.7 / 3.1 / 25 / 43 / 60%) — one plot that
  carries the whole causal story; becomes Fig. 2 candidate.
- [ ] Re-render Fig. 1 (complementarity) with final 5-seed numbers; check fonts/size
  at print scale.
- [ ] Check every figure caption is self-contained (reviewer skim path:
  abstract → figures → tables).

## C. Writing pass (order matters: do after A lands)
- [ ] **Abstract final pass**: add one clause for the causal in-dist result (it is
  currently absent from the abstract and it is the paper's newest strongest card).
- [ ] **Intro/contributions**: add the in-dist causal test to contribution (1) or as
  its own bullet; re-check the contribution list reads as 5 crisp items.
- [ ] Promote-or-keep decision: in-dist subsection placement (currently before the
  diversity subsection — consider moving directly after the main comparison).
- [ ] Consistency sweep of the two new subsections against final table values
  (rerun `verify_iclr_tables.py` extended with the in-dist cells; add those cells
  to the script).
- [ ] Kill remaining "Preliminary:" in the diversity subsection title if the
  in-dist + 2×2 results now make it non-preliminary.
- [ ] Page budget: main text crept past 10 pages with the new subsection — re-measure;
  candidates to compress: lr-grid pointer paragraph, classical-baselines paragraph.
- [ ] Global pass for stale "appendix" promises, dangling refs (grep `??` after
  compile), duplicate parentheticals.

## D. External / blocked on others
- [ ] **VGPA camera-ready PDF** (via Fadi): verify citation authors/title; read their
  1DSfM/MegaDepth tables; add one Limitations sentence if their numbers invite
  cross-paradigm comparison. (→ user/Fadi)
- [ ] **Fadi author tracks**: replication check if they arrive pre-deadline; one
  sentence in track-provenance if they corroborate. (→ external)
- [ ] **Advisor read-through** (Shai/Ronen): framing, baseline anchoring sign-off,
  venue confirmation. (→ user)
- [ ] **3DV withdrawal on the submission site** — must precede ICLR submission
  (dual-submission policy). (→ user)

## E. Logistics
- [ ] Swap placeholder ICLR 2026 style → official ICLR 2027 style when released;
  re-measure page budget after swap.
- [ ] Reproducibility statement + ethics statement (ICLR requires both sections;
  currently absent).
- [ ] Anonymity sweep: no author-identifying paths/acknowledgments; check appendix
  config table and code-release sentence.
- [ ] Code/data release prep: clean repo snapshot matching the config appendix
  (the CODEMAP doc is the internal guide; decide what ships).
- [ ] Final `verify_iclr_tables.py` run + prose-vs-table grep on the frozen candidate.
- [ ] Submission dry-run on OpenReview a few days early (PDF compiles under their
  checker, abstract field, keywords, TLDR).

## F. Promoted (3-week timeline confirms feasibility)
- [ ] **MegaDepth track-pipeline robustness check**: source 5–8 test scenes with numeric
  IDs from a per-scene mirror (try D2-Net/LoFTR undistorted-MegaDepth mirrors first;
  MegaDepth-X uses landmark names and needs mapping), rebuild tracks with our
  Appendix-C builder, evaluate EXISTING checkpoints (test-time only, no retraining),
  report "ordering/margins stable under independent track build" as one appendix
  paragraph. Closes the builder-confound objection. (→ me, week 1)

## G. Nice-to-have (only if time remains)
- [ ] Realistic-COLMAP-label third arm for the in-dist experiment (the "price of
  realistic labeling" number) — strongest possible extension, ~1 day of compute.
- [ ] Seed the multids two-stage (its 1DSfM 7.83 is single-seed; only if we want to
  mention it beyond a footnote).
- [ ] BMVS class-imbalance probe (train supervised with re-weighted BCE) to upgrade
  the imbalance hypothesis from plausible to tested.
- [ ] OpenReview fields: keywords "self-supervised learning; structure-from-motion; outlier rejection; pseudo-labels; robust 3D reconstruction; bundle adjustment; label noise"; TLDR as a search-result sentence; arXiv cross-list cs.CV + cs.LG at posting.
