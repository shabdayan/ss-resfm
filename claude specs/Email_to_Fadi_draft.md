# Questions on RESfM

**To:** Fadi Khatib (RESfM, first author)
**Subject:** Follow-up on RESfM — reproducing baselines and evaluation conventions

Dear Fadi,

Thank you again for the earlier discussion and for the metric-priority guidance. I am working on an unsupervised / self-supervised approach to outlier handling for equivariant SfM, building directly on RESfM, and I have reached a point where a few specifics about your setup would materially change what I run next. I have ordered them by how much each answer would help; the first four are the ones I would most value.

**1. Your preprocessed point tracks for the cross-dataset experiments (1DSfM, Strecha, BlendedMVS).**
I reconstructed these tracks myself and find they are systematically more contaminated than your published outlier percentages — on 1DSfM, mean 43% outliers on my tracks versus 31.9% in your Table 2, higher on 8 of the 10 scenes. Since your outlier head and its 0.6 removal threshold were fitted to your track construction, evaluating your released checkpoint on my tracks is not a like-for-like comparison. On MegaDepth, where our tracks are identical, your released checkpoint reproduces cleanly (about 1.4× the published rotation, consistent with BA-environment differences) — so the evaluation path is fine; it is the out-of-distribution track distribution I cannot match. If you could share the preprocessed 1DSfM/Strecha/BlendedMVS tracks, or the exact construction procedure and their source, it would let me report the decisive "your checkpoint, your tracks, my evaluation" row.

**2. Your trained ESFM and GASFM weights.**
Section 4.2 notes both were retrained by you on MegaDepth with your tracks (ESFM twice, including ESFM\* on outlier-free tracks), each with 1000 epochs of inference-time fine-tuning. Loading the public ESFM release does not reproduce your columns and surfaces as an unexplained gap. Your weights would remove a full retrain and a source of divergence.

**3. GLOMAP (and Theia / COLMAP) versions and settings, and how the precomputed tracks were passed in.**
Section 4.3 says all methods ran on the same tracks, but these pipelines normally build their own from features and matches, so the ingestion path is a real interface decision. GLOMAP is the baseline I most need to reproduce faithfully, since on 1DSfM it beats RESfM on rotation (1.27 vs 3.98) while losing on translation.

**4. The metric priority you gave me — translation, then rotation, then registered cameras — is it lexicographic or weighted?**
That is, does translation decide the winner outright with the others as tie-breaks, or can a large rotation gap outweigh a small translation one? This directly determines how I state my main claim, since my current cross-dataset result is strongest on rotation.

If you have a little more time, four further questions would help:

**5.** Running your released checkpoint on a few test scenes, what gap to the published numbers would you consider normal rather than a sign that my evaluation path is wrong?

**6. What was the outlier-percentage distribution across the 27 training and 4 validation scenes?**
The paper reports contamination for test scenes only, and I need this to say anything defensible about whether the head was exposed to high-contamination regimes.

**7. Did you ever regenerate labels on a target domain such as 1DSfM and retrain the outlier head there?**
This is the obvious steelman against a claim that the supervised head does not transfer, and I would rather learn from you whether it has been tried than spend a week on it.

**8. How much of the outlier robustness comes from the learned head versus the robust BA?**
Table 11 shows robust BA doing a great deal on its own (Tower of London 0.67 vs 57.41). Did you ablate a version with the head replaced by a purely statistical rule such as a MAD or percentile threshold on reprojection error? If a statistical rule matches the learned head, that reframes my paper considerably.

Thank you very much — even partial answers to 1, 4, 7 and 8 would save me significant compute and sharpen the claim.

Best regards,
Ortal Dayan
