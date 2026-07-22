# Roadmap

This roadmap reflects the current project state and the next major build priorities for the multimodal pancreatic cancer detection repository.

## Now: Stabilize The Hybrid Research Repo

Current baseline:

- `uv`-managed environment and lockfile
- modular helper layers under `src/`
- one primary notebook driving the full experimental flow
- tracked reports and figures for the latest results
- local-only raw data, processed data, embeddings, and model checkpoints

Highest-priority remaining work:

- continue refactoring reusable notebook logic into `src/`
- make the notebook more orchestration-focused and less implementation-heavy
- keep documentation aligned with the repo's real state
- preserve a clean boundary between tracked lightweight outputs and ignored heavy assets
- keep the thesis interpretation visible in the codebase: strong biomarker reproducibility, ambiguous CT generalization, and exploratory-only fusion claims

Status:

- Active delivery focus

## Next: Reproducibility And Research-Code Hardening

- extend the current lightweight test baseline beyond stable helper modules
- add more explicit run paths for common tasks such as report generation and preprocessing
- reduce notebook-only helper duplication in EDA and diagnostics sections
- tighten artifact naming and report consistency across `reports/` and `figures/`
- keep `uv` dependencies aligned with actual runtime imports
- keep the processed-data-first run path explicit so the notebook can run without raw assets in normal use
- codify negative-control and multi-seed evaluation patterns so future fusion work stays methodologically honest

Status:

- In progress

## Later: Cleaner Experiment Interfaces

The next major structural improvement after modularization is a cleaner runner surface around the existing notebook and modules.

Targets:

- expose common experiment steps through small scripts or thin CLI entry points
- make key reports reproducible without manually re-running the full notebook
- separate stable experiment interfaces from one-off thesis exploration code
- keep notebook usage focused on analysis, visual inspection, and narrative assembly
- make it easier to rerun the bias-analysis and summary-generation parts of the workflow outside a full notebook session

Status:

- Planned, not started

## Future: Expanded Research Extensions

Potential later work includes:

- stronger uncertainty tooling under `src/uncertainty/`
- cleaner support for repeated ablation runs
- broader experiment packaging for publication or demo purposes
- adaptation to real paired multimodal cohorts if such data becomes available
- external CT validation on additional institutions to resolve whether the current CT signal is pathological, domain-driven, or mixed
- domain-adversarial CT training, likely via a gradient reversal layer, to suppress dataset-of-origin information in learned representations
- matched CT-plus-biomarker cohorts so fusion can be evaluated as a real clinical question instead of a synthetic pairing exercise
- volumetric CT architectures or transformers once the data and validation setup justify moving beyond slice-based modelling

Status:

- Deferred until the current modularization and reproducibility work is stable

---

## Publication Plan

This section records the path from the current thesis to a peer-reviewed paper. It reflects an honest read of the three results: the biomarker branch is a clean reproduction (defensible, not novel), the CT branch is confounded by dataset-of-origin (cancer from Pancreas-CT-CB, control from PANCREAS; ResNet50 embeddings cluster by source at k=2), and fusion is exploratory under synthetic pairing. The publishable contribution is therefore the *bias-aware methodology and the shortcut-learning diagnosis*, not a multimodal performance claim.

### Now: Q2 paper, no new data (near-term, achievable from current results)

A reframed methods / reproducibility paper is publishable today in a decent Q2 venue without collecting anything new. The negative result *is* the contribution.

- **Reframe the narrative** away from "multimodal PDAC detection" toward: *"Near-perfect cross-dataset pancreatic CT classification is a domain-confounding artifact â€” a diagnostic protocol for detecting it, and evidence that pixel-level mitigation is insufficient."*
- **Lead contributions:** (1) a stepwise bias detection + mitigation pipeline (pixel-mean logistic probe AUC 0.866 -> 0.569 after extreme standardization); (2) the demonstration that the shortcut survives in deep features (K-means-by-source at k=2, silhouette peak, cross-cluster non-identifiability with two institutions); (3) independent reproduction of the Debernardi et al. (2020) urinary panel (AUC 0.944 vs 0.936); (4) a negative-control-based fusion-evaluation framework showing modality dominance under synthetic pairing.
- **Reuse existing figures:** dataset bias check, K-selection (elbow/silhouette), Grad-CAM, final model comparison, biomarker calibration â€” they already support this framing.
- **Target venues (Q2):** *Diagnostics*, *Journal of Imaging*, *BMC Medical Imaging*, *Computers in Biology and Medicine*, or a reproducibility / negative-results venue.
- **Effort:** weeks of rewriting, no new experiments. **Superseded 2026-07:** this was the recommended first submission while data acquisition was the Q1 blocker. PANORAMA is now staged, so the plan is Q1-direct and this reframe is retained only as a fallback (see Status below).

### Next: four steps required to reach Q1

A Q1 venue (npj Digital Medicine, Medical Image Analysis, IEEE TMI, Radiology: AI) requires new substance, because the field already has 2025 tooling for this problem. Do these in order of leverage:

1. **External multi-centre CT validation — DONE (data acquired 2026-07-04).** PANORAMA is staged locally: 2,238 studies / 2,224 patients, MD5-verified, masks and labels reconciled. Two findings change the plan: there is **no institution column**, so leave-one-*site*-out is impossible and the domain axis becomes **scanner (manufacturer)**; and scanner is itself a **measured confound** (Cramér's V = 0.44). Details in the local CT preprocessing audit and `reports/panorama_confound_audit.md`.
2. **Comparative evaluation of mitigation methods — NOT a proposed method.** Do *not* lead with gradient reversal: DANN is a 2015-16 method, and the DomainBed literature (Gulrajani & Lopez-Paz; *Failure Modes of DG Algorithms*, CVPR 2022) shows a well-tuned **ERM** matches or beats the whole family. Run a panel under one pre-registered model-selection rule — tuned ERM, **DFR** (last-layer retraining on a scanner-balanced subset), GRL, and an **SSL / foundation-model backbone** — and report honestly whether *any* of them removes a real, measured clinical confound. A rigorous negative result here is the contribution.
3. **Benchmark against ShortKit-ML** (medRxiv 2026; 20+ detection methods, 6 mitigation strategies) rather than presenting the protocol standalone. Also position against intermediate-layer knowledge distillation (arXiv 2511.17421) and feature-disentanglement benchmarks (arXiv 2602.18502).
4. **Genuine paired CT + biomarker cohort for fusion — OFF the critical path.** Collaboration-dependent and open-ended; it must not gate the Q1 paper. Defer to paper #2. Fusion stays a limitation paragraph, not a results section.

**Novelty guard:** Ong Ly et al. (*npj Digital Medicine* 2024, 72 citations) already published a shortcut-diagnosis protocol showing up to 20% performance overestimation from acquisition bias. "We propose a shortcut diagnostic protocol" is therefore **not** novel on its own. Differentiate on: two independent confounds in one protocol, a measured confound in the flagship public PDAC cohort, the `level` label-leakage finding, and whether mitigation actually works.

**Headline contrast for the paper:** the PANORAMA challenge winner (PanDx) reached **AUROC 0.926** on 957 held-out cases and only three teams beat the baseline — against our thesis CT model's **0.9999** on the confounded two-source set. That gap quantifies the inflation.

### Parallel option: standalone biomarker screening paper

The biomarker branch is the most translatable component (non-invasive, reproducible). A smaller separate paper could extend it with screening-utility analysis (decision-curve analysis, calibration, high-risk subgroup performance) and the original three-class task. Modest novelty, but a clean clinical-utility angle.

Status:

- **Recommendation updated 2026-07 — go Q1-direct, skip the Q2 reframe.** The original "Q2 first" advice assumed data acquisition was the blocker (6-12 months, uncertain). PANORAMA is now staged and audited, compressing the Q1 timeline to roughly 4-6 months. The two papers share ~70% of the same narrative, so publishing the Q2 reframe first would spend the story and make the Q1 submission look incremental.
- Q1 programme: step 1 complete (data); steps 2-3 are the work; step 4 deferred off the critical path.
- Realistic venues: **Medical Image Analysis** or **Radiology: AI**. npj Digital Medicine is a stretch without clinical impact, and IEEE TMI wants methodological novelty that an off-the-shelf GRL will not supply.
- Q2 reframe: retained as a fallback if the Milestone-A checkpoint (below) shows no image-level confound.
- Biomarker screening paper: optional parallel track; in the Q1 paper the biomarker branch serves as a **clean positive control**, not a results section.

### Milestone A — the go/no-go checkpoint (4-6 weeks)

Everything downstream is wasted effort until this resolves. ROI-first preprocess PANORAMA, train a baseline, and measure three numbers:

1. cancer AUC (in-distribution),
2. **scanner predictability from the learned embeddings** — does the metadata confound propagate into image features?
3. the **leave-one-manufacturer-out generalization gap**.

A large LOMO gap ⇒ the Q1 paper is real. A negligible gap ⇒ the scanner confound does not reach the images; pivot honestly (still publishable, smaller venue). Do not build steps 2-3 before A reports.


---

## Validation Datasets to Source (for the Q1 external-validation step)

The confound to break: in the thesis, cancer came from one source (Pancreas-CT-CB) and control from another (NIH PANCREAS), so *any* feature separating the two datasets also separated the two classes. The fix is validation cohorts where **cancer and control come from the same multi-centre pipeline**, so class is not tangled with institution. Ranked by usefulness:

1. **PANORAMA** (recommended primary). First public PDAC-detection grand challenge; to-date largest public PDAC CT dataset. Portal-venous contrast-enhanced CT, clinical metadata, segmentation masks for six PDAC-related structures, patient-level likelihood labels, **multi-centre with PDAC and non-PDAC from the same pipeline**, public leaderboard for honest benchmarking. This is the dataset that lets us decouple cancer from dataset-of-origin and also provides masks needed for pancreas-ROI localization. (arXiv 2503.10068)
2. **Medical Segmentation Decathlon (MSD) Task07 Pancreas**. 420 contrast-enhanced CTs with pancreatic lesions (PDAC, PNET, IPMN) from MSKCC, with tumour segmentation masks. Good independent single-source test and provides masks for ROI cropping.
3. **TCIA Pancreas-CT / NIH** (use with care). NIH Pancreas-CT is the *healthy/normal* set that formed the confounding control arm in the thesis; do NOT reuse it as controls against a different-source cancer set or the bias reappears. Useful only as a normal-pancreas reference within a same-source design.
4. **Benchmark comparator (cite, don't validate on):** PANDA, Nature Medicine 2023 â€” non-contrast CT, multi-centre validation on 6,239 patients, AUC 0.986â€“0.996. Sets the performance bar reviewers expect; reinforces that our contribution should be methodology/honesty, not raw performance.
5. **Published external-validation precedent to benchmark against:** 2025 radiomics PDAC study, internal 95% â†’ external 86.5% accuracy on TCIA/MSD â€” the kind of honest generalization-gap result to reproduce and report.

## Notebook Audit Findings (from 01_multimodal_cancer_detection.ipynb)

Concrete strengths and gaps found by reading the actual cells, to guide the rewrite.

**What was done well (keep):**

- Transparent bias *diagnosis*: pixel-mean logistic probe (Cell 1.7), ResNet embedding + K-means/elbow/silhouette domain audit (Cells 1.15â€“1.23, 1.41â€“1.45), per-cluster k=2 evaluation, cross-cluster generalization test.
- Patient-level stratified splits (Cell 2.0) â€” leakage control is correct.
- Biomarker branch is genuinely solid and under-sold: single clean source, plus calibration, permutation importance, decision-curve and gain/lift analysis (Cells 3.6â€“3.7). This is the most paper-ready component.
- Fusion done responsibly: multi-seed + label-mismatch negative controls for both decision- and feature-level fusion (Cells 4.1b, 4.2b).

**The central methodological gap â€” the bias check is partly circular:**

- Mitigation (`extreme_standardize`, Cell 1.10) forces body-pixel **mean=128, std=40**. The bias check (Cell 1.11) then tests whether a logistic model on **pixel mean/std** can separate classes. Because mitigation forces exactly those statistics equal, the probe necessarily drops to ~random (0.569). The detector and the fix target the *same low-order statistic*, so the "bias removed" conclusion is self-fulfilling.
- It does nothing about the higher-order cues a CNN actually exploits: noise/reconstruction-kernel texture, edge/frequency content, field-of-view and body-shape geometry, contrast-phase and slice-thickness signatures. That is exactly why K-means on ResNet embeddings still recovers the source split after standardization.
- Fix: the bias detector must probe the **learned feature space**, not raw pixel moments (e.g. a domain classifier on embeddings, or a dependence measure such as HSIC between representation and source), and mitigation must act in that space.

**CT modeling gap â€” receptive field too global:**

- The "cropped" dataset (`ct_cropped`) is a **whole-body** crop via segmentation, not a pancreas ROI. The ResNet50 still sees global body outline, FOV, and tissue-wide noise texture â€” all scanner/source fingerprints. Rapid convergence to ceiling AUC (Cell 2.7 history) is itself a tell of trivially separable domain signal.

**Upgrade for Q1 (feature-space debiasing â€” current 2025/26 standard):**

- **Superseded 2026-07 (see "four steps to Q1" above).** GRL is no longer the proposed fix. Run a *panel* under one pre-registered model-selection rule: a properly tuned **ERM baseline** (per DomainBed, the real bar to clear), **DFR** (last-layer retraining on a scanner-balanced subset), **GRL**, and an **SSL / foundation-model backbone**. On PANORAMA the domain label is **scanner**, not institution.
- Alternatives/complements from the 2025/26 literature: feature disentanglement (latent-space splitting), dependence-minimization (HSIC-style), knowledge distillation from a specialist teacher.
- Acceptance criterion, measured with our *own* K-means/silhouette + embedding domain-classifier diagnostic: the source-aligned clustering that pixel standardization could not remove should collapse, while genuine cancer signal is retained (verified on PANORAMA where class â‰  institution).
- Benchmark against **ShortKit-ML** (medRxiv 2026) and a 2025/26 dependence-measure or disentanglement baseline rather than presenting the pipeline standalone.

**Architectural note (ROI vs whole-image):** moving to a pancreas-ROI model (localize then classify) is good practice and removes the *easiest* global shortcuts, but it is necessary-not-sufficient: scanner/reconstruction texture lives inside the pancreas tissue too, and no receptive field fixes a data-design confound where one source = all cancer and the other = all control. The decisive fix is same-source class balance (PANORAMA) + feature-space debiasing; ROI cropping is a robustness improvement layered on top, and it requires pancreas masks (available in PANORAMA/MSD, absent in the thesis two-source set).


### PANORAMA access details (added)

- **License:** CC BY-NC 4.0 (non-commercial) - fine for thesis/paper and academic validation with citation; NOT usable in a commercial product without separate permission.
- **Download (images, ~193 GB):** the image set is published as **4 Zenodo batches** (CC BY-NC 4.0), images-only: batch_1 = record 13715870, batch_2 = record 13742336, batch_3 = record 11034011, batch_4 = record 10999754. File-download URL pattern: `https://zenodo.org/api/records/<record>/files/batch_N.zip/content`. Mirrored on TCIA (wiki.cancerimagingarchive.net/display/Public/PANORAMA). Challenge: panorama.grand-challenge.org. The batch zips contain **only CT volumes** as flat `<caseid>_<exam>_0000.nii.gz` files — no masks or labels.
- **Contents:** 2,238 anonymized contrast-enhanced CT scans from two Dutch centres (Radboud UMC + UMC Groningen), plus 194 MSD and 80 NIH cases - unified multi-centre labelled cohort where class is NOT tied to a single source.
- **Masks + labels (SEPARATE download — NOT in the Zenodo zips):** segmentation masks and patient-level labels come from the GitHub repo `DIAGNijmegen/panorama_labels` (~1.3 GB): `manual_labels/` + `automatic_labels/` (2,238 `.nii.gz` masks) and `clinical_information.xlsx` (patient-level labels / clinical data). These support the pancreas-ROI localization step the thesis two-source data lacked.
- **Baseline:** official implementation at github.com/DIAGNijmegen/PANORAMA_baseline (benchmark comparator).
- **Caveat:** PANORAMA folds in the NIH cases (same family as the old confounding control set). Use the unified labelled cohort as-is; do NOT extract the NIH subset as a standalone control arm or the dataset-of-origin confound returns.

---

## Q1 Experiment Battery (what reviewers will expect)

Beyond the four core steps, these analyses are near-mandatory for a strong medical-AI submission. The first three turn the CT result from "near-perfect" into "honestly characterized"; the rest are standard rigor.

- **Calibration, not just AUC.** Report Expected Calibration Error (ECE), Brier score, and reliability diagrams. Deployment needs calibrated probabilities. (Biomarker branch already has calibration + decision-curve code in notebook cells 3.6-3.7 - reuse it.)
- **Missing-modality robustness.** Evaluate CT-only, biomarker-only, both-present, and degraded inputs (missing CT, missing biomarker, noisy biomarker, low-quality CT). A fusion model is only interesting if it degrades gracefully.
- **Uncertainty / OOD detection.** Add predictive uncertainty (MC-dropout or deep ensembles) and an out-of-distribution flag. This is the distinctive angle: the same OOD machinery that flags an unseen scanner is what would have caught the original domain shift. "The system knows when it doesn't know."
- **Ablation table.** CT-only / biomarker-only / decision-fusion / feature-fusion / proposed, each with AUROC + CI, so fusion's marginal value (or lack of it) is explicit.
- **Subgroup / fairness reporting.** Performance by source/scanner, sex, and age where metadata allows - this is what makes "bias-aware" demonstrated rather than asserted.

## Statistical Rigor

- Report **95% confidence intervals** on all headline metrics (DeLong for AUROC; bootstrap for the rest). With small n, point estimates alone will be challenged.
- Note the **power limitation** explicitly given cohort sizes; pre-register the analysis plan where possible.
- Keep an **out-of-domain split** as the primary generalization metric, never random splits. Note: on PANORAMA, leave-one-*site*-out is **not possible** (no institution column), so the headline metric is **leave-one-manufacturer-out**; splits must also be **grouped by patient** (11 patients have >1 exam).
- Include a **properly tuned ERM baseline** and a single pre-registered model-selection rule in any debiasing comparison — without these the DomainBed critique invalidates the result.

## Reporting Standards & Checklists (attach at submission)

Q1 clinical-AI venues increasingly require a completed reporting checklist. Target compliance with:

- **TRIPOD-AI** (prediction-model reporting) and/or **STARD-AI** (diagnostic-accuracy studies).
- **CLAIM** (Checklist for AI in Medical Imaging) for the CT component.
- A **model card** (already drafted in docs/model_card.md - extend it) and a **data statement** (docs/data_and_ethics.md).

## Target Venue Shortlist (consolidated)

- **Q2, now (reframed shortcut-learning methods paper):** Diagnostics; Journal of Imaging; BMC Medical Imaging; Computers in Biology and Medicine; or a reproducibility / negative-results venue.
- **Q1, after the external-validation + feature-space-debiasing work:** npj Digital Medicine; Medical Image Analysis; IEEE Transactions on Medical Imaging; Radiology: Artificial Intelligence.
- **Biomarker-only screening paper (parallel):** a clinical or screening-oriented journal, leaning on the calibration + decision-curve analysis already implemented.

## Open Decisions To Resolve Before Writing

- ~~Which paper goes first~~ **RESOLVED 2026-07: go Q1-direct.** The data blocker is gone and the two papers share too much narrative for Q2-first to be safe. The Q2 reframe is retained only as a fallback if Milestone A shows no image-level confound.
- **Input geometry** for the v2 pipeline: 2D slices, 2.5D stacks, or full 3D volumetric. Open.
- **Mask policy**: use all 2,238 masks (482 manual + 1,756 automatic) or restrict to manual-only as a robustness arm — automatic masks carry label noise and provenance correlates with class. Open.
- Whether to pursue a real/quasi-paired CT+biomarker cohort for genuine fusion (collaboration-dependent) or keep fusion as an explicitly exploratory section.
- ~~Scope of feature-space debiasing~~ **RESOLVED 2026-07:** run the panel (tuned ERM + DFR + GRL + SSL backbone). GRL alone is not defensible as a contribution, and a tuned ERM baseline is mandatory.

