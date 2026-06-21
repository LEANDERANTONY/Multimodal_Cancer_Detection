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
- **Effort:** weeks of rewriting, no new experiments. This is the recommended first submission.

### Next: four steps required to reach Q1

A Q1 venue (npj Digital Medicine, Medical Image Analysis, IEEE TMI, Radiology: AI) requires new substance, because the field already has 2025 tooling for this problem. Do these in order of leverage:

1. **External multi-centre CT validation (THE blocker).** Re-run CT classification on cohorts where cancer and control are *balanced across sources*, so disease is decoupled from dataset-of-origin. Candidate public sources: MSD Task07 Pancreas, NIH Pancreas-CT, and additional TCIA PDAC collections across different scanner vendors/protocols. Without this, no CT performance claim is defensible.
2. **Domain-adversarial training (gradient reversal).** Add a GRL branch to the ResNet50 backbone that penalizes encoding dataset-of-origin. Evaluate rigorously: show whether it removes embedding-level domain clustering, and quantify the effect on the cancer signal. A rigorous negative result here is still publishable.
3. **Benchmark the bias pipeline against existing 2025 methods** (e.g. ShortKit-ML and related shortcut-detection frameworks) rather than presenting it standalone, to position the contribution against the current state of the art.
4. **Genuine paired CT + biomarker cohort for fusion.** Replace synthetic pairing with real or quasi-paired same-patient data (even 50-100 patients) so fusion can be evaluated as a real clinical question. Hardest step; collaboration-dependent. Consider attention/cross-attention fusion once paired data exists.

### Parallel option: standalone biomarker screening paper

The biomarker branch is the most translatable component (non-invasive, reproducible). A smaller separate paper could extend it with screening-utility analysis (decision-curve analysis, calibration, high-risk subgroup performance) and the original three-class task. Modest novelty, but a clean clinical-utility angle.

Status:

- Q2 reframe: ready to write (recommended first action)
- Q1 four-step programme: 6-12 months, new data required
- Biomarker screening paper: optional parallel track


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

- Add a **domain-adversarial branch (gradient reversal)** to the ResNet50 to penalize encoding of dataset-of-origin in the representation.
- Alternatives/complements from the 2025/26 literature: feature disentanglement (latent-space splitting), dependence-minimization (HSIC-style), knowledge distillation from a specialist teacher.
- Acceptance criterion, measured with our *own* K-means/silhouette + embedding domain-classifier diagnostic: the source-aligned clustering that pixel standardization could not remove should collapse, while genuine cancer signal is retained (verified on PANORAMA where class â‰  institution).
- Benchmark the pipeline against a 2025 dependence-measure or disentanglement baseline rather than presenting it standalone.

**Architectural note (ROI vs whole-image):** moving to a pancreas-ROI model (localize then classify) is good practice and removes the *easiest* global shortcuts, but it is necessary-not-sufficient: scanner/reconstruction texture lives inside the pancreas tissue too, and no receptive field fixes a data-design confound where one source = all cancer and the other = all control. The decisive fix is same-source class balance (PANORAMA) + feature-space debiasing; ROI cropping is a robustness improvement layered on top, and it requires pancreas masks (available in PANORAMA/MSD, absent in the thesis two-source set).


### PANORAMA access details (added)

- **License:** CC BY-NC 4.0 (non-commercial) - fine for thesis/paper and academic validation with citation; NOT usable in a commercial product without separate permission.
- **Download:** Zenodo (v1: zenodo.org/records/11034178 ; v2: zenodo.org/records/13742336), mirrored on TCIA (wiki.cancerimagingarchive.net/display/Public/PANORAMA). Challenge: panorama.grand-challenge.org.
- **Contents:** 2,238 anonymized contrast-enhanced CT scans from two Dutch centres (Radboud UMC + UMC Groningen), plus 194 MSD and 80 NIH cases - unified multi-centre labelled cohort where class is NOT tied to a single source.
- **Masks included:** segmentation masks for six PDAC-related structures - supports the pancreas-ROI localization step the thesis two-source data lacked.
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
- Keep **leave-one-site-out / external** as the primary generalization metric, never random splits (mirrors the agroforestry repo's discipline).

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

- Which paper goes first - the Q2 methods reframe (fast, low-risk) or hold for the Q1 swing. Recommendation on record: submit the Q2 reframe first; it banks a publication and de-risks the narrative.
- Whether to pursue a real/quasi-paired CT+biomarker cohort for genuine fusion (collaboration-dependent) or keep fusion as an explicitly exploratory section.
- Scope of feature-space debiasing: domain-adversarial only (minimum) vs. adding disentanglement/dependence-minimization baselines (stronger, more work).

