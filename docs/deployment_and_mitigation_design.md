# Design: deployment-ROI experiment + mitigation panel

_Pre-registered design (written 2026-10-05, before running either). Practices borrowed from the Kaggle
grandmasters we study (`D:/Documents/Projects/Kaggle/Grandmasters/`: ren4yu, sersasj, samson8) are tagged
[ren4yu] / [sersasj] / [samson8]. Companion: `docs/q1_readiness_and_gaps.md`._

## Principles carried into both experiments
1. **Trust the validation before the model.** Patient-level bootstrap CIs, paired comparisons on the same
   patients, CV + LOMO + external MSD as the three fixed test settings. [all three]
2. **Strong baseline first, measured steps after.** Every new arm is compared with the tuned baseline under
   the same rule; no arm is judged on its own. [all three]
3. **Geometry exactly right before modelling** — measure the crop, not just the score. [ren4yu]
4. **Data for the failure you observe** — measure an error distribution, then train on inputs that contain
   it. [sersasj, samson8]
5. **Keep a "didn't work" list** in this doc, filled as we go. [all three]

## A. Deployment-ROI experiment (the novel piece)
**Question.** Our models were trained and tested on crops cut from the provided (oracle) pancreas masks,
which a clinic doesn't have. With a predicted pancreas mask instead: how much detection is lost, and does
the segmentation stage re-introduce the scanner confound?

**Stage 1 — pancreas segmenter (two, for robustness):**
- **Arm A, generalist: TotalSegmentator 2.18** (`--roi_subset pancreas`, full-resolution model, not
  `--fast`; off-the-shelf, never saw PANORAMA) — what a hospital would run.
- **Arm B, domain standard: the official PANORAMA baseline pancreas nnU-Net** (Dataset103, Zenodo
  11160381; nnU-Net v2 3d_fullres at 4.5×4.5×9 mm). This is the stage 1 the PANORAMA winner (PanDx,
  AUROC 0.926) reused unchanged, with the same 100×50×15 mm crop we use. Run **out-of-fold**: its fold file
  holds each of the 2238 cases out in exactly one fold, so every case is segmented by the fold model that
  never saw it (no training needed).

_Amendment 2026-10-06 (before any stage-1 run):_ arm B replaces "train our own nnU-Net pancreas
segmenter". Reasons: it is the field-standard stage 1 for this exact task, public weights make it
reproducible, and the official folds give a leakage-free out-of-fold run on all cases including MSD/NIH.
**Both segmenters are used as published, without our own improvements** — the experiment asks what happens
with a standard segmenter; tuning the segmenter on our own evaluation data would make it non-standard and
optimistic. The question "would a better segmenter close the gap?" is answered by the dose-response curve
below, not by tuning.

**Crop.** Same builder and margins as the oracle crops (`build_roi_dataset.py`, tight 100×50×15 mm), bbox
from the predicted mask instead of labels 4+5. Nothing else changes.

**Measure the segmentation and the crop first [ren4yu]** — per case and **per scanner**:
- **segmentation quality vs the reference masks** (pancreas Dice, against labels 4+5 and 1+4+5) — shows
  each segmenter performs at its published level on our data (TotalSegmentator smoke test, 1 MSD PDAC
  case: Dice 0.85, 96 % of the lesion inside the mask);
- bbox IoU with the oracle crop, centroid offset (mm), and **lesion containment** (fraction of the PDAC
  lesion inside the predicted crop) — the risk is a segmenter leaving the tumour out of "pancreas";
- segmentation failure rate (empty / implausible pancreas).
If containment or IoU differs by scanner, that is the mechanism by which stage 1 could re-inject the confound.
_Result, arm B on the 274 external scans (2026-10-06, `reports/deployment_roi/`):_ 0 empty masks; pancreas Dice median 0.55 (low by design at 4.5×4.5×9 mm); centroid offset median 3.5 mm (95th pct 11 mm); tight-crop IoU with the oracle crop median 0.85 (min 0.62); **all 98 MSD tumours entirely inside the predicted crop**. Same on MSD and NIH.
_Arm A, TotalSegmentator, same 274 scans:_ 0 empty masks; pancreas Dice median 0.85 (5th pct 0.61; its published level); crop IoU with the oracle median 0.93; centroid offset median 3.2 mm but a heavier tail (95th pct 14 mm, worst 32 mm). **Its mask holds only 80 % of the tumour on average (arm B 99 %)** — it tends to leave PDAC out of "pancreas", the risk named above — yet the tight-crop margins still keep **all 98 MSD tumours entirely inside the crop**. So on external data neither segmenter loses a tumour from the crop; any detection gap must come from crop framing, not missed lesions.

_Detection on predicted crops, arm B (2026-10-08; `tools/deploy_infer.py` + `tools/deploy_gap.py` → `reports/deployment_roi/deploy_gap_external.csv`; tight 5-fold + TTA; oracle side = `tight_external.csv`, reproduced locally to 4 decimals):_ MSD AUROC p_max 0.823 → 0.808, gap **+0.015 [−0.024, +0.056]**; cc_psz 0.781 → 0.759, gap +0.022 [−0.012, +0.057]; lesion Dice 0.555 → 0.544, gap +0.011 [−0.000, +0.025]; NIH specificity at the Dutch Youden threshold p_max 0.875 → 0.825 (4 of 80 more false positives), cc_psz 0.812 → 0.800. Paired patient bootstrap; every CI includes zero.

_Arm A, TotalSegmentator, same protocol:_ MSD AUROC p_max 0.823 → 0.833, gap **−0.010 [−0.045, +0.026]**; cc_psz 0.781 → 0.780, gap 0.000 [−0.029, +0.029]; lesion Dice 0.555 → 0.562 (gap −0.008 [−0.015, −0.000], predicted crops marginally better); NIH specificity p_max 0.875 → 0.825, cc_psz 0.812 → 0.850.

**External deployment verdict:** with either standard segmenter, used as published, detection and Dice on a new hospital match the oracle-mask pipeline; the paired gaps are centred near zero and bounded at ≈ +0.03 (TotalSegmentator) to +0.06 (baseline) AUROC. NIH specificity moves by a few scans of 80 in both directions. Next: the Dutch cohort (confound tax under predicted crops, per scanner).

_Stage 1 on the Dutch cohort, arm B (2026-10-10, interim: 1721 of 1964 scans; the 243 largest (> 150 M voxels) deferred to a pod because their export needs ~20 GB RAM; `reports/deployment_roi/stage1_quality_baseline_oof_dutch.csv`):_ 0 empty masks; per scanner (Siemens 871 / Toshiba 543 / Philips 270 / Other 37) pancreas Dice 0.58 / 0.59 / 0.56 / 0.56, centroid offset median 3.2 / 2.7 / 3.9 / 3.0 mm (95th pct 11.7 / 9.4 / 15.7 / 14.8 mm), crop IoU 0.87 / 0.87 / 0.86 / 0.87; 463 of 466 tumours entirely inside the predicted crop (the 3 partial: 2 Siemens, 1 Philips, 60–98 % inside). Crop placement is near-identical across scanners — Philips has a slightly heavier error tail, still inside the dose-response flat zone (≤ 20 mm) — so stage 1 alone is unlikely to re-inject the scanner confound; the tax under predicted crops (stage 2) is the test.

_Stage 1 on the Dutch cohort, arm A TotalSegmentator (2026-10-10, interim: 1702 of 1964; 262 largest deferred):_ **97 empty pancreas masks (5.7% segmenter failure; 0 on the external set)** — reported as deployment failures; per-scanner breakdown and geometry pending (`stage1_quality_totalseg_dutch.csv`). Stage 2 (out-of-fold tight model on predicted crops, `tools/deploy_infer_oof.py`) running locally; the deferred large scans (45% PDAC, Toshiba-heavy — must not be dropped) follow.

_Dose-response (2026-10-09; `tools/dose_response.py`, `tools/plot_dose_response.py` → `reports/deployment_roi/dose_response_{msd,summary}.csv`, `figures/panorama/deployment_dose_response.png`; MSD 194, tight 5-fold without TTA, Δ vs the unperturbed no-TTA reference 0.822 p_max, paired bootstrap):_ **centroid shift** 10 / 20 / 30 / 45 mm → Δ p_max −0.004 / −0.015 / **−0.073 [−0.139, −0.008]** / −0.060, tumours entirely inside the crop 100 / 100 / 94 / 81 %; cc_psz degrades from the first 10 mm (−0.030, −0.045, −0.066, −0.063). **Margin scale** 0.6 / 0.8 / 1.3 / 1.6 → Δ p_max −0.010 / −0.036 / **−0.069 [−0.134, −0.006]** / **−0.097 [−0.169, −0.030]** with every tumour inside: shrinking is harmless, enlarging hurts (the model was trained on tight crops; consistent with loose < tight). Both real segmenters' centroid errors (median ~3 mm, whiskers ~11–13 mm) sit inside the flat region (≤ 20 mm), which is why there is no deployment gap; the few > 25 mm misses are where jittered-crop training would help. Reading: crop *placement relative to the training geometry* matters, not box overlap (IoU 0.53 at scale 0.6 costs nothing).

The reference masks are partly machine-generated themselves (PANORAMA automatic labels), so the oracle is
an upper bound on crop placement, not perfect manual truth — stated in the paper.

**Dose-response curve (pre-empts "a better segmenter would fix it").** Shift / scale the oracle crop by
increasing amounts (centroid offsets in mm, box scale factors) and plot detection AUROC against crop
error, with the two real segmenters placed on the same axes. This shows how much segmentation quality
matters and where each segmenter sits, so the conclusion does not depend on the segmenter chosen.

**Then the scores** (tight model primary; loose if time allows):
1. External MSD/NIH (274): detection AUROC + Dice, oracle vs predicted crop, paired bootstrap.
2. Dutch CV (out-of-fold, 1964): detection AUROC, **confound tax under predicted crops** vs oracle
   (+0.031 [+0.010, +0.053]), per-scanner AUROC.
Headline: the deployment gap (oracle − predicted) and whether the tax grows.

**Fix the gap, if there is one [sersasj + samson8]:** fine-tune stage 2 on **perturbed crops** — random bbox
shifts/scales drawn from the *measured* segmenter error distribution (not arbitrary jitter) — so the
classifier learns to tolerate stage-1 errors. Re-measure the gap. This turns the finding into a remedy.

## B. Mitigation panel (critical for Q1)
**Question.** On a real clinical confound (scanner), do debiasing methods beat a well-tuned standard model?

**Backbone: 2.5D, local (fits the 8 GB GPU) [ren4yu, sersasj].**
- Pretrained 2D backbone on **slice triplets** (adjacent slices as RGB) inside the tight ROI;
- features pooled across depth with **attention pooling + an auxiliary per-slice loss** (what made ren4yu's
  multi-slice model train);
- sample all lesion slices + evenly spaced negative slices per scan [sersasj].

**Arms (scanner manufacturer = domain label):**
| Arm | What |
|---|---|
| ERM (tuned) | standard training, the bar every other arm must clear |
| Group-balanced sampling | equal scanner × label sampling (cheap baseline debiaser) |
| DFR | retrain the last layer on a scanner-balanced held-out split |
| GRL / DANN | adversarial scanner head with gradient reversal |
| SSL (MAE) | **masked-autoencoder pretraining on all CT incl. external-domain scans, then fine-tune** — sersasj used MAE for robustness to an unseen domain |

**Pre-registered model selection (one rule for every arm, fixed now):** select by mean validation AUROC
across scanners, using a **model soup** of the last checkpoints rather than a single best epoch [sersasj].
3 seeds per arm; report mean ± SD.

**Evaluation:** the same three settings as the nnU-Net models — CV confound tax, LOMO, external MSD — so
mitigation results line up with the main results.

**Planted-shortcut control [samson8's synthetic-data habit]:** add an artificial scanner-correlated marker
(e.g. a faint intensity offset applied to one scanner's training scans) to create a shortcut we *know* is
there. A mitigation method that can't remove a planted shortcut can't be trusted on the real one. This
validates the methods themselves and is cheap.

**Expected readings:** if tuned ERM ≈ the debiasers on the real confound (consistent with the small tax),
that is a citable negative result; the planted-shortcut control shows the methods work when there is
something to remove.

## C. Fusion arm rule (synthetic biomarkers) [samson8]
Synthetic samples may augment training only; they are always mixed with real data; **every reported number
comes from real patients only**.

## Didn't work
_(fill as we go)_
