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
