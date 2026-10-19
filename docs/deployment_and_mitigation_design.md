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

## Status and next steps (updated 2026-10-13, all local; Zenodo still throttled to ~0.2 MB/s per connection, so no pods)

**Deployment-ROI — done:** external (no gap), dose-response, Dutch full cohort for both pre-registered arms
(baseline segmenter: AUROC −0.013, tax n.s.; TotalSegmentator as published: 118 empty masks, deployed
AUROC 0.787 → 0.700, tax +0.031 → +0.046). Cause of the TotalSegmentator failures found: the coarse 6 mm
cropping pass of `roi_subset pancreas` picks a tiny, misplaced region (case 100002: 73×81×52 voxels → 0);
probe on all 118 failures (`tools/totalseg_organ_probe.py`, `totalseg_organ_probe.csv`, 2026-10-14): requesting
neighbouring organs finds the pancreas on **118 / 118**, the whole `total` task (no cropping pass) on **117 / 118**,
but the developers' documented fix `robust_crop` (3 mm cropping model) on only **63 / 118 (53 %)** — and only
29 / 67 (43 %) of the PDAC failures (non-PDAC 34 / 51). So the cause is the pancreas-only cropping pass, the
documented remedy recovers about half, and adding neighbouring organs (or running the full task) recovers all.

**Running / queued locally (one chain, resumable; log `data/processed/ct/probe.log`, then
`data/processed/ct/totalseg_sensitivity.log`; queue script `data/processed/ct/queue_totalseg_nb.ps1`):**
1. ~~Organ probe `full`~~ done: 117 / 118.
2. ~~Organ probe `robust`~~ done: 63 / 118.
3. ~~Sensitivity arm `totalseg_rc`~~ done 2026-10-16: **66 empty masks (3.4 %) vs 118** — recovers 63 of the 118 but creates 11 new failures; Siemens 43 / 938 (30 of 170 PDAC). Deployed (ITT) mixed cc_psz 0.787 → 0.719 (−0.068 [−0.091, −0.044]; as published −0.087), within-Siemens 0.807 → 0.634, tax +0.031 → +0.046 (+0.015 [−0.001, +0.031]); when it works (paired, n = 1898) accuracy matches the oracle (−0.001). So the documented fix recovers about a fifth of the AUROC loss and leaves the confound increase unchanged.
4. ~~Sensitivity arm `totalseg_nb`~~ done 2026-10-18: **0 / 1964 empty masks** (all 118 recovered, none new). Deployed
   mixed cc_psz 0.787 → **0.782 (−0.004 [−0.015, +0.007])**, tax +0.031 → **+0.034 (+0.003 [−0.006, +0.012])**,
   within-scanner Siemens 0.811 / Toshiba 0.765 / Philips 0.668 — matches the oracle crop and is slightly better
   than the official baseline segmenter (0.774, tax +0.036).

**Four-arm summary (Dutch, 1964, out-of-fold, failures called negative; `deploy_gap_dutch.csv`, `deploy_fail_dutch.csv`,
figure `figures/panorama/deployment_arms.png` via `tools/plot_deployment_arms.py`):**

| Stage-1 segmenter | Empty masks | Deployed AUROC (cc_psz) | Δ vs oracle 0.787 | Tax (oracle +0.031) | Within Siemens (oracle 0.807) |
|---|---|---|---|---|---|
| Official PANORAMA baseline (trained on this cohort) | 0 | 0.774 | −0.013 [−0.025, +0.000] | +0.036 | 0.783 |
| TotalSegmentator, `roi_subset pancreas` (as published) | 118 (6.0 %) | 0.700 | −0.087 [−0.111, −0.063] | +0.046 | 0.587 |
| TotalSegmentator + `robust_crop` (documented fix) | 66 (3.4 %) | 0.719 | −0.068 [−0.092, −0.045] | +0.046 | 0.634 |
| TotalSegmentator + neighbouring organs | 0 | 0.782 | −0.004 [−0.015, +0.007] | +0.034 | 0.811 |

Reading: the deployment cost and the confound increase come entirely from segmenter *failures*, which fall on
PDAC and Siemens scans; when every scan gets a crop, a deployed two-stage pipeline matches the oracle and the tax
stays ≈ +0.03. The developers' `robust_crop` recovers half the failures but none of the confound increase; requesting
the neighbouring organs (or the full task) removes the failures at the cost of ~50 % more segmentation time.

**After the chain (analysis, CPU, ~1 h):**
5. Summarise the probe (pancreas found per mode) → `docs` + `DEAD_ENDS.md` (pancreas-only roi_subset is a
   usage trap; say whether robust_crop fixes it).
6. ~~`tools/deploy_gap_dutch.py` all four arms~~ done (table above).
7. ~~Stage-1 quality, all arms, full 1964~~ done 2026-10-19 (`stage1_quality_<arm>_dutch.csv`; failures count as tumour not in crop):

   | Arm | Empty | Pancreas Dice (median) | Centroid offset median / 95th pct (mm) | Crop IoU vs oracle | Tumours fully inside crop (578) |
   |---|---|---|---|---|---|
   | Official baseline segmenter | 0 | 0.58 (low by design, 4.5 mm grid) | 3.2 / 12.1 | 0.87 | 99.3 % |
   | TotalSegmentator, pancreas only | 118 | 0.87 | 2.9 / 17.6 | 0.93 | 81.8 % |
   | TotalSegmentator + robust_crop | 66 | 0.86 | 3.1 / 22.0 | 0.93 | 84.8 % |
   | TotalSegmentator + neighbouring organs | 0 | 0.89 | 2.8 / 14.9 | 0.94 | 96.0 % |

   TotalSegmentator draws a far more accurate pancreas than the baseline segmenter, yet the baseline keeps more tumours
   inside the crop (its coarse mask includes the lesion); with neighbours requested, 23 tumours are only partly inside
   and detection still matches the oracle, so partial containment costs little.
   quality CSV covers 1702); refresh `totalseg_failure_*` with the 21 large-scan failures.
8. Write the deployment-ROI results paragraph + figure (AUROC and tax per arm, failures per scanner × class).

**Jitter fine-tune (pre-registered "fix the gap, if there is one"): not triggered.** The baseline segmenter's
gap is −0.013 AUROC with an unchanged tax, and both segmenters' centroid errors sit in the flat region of
the dose-response; TotalSegmentator's loss comes from empty masks, which no crop-jitter training can fix.

**Then, in order:** B. mitigation panel (2.5D *classifier* panel below — tuned ERM / group-balanced / DFR /
GRL / MAE-SSL + planted-shortcut control; local 8 GB GPU) → C. biomarker / fusion arm → statistics (DeLong,
calibration, subgroups) → thesis-cohort control → figures and write-up.

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

_Stage 1 on the Dutch cohort, arm A TotalSegmentator (2026-10-10, interim: 1702 of 1964; 262 largest deferred):_ **97 empty pancreas masks (5.7% segmenter failure; 0 on the external set)** — reported as deployment failures. Per scanner (Siemens 864 / Toshiba 536 / Philips 265 / Other 37): empty 64 / 18 / 14 / 1 (7.4 / 3.4 / 5.3 / 2.7 %), so failures are uneven across scanners and Siemens-heavy; centroid offset median 2.9 / 2.4 / 4.1 / 4.8 mm (`stage1_quality_totalseg_dutch.csv`). Stage 2 on these crops running locally.

_Why TotalSegmentator fails (2026-10-12; `tools/totalseg_failure_analysis.py`, `tools/totalseg_failure_model.py`, `totalseg_failure_features.csv`, `totalseg_failure_model.csv`):_ failures are **3× more common on PDAC scans (54 / 461 = 11.7 %) than non-PDAC (43 / 1241 = 3.5 %)**, and they track **scan coverage**: failed scans have the pancreas far from the top of the scan (median 246 mm of scan above it vs 94 mm) and close to the bottom (90 mm below it vs 256 mm), i.e. chest-plus-upper-abdomen scans rather than abdomen-pelvis scans. PDAC scans ending 50–150 mm below the pancreas fail 44–47 % of the time (non-PDAC 3–9 %); scans reaching > 150 mm below fail ≤ 4 % in both classes. Slice thickness, field of view, pixel spacing, sex and pancreas HU matter little. With scanner and label matched away (97 failures + 291 controls, same scanner and label), the features predict failure at held-out AUROC 0.83 (logistic) / 0.86 (boosted trees), and pancreas position in the scan dominates the permutation importance. Reading: TotalSegmentator struggles when a tumour-bearing pancreas sits low in a chest-heavy (likely staging) scan; since that protocol correlates with the label, segmenter failure is itself a label-correlated pathway, and under intention-to-treat it falls on PDAC cases. A neighbouring-organ probe (does it still find liver, spleen, kidneys on these scans?) follows once the GPU is free.

_Stage 2 on the Dutch cohort, arm B (2026-10-11, interim: the same 1702 non-deferred cases, each scored out-of-fold with TTA, paired with its oracle CV score; `tools/deploy_gap_dutch.py`, `detect_baseline_oof_dutch.csv`):_ no stage-2 failures. **The confound tax does not change under predicted crops**: cc_psz +0.034 → +0.031 (predicted − oracle −0.002 [−0.015, +0.011]), p_max +0.027 → +0.033 (+0.006 [−0.014, +0.025]), p_sum +0.023 → +0.027 (+0.004 [−0.007, +0.018]). Mixed-scanner AUROC drops slightly: cc_psz 0.793 → 0.784 (−0.009 [−0.022, +0.004]), p_max 0.780 → 0.776 (−0.004 [−0.021, +0.012]), p_sum 0.731 → 0.715 (−0.016 [−0.027, −0.005], the only CI excluding 0). Within-scanner cc_psz AUROC: Siemens 0.805 → 0.785, Toshiba 0.794 → 0.804, Philips 0.680 → 0.670, all CIs include 0. Interim, on the non-deferred subset (oracle tax here +0.034, vs +0.031 on all cases); the deferred large scans (45% PDAC, Toshiba-heavy — must not be dropped) follow.

_Stage 2, arm B, FULL Dutch cohort (2026-10-13; all 1964 cases incl. the 262 large scans, 0 segmenter failures):_ the oracle side reproduces the published tight tax exactly (cc_psz +0.031). **Tax under predicted crops: cc_psz +0.031 → +0.036 (predicted − oracle +0.006 [−0.005, +0.017]); p_max +0.026 → +0.034 (+0.008 [−0.007, +0.023]); p_sum +0.026 → +0.032 (+0.006 [−0.004, +0.017]) — no significant change, and every point estimate stays small and positive.** Accuracy drops slightly: mixed AUROC cc_psz 0.787 → 0.774 (−0.013 [−0.026, −0.000]), p_sum −0.019 [−0.029, −0.008], p_max −0.004 [−0.021, +0.010]; within-scanner cc_psz Siemens 0.807 → 0.783 (−0.024 [−0.042, −0.005]), Toshiba 0.784 → 0.773, Philips 0.676 → 0.656 (CIs include 0). Reading: deploying with the official baseline segmenter costs about 0.01 AUROC, spread evenly across scanners, and does not amplify the scanner confound.

_Stage 2, arm A TotalSegmentator, FULL Dutch cohort (2026-10-13; 1964 cases, **118 empty masks = 6.0 %**: Siemens 74 / 938 (40 of 170 PDAC = 24 %), Toshiba 27 / 651 (18 of 134 PDAC), Philips 16 / 319, Other 1 / 56; failed cases have no crop):_ **When it works (paired, n = 1846)**, detection matches the oracle: mixed cc_psz 0.787 → 0.781 (−0.007 [−0.019, +0.006]); tax +0.029 → +0.038 (+0.009 [−0.003, +0.022]); p_max tax +0.026 → +0.043 (+0.017 [+0.002, +0.033], the one tax increase whose CI excludes 0); but within-Siemens cc_psz falls 0.797 → 0.743 (−0.054 [−0.095, −0.017]). **Deployed as is (intention-to-treat, n = 1964, a failure = called negative)**, accuracy collapses: mixed cc_psz 0.787 → 0.700 (−0.087 [−0.111, −0.063]), p_max −0.077, p_sum −0.079; within-scanner Siemens 0.807 → 0.587 (−0.220), Toshiba 0.784 → 0.674 (−0.111), Philips 0.676 → 0.700 (+0.024, n.s.); and the tax rises: cc_psz +0.031 → +0.046 (+0.015 [−0.002, +0.033]), p_max +0.026 → +0.050 (+0.024 [+0.004, +0.043]). Reading: the segmenter choice decides the deployment result. The official baseline segmenter (trained on this cohort) costs ~0.01 AUROC and leaves the confound unchanged; an off-the-shelf segmenter used as published fails on 6 % of scans, preferentially on PDAC in chest-heavy scans and on Siemens, which costs ~0.09 AUROC, hits scanners unequally and pushes the tax up. A deployment-ROI audit therefore has to report segmenter failures per scanner and per class, not just accuracy on the cases it could process.

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
