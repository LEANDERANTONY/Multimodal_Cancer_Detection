# Dead ends: Multimodal Pancreatic Cancer Detection

Read before any experiment; add a row the moment something fails or is ruled out. These are the paper's
negative results and the answers to "did you try X?". Details are in `DEVLOG.md` (by phase), the ADRs, and
`docs/deployment_and_mitigation_design.md`; thesis-era rows come from the local preprocessing audit.

## Methods and data

| Date | Idea | What we ran | Result (evidence) | Why it failed / lesson | Revisit only if |
|---|---|---|---|---|---|
| 2025 | Histogram matching, then global standardisation, to remove the pixel-statistic shortcut (thesis) | notebook Cell 1.10, two-source TCIA set | the source probe stayed high | air and background dominated the statistics | never; nnU-Net foreground normalisation replaces it |
| 2025 | `extreme_standardize`: force each image's body pixels to mean 128 / std 40 (thesis) | same set, logistic probe on pixel mean | probe AUC 0.866 → 0.569, but k=2 clustering still split by source and Grad-CAM still showed the table | the detector and the fix target the same statistic (circular); perceptual remapping destroys HU fidelity (contraindicated in the 2026 Khan review) | never; detect bias in feature space instead |
| 2025 | Remove the aberrant cancer "cluster 0" found by embedding K-means (thesis) | k=6 on ResNet embeddings | dropped 598 slices = 6 whole cancer patients | it was a series-selection artefact: "largest CT series" grabbed Varian RT-aligned recons for some patients | never; select the diagnostic series from metadata |
| 2025 | Body-ratio [0.20, 0.50] as an "abdominal slice" filter (thesis) | thesis pipeline | worked, but depends on field of view | a geometric proxy for anatomy | never; crop to the pancreas instead |
| 2025 | ResNet50 on the two-source thesis set as a cancer classifier | patient-level split | slice AUC 0.9999, patient AUC 1.0 | source = label, so every split is confounded the same way; this is domain leakage, not performance | never; generalisation needs external / leave-one-manufacturer-out data |
| 2026-07 | Use PANORAMA's `level` column, or a metadata probe, as a predictor | metadata logistic probe | AUC 0.90, permutation importance level 0.327 ≫ scanner 0.016 | `level` encodes how the diagnosis was confirmed: near-label leakage | never; exclude `level` everywhere |
| 2026-07 | Leave-one-site-out generalisation | read `clinical_information.xlsx` | no institution column | Radboud vs Groningen can't be separated | the organisers release site labels; until then leave-one-manufacturer-out |
| 2026-07 | NIH subset as a control arm or an AUROC set | audit | 80 cases, 0 % PDAC | no positives, and a standalone control arm reintroduces a dataset-of-origin confound | never; NIH is for specificity only, never pooled with MSD |
| 2026-07 | ROI from the lesion mask, or a "manual masks only" arm | audit | 479 of 482 manual lesion masks are PDAC; manual-only leaves ~3 controls | the lesion mask and its provenance leak the label | never; use the label-blind pancreas (4) + duct (5) masks |
| 2026-07 | GRL (gradient reversal) as the headline mitigation | literature (DomainBed, CVPR 2022 failure-modes study) | tuned ERM matches or beats the domain-generalisation family | a 2015 method that doesn't beat a well-tuned baseline | never as the headline; GRL is one arm of a panel against tuned ERM (ADR-003) |
| 2026-07 | Image-level GAN harmonisation; CLAHE-style perceptual normalisation | literature (Mali 2025; Khan 2026) | GAN harmonisation lowered AUC (−2.6 %) and stability; perceptual methods distort HU | HU-preserving, feature-level methods (ComBat) do better | never |
| 2026-07 | Multi-window 3-channel float input as the default | not run | no PANORAMA pipeline windows manually | no precedent to cite | only as an ablation against nnU-Net normalisation |
| 2026-09 | TotalSegmentator pancreas masks for the training crop | fast 3 mm and full resolution on PANORAMA scans | empty pancreas on ~5–14 % (fast), some still empty at full resolution | the provided masks cover every case | never for training crops; it stays only as a deployment-ROI arm |
| 2026-09/10 | Loose crop (150×100×40 mm, Dataset700) as the primary model | 5-fold CV + LOMO + external, nnU-Net 3d_fullres 250 ep | CV Dice ~0.33, cc_psz ~0.70; external MSD AUROC 0.713 (tight 0.505 / 0.787 / 0.823) | too much non-pancreas context; the field-standard tight margin wins clearly | never as primary; kept as the loose-tax robustness arm (+0.021 [−0.002, +0.045]) (ADR-004) |
| 2026-10-05 | p_sum (summed foreground probability) as the detection score | tight CV, external, deployment | weakest score (mixed AUROC 0.731 vs cc_psz 0.793); the only one losing accuracy under predicted crops (−0.016 [−0.027, −0.005]) | sums scale with crop and lesion volume, not just confidence | only as a secondary score |
| 2026-10-10/13 | TotalSegmentator as the deployment stage-1 segmenter | all 1,964 Dutch scans, full resolution, as published | 118 empty masks (6.0 %; Siemens 7.9 %, 24 % of Siemens PDAC; 0 of 274 external); deployed with failures called negative, AUROC 0.787 → 0.700 (−0.087 [−0.111, −0.063]), Siemens 0.807 → 0.587, tax +0.031 → +0.046 | failures are uneven across scanners and 3× more common on PDAC (11.7 % vs 3.5 %), concentrated in chest-plus-upper-abdomen scans whose pancreas sits near the scan's lower edge (PDAC scans ending 50–150 mm below the pancreas fail 44–47 %); matched model held-out AUROC 0.83–0.86 | cause: the pancreas-only `roi_subset` cropping pass picks a tiny, misplaced region. On the 118 failures, adding neighbouring organs recovers 118, the full task 117, the documented `robust_crop` fix only 63 (PDAC 29 / 67) | reported as a deployment arm, not recommended as configured. Fixed by requesting neighbouring organs: 0 / 1964 failures, deployed AUROC 0.782 (oracle 0.787), tax +0.034. `robust_crop` alone: 66 failures, AUROC 0.719. The official PANORAMA baseline segmenter had 0 failures |
| 2026-10-09/10 | Zenodo as the data source for a deployment pod | 3 pods, resumable downloads | Zenodo throttled globally to 7–180 KB/s; pods stopped with no compute done | an external server you don't control sets the pace; renting GPUs while waiting burns money | stage the data once to the volume (`panorama_raw_dutch.tar`), then rent compute |
| 2026-10-10 | Run the largest Dutch scans (>150 M voxels) locally | stage 1 on a 1024×1024×331 scan | NIfTI export needs ~20 GB RAM; the watchdog stopped the job within a minute | doesn't fit next to the user's normal work (≤14 GB for jobs) | the user closes other apps, or on a pod; these scans are 45 % PDAC and must not be dropped |

## Not chosen (decided without running)

- **Our own pancreas segmenter for the training crop:** the field trains one only because it must deploy without masks. On the labelled dev set the provided, label-blind masks are public and reproducible. Deployment is tested separately with published segmenters (ADR-005).
- **H100 / RTX 5090 for nnU-Net:** nnU-Net is limited by the CPU dataloader, so the H100 costs about 4× for 2–3× the speed, and the 5090 has a virtualisation reset bug that is risky on an unattended rented pod.
- **Training straight off the RunPod Global volume:** random reads, per-operation fees and beta limits. Copy once to the pod's NVMe instead.

## Process mistakes

| Date | Mistake | Cost | Guard now in place |
|---|---|---|---|
| 2025 | Imputer and scaler fitted on the whole biomarker set before splitting | small leak (<2 % missing) | fit preprocessing inside the CV folds |
| 2025 | Bias check measured with the same statistic it fixed (circular) | an overstated mitigation claim in the thesis | the detector must be independent of the mitigation (feature-space probe) |
| 2026-09 | First crop pass zeroed the image origins | a re-run when building Dataset700 | origins restored in Dataset700; check the affine of every new crop set |
| 2026-09 | Torch tool environment at a long Windows path | silent corruption (260-char MAX_PATH) | deep-dependency environments live at short paths |
| 2026-09 | Confound tax first reported without a CI ("≈ +0.02") | claims had to be reworded later | every paper-facing metric gets a patient-level bootstrap CI |
| 2026-10 | Fold 2 (Philips held out) collapsed to all-background | one fold re-run on a rented pod | per-sample Dice rewards empty predictions on low-prevalence folds; check predictions are non-empty on the fold's own training scans |
| 2026-10 | Loose-LOMO detection AUROCs recorded as Dice in notes | wrong numbers in the docs until caught | read metrics from the committed summary CSVs, never from memory |
| 2026-10-06/08 | Long jobs launched in the app's terminal tabs | three runs killed by app restarts | detached `Start-Process` jobs with per-case checkpointing; every script resumable and refuses a second copy |
| 2026-10-09 | First deployment pod: `TOTALSEG_HOME_DIR`'s parent missing, so it crashed after the identity gate passed | a pod session; stopping the pod wiped its container disk | `mkdir -p` in the driver; outputs saved to the volume before any stop; assume a stopped pod keeps nothing |
| 2026-10-09 | Download loop deleted a partial file on an MD5 mismatch and restarted from zero | hours of Zenodo download | resume to the published byte size, then check the MD5 |
| 2026-10-09 | Driver review found a bare `wait` (would wait on the 30 h failsafe), a precedence bug in the stage chain, and a weight-download race | caught before launch | explicit `wait $PID`, grouped chains, weights fetched once before the workers start |
| 2026-10-10 | `psutil.BELOW_NORMAL_PRIORITY_CLASS` is Windows-only | the runner would crash on Linux pods | `getattr` fallback to nice 10 |
| 2026-10-11 | The Dutch gap script counted 19 cases outside the run list as segmenter failures | caught before the numbers went into the docs | a failure is a case stage 2 logged in `skipped.txt`, nothing else |
