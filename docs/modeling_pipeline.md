# CT modeling pipeline — how the PANORAMA nnU-Net models were built (reproducible)

_Durable record so the whole chain can be re-run / understood without chat history. Scripts in `scripts/runpod/`. Companion: `docs/preprocessing_audit.md` (rationale; local only, git-ignored), `docs/q1_readiness_and_gaps.md` (paper plan)._

## 0. Data provenance
- PANORAMA: 2238 studies / 2224 patients, local at `data/raw/ct/panorama/images` (182 GB) + masks `data/raw/ct/panorama_labels/{automatic,manual}_labels` + `clinical_information.xlsx`.
- Mask label legend: **1=PDAC lesion, 2=veins, 3=arteries, 4=pancreas parenchyma, 5=pancreatic duct, 6=common bile duct**.
- Cohorts (from `clinical_information.xlsx` scanner col / build manifest `source`): **Dutch=1964 (train/CV/LOMO), MSD=194, NIH=80** (the 274 MSD+NIH are held out as external test).

## 1. ROI-crop dataset build — `scripts/runpod/build_roi_dataset.py` (local, CPU)
Crop each case to the **bbox of pancreas(4)+duct(5)** (label-blind; never the lesion mask) + a fixed per-side margin, at native resolution; training label = lesion (mask==1). One flag sets the margin:
- **Loose = Dataset700**, margin `150 100 40` mm → median crop ~374x272x171 mm (near-full-FOV; pancreas is wide).
- **Tight = Dataset701**, margin `100 50 15` mm (field/PanDx standard) → median ~319x174x114 mm.
Run: `python scripts/runpod/build_roi_dataset.py --margin-mm 100 50 15 --out data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight`. Writes nnU-Net raw (imagesTr/labelsTr + dataset.json) + per-case `roi_build_qc.csv` (lesion containment). Tight containment: 667/676 PDAC fully contained, 10 mildly clipped (accepted; fixed margin keeps ROI label-independent). Machine-safe (streams one case, below-normal priority, resumable).
- **Local copies:** loose `data/processed/ct/nnunet_raw/Dataset700_PanoramaPDAC/` (1964 Tr + 274 Ts); tight `data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight/` (2238). No local tars (re-create before any upload; the volume keeps `Dataset701_tight.tar`). Trained models: `models/nnunet/{loose_cv,loose_lomo,tight_cv,tight_lomo}/`. Full map: `docs/data_layout.md`.

## 2. Staging to RunPod (Global volume `panaroma_roi`, fuse.geesefs at /workspace)
- **KEEP the dataset tar on the volume** (`/workspace/Dataset701_tight.tar`, 22.98 GB) — deleting it once cost an 8h re-upload. Storage is cheap (~$0.07/GB/mo).
- Per-run workflow: stage tar from volume → pod local `/root` (fast), train from local NVMe, results → volume. Never train off the S3-FUSE volume.
- **Large uploads (local→pod) must run in the USER's terminal**, not an agent background task (those are time-capped and get killed). Resumable uploader: `scripts/runpod/upload_resume.sh` (ssh-append loop, resumes from remote size). Invoke via `run_in_terminal` with the full git-bash path: `& "C:\Program Files\Git\bin\bash.exe" "/c/.../upload_resume.sh"` (plain `bash` is NOT on PowerShell PATH).

## 3. Preprocess + train — `scripts/runpod/run_train701.sh` (CV), `run_lomo701.sh` (LOMO)
Env: `nnUNet_raw=/root/nnunet_raw  nnUNet_preprocessed=/root/nnunet_prep  nnUNet_results=/workspace/nnunet_results_tight[_lomo]`; cap `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nnUNet_n_proc_DA=8` (host nproc misreports; real vCPU is lower).
Steps (both drivers): extract tar → **strip to 1964 Dutch** (keep only cases in `/root/dutch_cases.txt`, so the fingerprint matches across CV/LOMO) → **count-gate (abort if != 1964)** → `nnUNetv2_plan_and_preprocess -d 701 -c 3d_fullres --verify_dataset_integrity -np 8` → train → pod-side auto-stop.
- **CV:** nnU-Net auto-generates 5-fold splits (deterministic on sorted Dutch cases → matches Dataset700). Train: `nnUNetv2_train 701 3d_fullres {0..4} -tr nnUNetTrainer_250epochs --npz`.
- **LOMO:** after preprocess, overwrite `splits_final.json` with the manufacturer-holdout splits (`/workspace/splits_final_lomo.json`, filtered to present Dutch cases): **fold 0=Siemens, 1=Toshiba, 2=Philips** holdout. Train folds 0,1,2 same command.
- nnU-Net config: PlainConvUNet 3d_fullres, patch [48,160,256], CTNormalization (clip+z-score), ~3.3 h/fold on a 3090, ~5.5 GB VRAM (dataloader/CPU/RAM-bound → pick GPU by vCPU+RAM, not VRAM).

## 4. Detection metrics — `scripts/runpod/detection_candidate.py` (reads the `--npz` softmax)
Per-case score from the foreground softmax, 4 ways: `p_max`, `p_sum`, `cc_peak`, **`cc_psz`** (PanDx-style peak x size^(1/15)). Per-fold + pooled AUROC. Run: `python scripts/runpod/detection_candidate.py <results_base>`.

## 5. Feature-space confound probe — `scripts/runpod/feature_diag.py`
Extracts the trained encoder bottleneck (320-d, GAP of centre patch), patient-grouped linear/RF/MLP probes for scanner vs cancer. Saves `feature_diag_features.npz` (local copy: `reports/nnunet_summaries/`).

## 6. Results to date
- **Segmentation Dice (pos-case CV):** loose 0.33 → **tight 0.505 ± 0.02**.
- **Detection AUROC (best `cc_psz`):** loose 0.697 → **tight 0.787**; `p_max` loose 0.562 → tight 0.770. vs PanDx ceiling 0.926 (tuned two-stage ensemble; we are a single-stage reference).
- **Scanner metadata-only AUROC = 0.70** (confound in the DATA). **Feature probe:** scanner 0.59–0.61 (any probe) vs cancer 0.65–0.69 → the pipeline resists the confound (robust, not just weak).
- Loose LOMO pos-Dice: Siemens 0.365 / Toshiba 0.349 / Philips 0.287; loose LOMO detection AUROC (p_sum): 0.641 / 0.711 / 0.588.
- **Tight LOMO** (held-out manufacturer): pos-Dice Siemens 0.547 / Toshiba 0.450 / Philips 0.471 (tight-CV same scanner 0.508 / 0.483 / 0.516); detection cc_psz 0.811 [0.77,0.85] / 0.773 [0.73,0.82] / 0.706 [0.63,0.78] vs tight-CV within-scanner 0.807 / 0.784 / 0.676 → OOD ≈ in-distribution.
- **Tight confound battery** (`scripts/runpod/tight_battery.py`; per-case CSVs in `reports/nnunet_summaries/tight_battery/`): CV confound tax cc_psz **+0.031 [+0.012,+0.054]**, p_max +0.026 [+0.008,+0.047] (patient bootstrap) — small but non-zero; mixed (0.787) does not exceed the best within-scanner AUROC (Siemens 0.807). Feature probe (tight fold-0 encoder, n=750): scanner 0.618/0.644/0.622 (lin/RF/MLP; shuffle ~0.50) vs cancer 0.756/0.752/0.767.
- **External validation** (`scripts/runpod/run_external.sh`, 5-fold ensemble, TTA; per-case CSVs in `reports/nnunet_summaries/external/`): tight model on MSD (194, 98 PDAC) lesion Dice 0.555 (median 0.668), detection AUROC p_max 0.823 [0.764, 0.884], cc_psz 0.781 [0.714, 0.846]. MSD and NIH reported separately. Thresholds frozen on Dutch CV scores give low MSD sensitivity — likely because ensemble scores are smoother than single-model CV scores; per-fold check (`run_external_perfold.sh`) pending. Loose external pending.
- Tight-LOMO fold 2 first run collapsed to all-background at epoch 1 (per-sample Dice rewards empty predictions; fold 2 has the lowest positive rate, ~21%); re-run with the same config trained normally. Collapsed run archived as `fold_2_collapsed/`.

## 7. Volume cleanup policy
Keep on volume: **dataset tar(s)**, checkpoints, `summary.json`s. **Delete the big `--npz` softmax after extracting detection numbers** (regenerable from checkpoints). Pull `summary.json`s + feature npz to `reports/nnunet_summaries/` before deleting. Volume went 343 GB → ~6.5 GB this way (+ 22 GB tar kept = ~29 GB).
