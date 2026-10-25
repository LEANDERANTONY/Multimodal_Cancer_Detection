# Q1 Readiness & Gap Analysis — Paper #1 (PDAC CT confound)

_Living strategy note. Target: Q1 (MedIA / npj Digital Medicine / Radiology:AI), Q2 fallback (MIDL / specialised imaging journal). Written 2026-10-02; updated 2026-10-26: every experiment below is done, the write-up is next._

## 1. Findings so far (assets)
- **Measured scanner confound in PANORAMA** (flagship public PDAC cohort): scanner↔label Cramer's V 0.44, chi2 432, p 3e-91; **scanner-alone AUROC 0.70** (manufacturer-only, patient-grouped) — reproducible (`tools/scanner_only_shortcut.py`).
- **`level` near-label-leakage** in a public challenge dataset (full-metadata AUC 0.90 dominated by `level`) — a concrete community gotcha.
- **3D nnU-Net reference** (loose ROI, CV 5-fold + LOMO): pos-case Dice ~0.33; detection AUROC ~0.6–0.68 within-scanner; in-distribution confound tax +0.021 [−0.002, +0.045] (CI includes 0); generalises across manufacturers (CV ≈ LOMO within-scanner).
- **Feature-space probe**: the trained model's bottleneck features encode scanner only weakly (0.586 linear / 0.602 RF / 0.614 MLP; shuffle 0.513) but encode cancer more (0.650/0.615/0.691) → **pipeline resists the confound at the representation level**.
- **Thesis 2D model** (positive control): rode a dataset-of-origin confound (cancer=Pancreatic-CT-CBCT-SEG vs control=NIH Pancreas-CT) to **0.9999** — confound *was* exploited.
- **Tight-crop (100x50x15mm), 2026-10-05:** CV pos-Dice 0.505, detection cc_psz 0.787. **Tight LOMO** pos-Dice 0.547/0.450/0.471 (Siemens/Toshiba/Philips held out), detection 0.811/0.773/0.706 ≈ in-distribution within-scanner → no manufacturer-shift penalty. **Tight confound tax +0.031 [+0.010,+0.053]** (patient bootstrap) — small but CI excludes 0; mixed never exceeds best within-scanner. Feature probe: scanner 0.62–0.64 vs cancer 0.75–0.77.
- **Deployment-ROI (2026-10-19):** with a segmenter that never fails, a deployed two-stage pipeline matches the oracle crop (0.782 vs 0.787) and the tax stays ~+0.03; TotalSegmentator used as documented (`roi_subset pancreas`) fails on 6 % of scans (PDAC- and Siemens-heavy), costing 0.087 AUROC and raising the tax to +0.046; the documented `robust_crop` fix recovers only half; requesting neighbouring organs removes every failure.
- **Mitigation panel (2026-10-19/25):** 2.5D ERM CV 0.918, tax +0.057, external 0.884; GBS / DFR / GRL lower no tax significantly and all cost CV AUROC (DeLong p <= 0.002); MAE raises the tax (+0.026 [+0.012, +0.041]); planted-shortcut control: the same methods remove 64-91 % of a planted shortcut (ERM drop 0.149, GRL 0.014).
- **Statistics (2026-10-25):** 2.5D vs 3D +0.138 in-distribution (p<0.001), tie on MSD (0.827 vs 0.823); 2.5D ECE 0.039; Philips and age >= 70 weakest; no mask-provenance leakage.
- **Biomarker cohort (2026-10-25):** site ~ diagnosis V 0.42, site-only AUROC 0.68, panel 0.867 with site tax +0.062 [+0.027, +0.098]: not a clean control.
- **Fusion (2026-10-25):** unpaired cohorts; combined AUROC 0.917 at independence -> 0.884 at rho 0.75; pseudo-paired 0.909 = independence case; triage avoids ~49 % CTs at 1 % prevalence.
- **Thesis-cohort control (2026-10-25/26):** PANORAMA models 0.68-0.98 on the thesis set; our recipe retrained on it 0.996 vs 0.745 on clinical data at matched size; crop statistics separate the collections (0.872).
- **Headline wording update:** "present, only weakly exploited (bounded ≤~0.05 AUROC), no manufacturer-shift penalty" — not "not exploited". Loose tax +0.021 [−0.002, +0.045]; both models' upper bounds < 0.055.

## 2. The reframed thesis: **confound present != confound exploited**
We set out to show "CT PDAC detection is a confounded shortcut." On the **thesis** dataset that holds (0.9999 from source leakage). On **PANORAMA** the opposite largely held: the confound is strongly *in the data* (0.70 from metadata) yet the well-built nnU-Net **exploits it only weakly** — a small, bounded tax (tight +0.031 [+0.010, +0.053]), no manufacturer-shift penalty, and external-hospital performance at or above in-distribution. So the honest, sharper headline is: *the presence of a confound in the data does not mean the model rides it; pipeline design (HU-preserving normalisation + ROI crop + segmentation objective) is the mediator — and we give a protocol to measure the difference.*

## 3. Novelty positioning (vs literature)
Shortcut **diagnosis** itself is crowded — do NOT claim it as novel:
- Ong Ly et al., npj Digital Medicine 2024 (PEst; generalisation estimate w/o external data) — https://www.nature.com/articles/s41746-024-01118-4
- Boland et al. 2024 — locate shortcuts in network *layers* (Prediction Depth). **Closest neighbour to our feature-probe.**
- MICCAI 2024 "Shortcut Learning in Medical Image Segmentation"; HSIC dependence benchmarking (MLMI 2024); survey arXiv 2412.05152.

**Our defensible novelty (lead with these; superseded in detail by `docs/literature_novelty_check.md`, 2026-10-25):**
1. **"Present, only weakly exploited" on the flagship PDAC cohort** — measured confound (0.70) barely propagated (tax bounded ≈0.05 AUROC with CI, weak feature encoding, no LOMO or external penalty).
2. **The `level` label-leakage finding** in a public challenge dataset.
3. **Deployment-ROI confound analysis** — oracle-mask vs predicted-segmenter ROI (see §6). Unclaimed in the literature.
4. **Two independent confounds, one protocol** — thesis dataset-of-origin (positive control) vs PANORAMA scanner (test case).
5. **Mitigation panel on a *real* clinical confound** — result: tuned ERM not beaten. "ERM is hard to beat" is known (MEDFAIR, Yang et al. Nat Med 2024), so the novelty is the planted-shortcut positive control and MAE raising the tax, not the null result itself.

## 4. Gap list for Q1
| Gap | Status | Weight |
|---|---|---|
| Mitigation panel (tuned ERM / DFR / GRL / SSL), local 2.5D | **DONE 2026-10-25**: ERM not beaten; planted-shortcut control validates the methods; MAE raises the tax (`reports/panel/`, `figures/panorama/mitigation_panel.png`) | **critical** |
| Deployment-ROI experiment (segmenter vs oracle, §6) | **DONE 2026-10-19.** External: no gap. Dose-response: flat <= 20 mm. Dutch 1964, four stage-1 arms (failures called negative): official baseline segmenter AUROC 0.774 (oracle 0.787), tax +0.036; TotalSegmentator `roi_subset pancreas` 118 empty masks, 0.700, +0.046; + `robust_crop` 66 empty, 0.719, +0.046; + neighbouring organs 0 empty, 0.782, +0.034. Cost and confound increase come from segmenter failures (PDAC- and Siemens-heavy) caused by the pancreas-only cropping pass. Jitter fine-tune not triggered. Details: `deployment_and_mitigation_design.md` | **high / novel** |
| External validation (held-out MSD + NIH; §5) | DONE: tight MSD AUROC 0.82, Dice 0.555; loose 0.71, Dice 0.35; p_max operating points transfer per fold, cc_psz thresholds shift | **high** |
| Tight-crop ablation (loose vs tight) | DONE (tight better on Dice + detection); loose tax CI done (+0.021 [−0.002, +0.045]) | medium |
| Biomarker + fusion | **DONE 2026-10-25**: biomarker site confound (tax +0.062); fusion as a dependence-sensitivity simulation + pseudo-pairing demonstration (`reports/biomarker_confound.csv`, `reports/fusion/`) | medium |
| Nonlinear feature-probe | DONE | — |
| DeLong CIs, calibration (ECE/Brier), subgroup-by-scanner/provenance | DONE 2026-10-25 (`reports/final_stats/`, `reports/panel/panel_stats.csv`) | needed |

## 5. External validation plan
**Primary — the held-out MSD + NIH set (already carved out, never trained on; 274 cases = `imagesTs` in Dataset700).**
- **MSD** (n=194, ~50% PDAC): different institution, both classes → clean external AUROC/AP.
- **NIH** (n=80, 0% PDAC): specificity / false-positive-rate only (no positives → no AUROC).
- These have provided pancreas masks (part of PANORAMA auto-labels) → can crop with oracle ROI first, then segmenter ROI (§6).
- Because the model never trained on these, strong generalisation evidence; if performance holds, it cannot be the *PANORAMA* scanner shortcut.
- **Report MSD and NIH separately** (MSD: AUROC/AP/Dice; NIH: specificity / FP rate). A pooled MSD+NIH AUROC is itself confounded by dataset of origin (NIH all negative, MSD mostly positive) — the exact trap the thesis model fell into.
- Data: loose Ts = `Dataset700/imagesTs` (local only); tight = the 274 non-Dutch cases inside Dataset701 `imagesTr` (local + volume tar). Checkpoints local in `models/nnunet/`.

**Secondary — the thesis TCIA cohort (CBCT-SEG cancer [n~34–40] + NIH Pancreas-CT control [n~82]) as a *confound control*, not a clean performance test.**
- Caveats: (a) the thesis set has its OWN dataset-of-origin confound (cancer=CBCT-SEG vs control=NIH); (b) each CBCT-SEG patient has one native Philips / GE CT (used) besides the Varian cone-beam and MIM-aligned copies (skipped; the thesis's "largest series" rule picked those); (c) the NIH controls are healthy pancreases (PANORAMA's 80 NIH cases), not PDAC.
- **Result (2026-10-25/26):** PANORAMA models 0.682 (3D cc_psz) / 0.862 (p_max) / 0.975 (2.5D); our recipe retrained on the thesis set 0.996 vs 0.745 on clinical data at matched training size; crop statistics separate the collections at 0.872 (`reports/thesis_control/`).
- Its value: show the PANORAMA model does **not** reproduce the thesis 0.9999 (it cannot exploit a confound it never trained on) — a clean contrast demonstrating the confound is dataset-specific, not intrinsic to CT. Interpret as a control, not external accuracy.

## 6. Deployment ROI: the two-stage practice + the oracle-vs-segmenter experiment
**Field practice (PANDA Cao 2023; Chen Radiology 2022; PANORAMA baseline; PanDx): two-stage** — Stage 1 segments the pancreas on the raw scan, crop the ROI from that *prediction*, Stage 2 classifies the crop. Predicting on the whole scan is **not** standard (too much irrelevant FOV). So "segment -> ROI -> crop -> predict on crop" is correct; we do NOT predict directly on the full image.

**The gap / experiment:** we trained on *oracle* (provided) mask crops, unavailable clinically. Re-run detection + confound diagnostics using a **predicted-segmenter ROI** (TotalSegmentator and the official baseline stage 1, out-of-fold — both, as published; see the design doc), and compare to oracle ROI:
- detection AUROC: oracle vs segmenter (the deployment gap);
- confound tax + feature-probe: does the robustness survive, or does the **segmentation stage re-inject a scanner confound** (plausible if Stage-1 accuracy is scanner-dependent)?
This answers clinical validity AND is the novelty differentiator. Run for BOTH the loose and tight models.

## 7. Experiment matrix (run for BOTH models: loose Dataset700 + tight Dataset701)
| Test | Loose model | Tight model |
|---|---|---|
| CV Dice + detection AUROC | done | done |
| LOMO Dice + per-scanner detection | done | done |
| Confound tax with CI | done (+0.021 [−0.002,+0.045]) | done (+0.031 [+0.010,+0.053]) |
| External MSD/NIH (oracle ROI) | done (MSD p_max 0.713, Dice 0.349) | done (MSD p_max 0.823, Dice 0.555; per-fold thresholds checked) |
| Deployment segmenter ROI (vs oracle) | — | done: external no gap; Dutch four arms (best: TotalSegmentator + neighbours 0.782 vs oracle 0.787) |
| Feature-space scanner probe | done (0.59–0.61) | done (0.62–0.64) |
| Mitigation panel (2.5D, local) | — | done (separate 2.5D arm; `reports/panel/`) |
All external/segmenter/probe tests are **inference-only (cheap)** — run for both once a pod is free.

## 7b. Borrowed from Kaggle grandmasters (`D:/Documents/Projects/Kaggle/Grandmasters/`, 2026-10-05)
- **ren4yu (medical imaging specialist):** localise → crop → classify is his standard two-stage pipeline (= our deployment-ROI setup); get geometry exactly right first (cf. the zeroed origin we found in the first loose crop pass); **2.5D with a pretrained 2D backbone + depth pooling beat true 3D** in several competitions → backbone choice for the local mitigation panel; attention pooling with an auxiliary loss.
- **sersasj (3D imaging, Biohub solo 1st):** **MAE self-supervised pretraining for robustness to an unseen domain** → a concrete SSL arm for the mitigation panel (unseen scanner = unseen domain); **train on deliberately corrupted inputs** → for deployment-ROI, train stage 2 on jittered / predicted-mask crops so it tolerates segmenter error (candidate mitigation of the deployment gap); model soups when epoch selection is unreliable.
- **samson8 (synthetic data):** synthetic data matched to the real distribution, **always mixed with real data, and all validation on real data only** → rule for the fusion arm: synthetic samples may augment training, every reported number comes from real patients.

## 8. Sequence
1. Finish tight-crop CV (running) -> loose-vs-tight ablation.
2. External MSD/NIH validation (oracle ROI) for both models.
3. ~~Deployment segmenter-ROI experiment~~ DONE 2026-10-19 (tight model; four stage-1 arms).
4. ~~Mitigation panel~~ DONE 2026-10-25 (tuned ERM / GBS / DFR / GRL / MAE, planted-shortcut control).
5. ~~Biomarker + fusion arm; DeLong CIs + calibration + subgroup tables~~ DONE 2026-10-25.
6. ~~Thesis-cohort confound control~~ DONE 2026-10-25/26 (`reports/thesis_control/`): PANORAMA models 0.68-0.98 vs the thesis's 0.9999; retrain 0.996 vs 0.745 at matched size; source probe 0.872.
7. **NEXT:** write-up as an end-to-end audit (lead: measured PANORAMA confound + deployment-ROI; panel recast around the planted-shortcut control); target MedIA / npj-DM; re-run the risky literature searches before submission.
