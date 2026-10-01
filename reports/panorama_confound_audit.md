# PANORAMA Metadata Confound Audit

_Bias-aware pre-modeling audit. Measures how much the patient-level label (PDAC vs non-PDAC) can be predicted from **non-image metadata alone** (scanner, source/diagnostic-confirmation subset, demographics). No image models were trained; the dataset and source code were not modified._

**Source:** `data/raw/ct/panorama_labels/clinical_information.xlsx` (2238 rows, 8 columns).

## 1. Column overview & label distribution

| column | dtype | missing | missing_pct |
|---|---|---|---|
| PANORAMA_patient_id | int64 | 0 | 0.0 |
| PANORAMA_study_id | object | 0 | 0.0 |
| anonymized_study_date | datetime64[ns] | 194 | 8.7 |
| patient_age | object | 274 | 12.2 |
| patient_sex | object | 274 | 12.2 |
| scanner | object | 276 | 12.3 |
| label | object | 0 | 0.0 |
| level | object | 0 | 0.0 |

**Label distribution:**

| label | count | pct |
|---|---|---|
| non-PDAC | 1562.0 | 69.8 |
| PDAC | 676.0 | 30.2 |

Class imbalance ~70/30 in favour of non-PDAC (676 PDAC / 1562 non-PDAC).

### Missing-data handling

- `scanner`: 276 NaN + 1 literal `"0"` collapsed into a single **`Unknown`** category (277 rows). Unknown is kept as an explicit level, not dropped, so no rows are lost and the missingness itself is auditable.

- `patient_age`: stored as strings like `042Y`; parsed to numeric years (274 missing). In the logistic model, age is median-imputed **inside CV folds** (no leakage) and standardized.

- `patient_sex`: 274 NaN kept as an explicit `Unknown` category.

- `level`: complete (0 missing).

## 2. scanner x label

| scanner | PDAC | non-PDAC | total | pct_PDAC |
|---|---|---|---|---|
| SIEMENS | 170.0 | 768.0 | 938.0 | 18.1 |
| TOSHIBA | 134.0 | 517.0 | 651.0 | 20.6 |
| Philips | 241.0 | 78.0 | 319.0 | 75.5 |
| Unknown | 100.0 | 177.0 | 277.0 | 36.1 |
| GE MEDICAL SYSTEMS | 25.0 | 21.0 | 46.0 | 54.3 |
| Canon Medical Systems | 6.0 | 1.0 | 7.0 | 85.7 |

**Chi-square test of independence:** chi2 = 432.2, dof = 5, p = 3.47e-91. **Cramer's V = 0.437.**

Scanner is **strongly and highly-significantly associated with the label** (V ~ 0.44 is a large effect for a nominal association). The %PDAC spread across manufacturers is dramatic: SIEMENS 18.1% and TOSHIBA 20.6% (both *below* the 30.2% base rate) versus Philips **75.5%**, GE 54.3%, and Canon 85.7% (n=7, small). In other words, knowing only the scanner manufacturer already shifts the PDAC probability from ~18% to ~76%. This is exactly the kind of acquisition-signature shortcut a naive image model could latch onto, because scanner brand imprints reconstruction-kernel / noise / HU-calibration fingerprints into the pixels themselves.

## 3. level x label (source subset / diagnostic-confirmation method)

| level | PDAC | non-PDAC | total | pct_PDAC |
|---|---|---|---|---|
| radiology | 49.0 | 1163.0 | 1212.0 | 4.0 |
| histopathology | 262.0 | 91.0 | 353.0 | 74.2 |
| pathology | 124.0 | 86.0 | 210.0 | 59.0 |
| MSD_dataset | 98.0 | 96.0 | 194.0 | 50.5 |
| cytology | 143.0 | 40.0 | 183.0 | 78.1 |
| NIH_dataset | 0.0 | 80.0 | 80.0 | 0.0 |
| radiology / 3yFU | 0.0 | 6.0 | 6.0 | 0.0 |

**Chi-square:** chi2 = 1075.4, dof = 6, p = 4.34e-229. **Cramer's V = 0.691** (very large).

> **Important interpretation caveat.** The `level` column does **not** cleanly match the "two Dutch centres vs MSD vs NIH" framing in the brief. Its actual values are a *mix* of (a) external source subsets -- `MSD_dataset` (n=194, 50.5% PDAC) and `NIH_dataset` (n=80, **0% PDAC**, all controls) -- and (b) the **diagnostic-confirmation method** for the main (Dutch) cohort: `radiology` (n=1212, only 4.0% PDAC), `cytology` (78.1%), `histopathology` (74.2%), `pathology` (59.0%), and a tiny `radiology / 3yFU` (n=6, 0%).

PDAC prevalence differs **enormously** across these groups (0% to 78%). But most of this is a **label-definition artifact, not a physical acquisition confound**: PDAC cases are confirmed by pathology/cytology, whereas controls are typically confirmed radiologically (or by follow-up). So `level` is partly a proxy *for the label itself*. It should be treated as **near-leakage** and must **never** be used as (or correlated with) a model feature. Its value here is descriptive: it documents that the source subsets (MSD vs NIH especially) carry wildly different PDAC prevalence, which will bias any analysis that pools them without stratification.

## 4. Demographics x label

**Sex:**

| sex | PDAC | non-PDAC | total | pct_PDAC |
|---|---|---|---|---|
| F | 279.0 | 621.0 | 900.0 | 31.0 |
| M | 299.0 | 765.0 | 1064.0 | 28.1 |
| Unknown | 98.0 | 176.0 | 274.0 | 35.8 |

chi2 = 6.52, dof = 2, p = 0.038, Cramer's V = 0.045 -- a **negligible** association (%PDAC 31.0% F vs 28.1% M). Sex carries essentially no label information.

**Age:** PDAC mean 68.1 +/- 9.9 yr (n=578) vs non-PDAC 63.5 +/- 14.8 yr (n=1386). Welch t = 8.05, p = 1.58e-15 (Mann-Whitney p = 3.15e-09). PDAC patients are ~4.6 years older on average -- statistically significant given the sample size, but a **modest, clinically-expected** effect with heavy distribution overlap. Age is a weak-to-moderate signal, not a strong shortcut.

## 5. Metadata-only shortcut probe (the key number)

Logistic regression predicting `label` from **non-image metadata only** (one-hot scanner incl. Unknown, one-hot level, one-hot sex incl. Unknown, standardized age), stratified 5-fold CV.

- **Mean ROC-AUC = 0.901** (across-fold SD = 0.020; folds = [0.932, 0.882, 0.883, 0.905, 0.902]).

- **Out-of-fold pooled AUC = 0.900, 95% bootstrap CI [0.887, 0.912]** (2000 resamples).


**Permutation importance (mean AUC drop when the raw column is shuffled):**

| feature | perm_importance_AUC_drop | std |
|---|---|---|
| level | 0.3 | 0.0 |
| sex_clean | 0.0 | 0.0 |
| scanner_clean | 0.0 | 0.0 |
| age_num | 0.0 | 0.0 |

**Top logistic coefficients (one-hot / standardized):**

| feature | coef |
|---|---|
| level_radiology | -2.4 |
| level_NIH_dataset | -2.1 |
| level_cytology | 1.7 |
| scanner_clean_Canon Medical Systems | 1.5 |
| level_MSD_dataset | 1.5 |
| level_histopathology | 1.4 |
| level_radiology / 3yFU | -0.9 |
| level_pathology | 0.8 |
| scanner_clean_SIEMENS | -0.7 |
| sex_clean_Unknown | -0.7 |
| scanner_clean_GE MEDICAL SYSTEMS | -0.5 |
| sex_clean_F | 0.4 |

### Interpretation

An AUC of **0.90** (CI [0.89, 0.91]) is *far* above the 0.5 no-information line: the patient label leaks massively from metadata alone. **However, the leakage is dominated by `level`** (permutation AUC-drop 0.33, an order of magnitude larger than any other feature), and `level` is partly the label in disguise (diagnostic-confirmation method). So the headline 0.90 is *inflated by a definitional confound* and should not be read as "acquisition metadata alone predicts PDAC at 0.90." The honest, physically-meaningful shortcut signal is the **scanner** channel, whose standalone association (Cramer's V = 0.44) is already large. Age and sex contribute almost nothing (perm importance <=0.017). Bottom line: there is a **real, strong acquisition/source confound** in PANORAMA; a CT image model that appears to "detect PDAC" could be substantially exploiting scanner and source-subset signatures rather than tumour biology.

## 6. Implications for modeling

- **`scanner` and `level` must never be model input features**, directly or via engineered proxies. `level` in particular is near-label leakage.

- **Stratify and report all CT results by scanner and by source subset.** A single pooled AUC hides that Philips is 75% PDAC while SIEMENS/TOSHIBA are ~19%.

- **Audit whether image-model errors correlate with scanner.** After training, cross-tab predictions x scanner and x level; if the model's confidence tracks manufacturer, it is riding the confound.

- **Prefer scanner-balanced or scanner-held-out evaluation** (e.g. leave-one-manufacturer-out) to test whether performance survives an acquisition-domain shift. This is the honest generalization test.

- **Feature-space debiasing** (adversarial de-correlation against scanner, or reweighting) is justified by these numbers -- but the target of debiasing is *scanner/source*, not `level` (which is simply removed).

- **Do not pool MSD/NIH with the Dutch cohort naively:** NIH is 0% PDAC and MSD is 50% PDAC, so subset membership alone shifts prevalence from 0% to 50%.

- **Keep the ~12% missing-scanner rows visible as `Unknown`** rather than dropping them; the Unknown group is itself 36% PDAC (above base rate), i.e. missingness is not random.


## 7. Scanner-only shortcut ceiling (added 2026-10-01)

Section 5's headline 0.90 is inflated by `level` (near-label leakage), so it overstates the *physically deployable* shortcut. The honest, narrower question: **how well does the scanner manufacturer alone -- no `level`, no demographics, no pixels -- predict PDAC?** That is the slice of "detection" an image model can obtain for free by reading acquisition signatures.

Predictor = P(PDAC | scanner), estimated empirically; out-of-fold rates are fit on train folds only (unseen scanner -> global train rate), pooled, with a 2000-sample bootstrap CI.

| estimator | AUROC | AP |
|---|---|---|
| scanner-only, out-of-fold (**report this**) | **0.707**  95% CI [0.682, 0.731] | 0.560 |
| scanner-only, in-sample | 0.713 | 0.543 |
| chance | 0.500 | 0.302 (base rate) |
| full metadata incl. `level` (section 5) | 0.90 | -- |

**Scanner brand alone detects PDAC at AUROC ~0.71 with zero pixels** -- AP rises from the 0.30 chance floor to 0.56. This is the floor any CT detection result must be read against: a model reporting, say, 0.85 is claiming only ~0.14 of AUROC above what scanner metadata gives for free, and only a confound-controlled (scanner-stratified or leave-one-manufacturer-out) evaluation can show that margin is tumour signal rather than acquisition signature.

**Segmentation cross-check (3D nnU-Net, Dataset700).** A leave-one-manufacturer-out run (train on the other manufacturers, test on the held-out one) gave positive-case Dice of 0.365 (Siemens), 0.349 (Toshiba), 0.287 (Philips) -- all inside the random 5-fold CV band (0.28--0.39, mean ~0.33); the between-manufacturer spread (0.078) is smaller than the ordinary fold-to-fold spread (0.11). So the confound is **not** in how the network delineates a lesion (boundary-drawing is manufacturer-invariant); it is in **case selection** -- exactly where the scanner-only 0.71 lives. Positive-case Dice conditions on a lesion being present and is blind to the detection shortcut by construction, which is why the two tests must be read together.

Reproduce: `python tools/scanner_only_shortcut.py`.


---

**Artifacts:** figure `figures/panorama_confound_audit.png` (%PDAC by scanner and by level); script `tools/scanner_only_shortcut.py` (section 7, scanner-only AUROC). This report: `reports/panorama_confound_audit.md`.

**Anomaly note:** the `level` column's semantics differ from the task brief (it is diagnostic-method + source tags, with no separate Radboud/Groningen centre split available), and one scanner value was the literal string `"0"` (folded into Unknown). There is no explicit institution/centre column in the file.
