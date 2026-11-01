# Head-to-foot orientation in PANORAMA (investigation, 2026-10-27 to 2026-11-02)

**Finding.** In 160 of the 1,964 Dutch PANORAMA development scans (8.1 %) the slice order contradicts the NIfTI header: a
reader that follows the header sees the body upside down. Labels and masks are fine, because they share the voxel grid.
All 160 are in one group of 245 released files. Rewriting only the header both cures and creates the TotalSegmentator
failures. The mismatch has no detectable link with cancer within that group. The 2.5D classifier is sensitive to the
direction, but correcting it leaves the confound tax unchanged.

**This replaces the first version of this note (2026-10-27)**, which claimed 155 scans, a "curation artefact", "four times
more common in PDAC" and a classifier "shortcut". See section 6 for what was wrong.

## 1. The files are PANORAMA's own, unmodified
- Images come from the four Zenodo batch zips, each MD5-verified and extracted with `tar` (no conversion;
  `data/raw/ct/panorama/INVENTORY.md`, `download.log`). No project tool writes into `data/raw/`.
- Labels are at `panorama_labels` commit `bf1d6ba` (2024-10-24).
- **No PANORAMA DICOMs exist here.** The Zenodo release is 2,238 `.nii.gz` files only. The DICOM folders under
  `data/raw/ct/` are the thesis collections (Pancreatic-CT-CBCT-SEG, Pancreas-CT), not PANORAMA. So the origin of the
  mismatch cannot be proven from the sources.

## 2. Which scans (`tools/orientation_adjudicate.py` -> `reports/deployment_roi/orientation_adjudicated.csv`)
- A lung rule (`tools/orientation_check.py`) gave a first call. It made errors: 3 false positives (bowel gas, one rotated
  scan) and 8 misses (no lung in the end slices).
- Calls were reviewed on mid-sagittal (and where needed coronal) contact sheets: all 245 files of the group, all 129 other
  scans the rule called inverted or unclear, and a random 80 of the other 1,590 (0 of 80 inverted; upper 95 % bound about
  4 %, and these had also passed the lung rule). **The review was by the AI assistant, not a radiologist.** The changed
  calls are listed in the script so they can be checked.
- Result: 160 inverted, 1,802 correct, 1 rotated by 90° (101104), 1 uncertain (100199). External MSD and NIH: 0 of 274 by
  the lung rule, not visually reviewed.

## 3. Where it sits
- The group is exact: `sform_code = 0` coincides with a gzip header written on Windows (OS byte 11) in 245 files. 160 of
  them are inverted (65 %), and 0 of the other 1,719.
- Inside the group, by manufacturer (inverted / correct): Siemens 99 / 17, Toshiba 43 / 1, Philips 18 / 43, other 0 / 23.
  That pattern, and the origin values (origin minus stack length matches the foot-end distribution of correct scans:
  Kolmogorov-Smirnov distance Siemens 0.54 -> 0.17, Toshiba 0.63 -> 0.34, Philips 1.00 -> 0.50), would fit a converter that
  kept the vendor's slice order and ignored its direction. This is strong circumstantial evidence, not proof.
- Not reported in the `panorama_labels` issue tracker or commit history (checked 2026-10-27); the organisers' forum thread
  was not accessible.

## 4. Questions and answers

### Q1. Does the header mismatch cause the TotalSegmentator failures? Yes (`tools/orientation_totalseg_test.py`)
Same call as stage 1; voxels never changed, only the header's cranio-caudal direction. 353 runs, no errors.

| Scans | Header | Empty pancreas | Dice (median) |
|---|---|---|---|
| Correct, as stored | matches | 5 / 1,802 (0.3 %) | 0.87 |
| Inverted, as stored | contradicts | 112 / 160 (70.0 % [62.3, 77.0]) | |
| Inverted, corrected | **rewritten to match** | **0 / 157** ([0, 2.3]) | 0.84 (IQR 0.78-0.88) |
| Correct, Windows group | **rewritten to contradict** | 51 / 78 (65.4 % [53.8, 75.8]) | 0.39 where not empty |
| Correct, other files | **rewritten to contradict** | 54 / 80 (67.5 % [56.1, 77.6]) | 0.39 where not empty |

- Re-saving unchanged reproduced the stage-1 outcome in 30 of 30 scans.
- All 112 failures recover after correction (median Dice 0.83, minimum 0.44).
- The two "broken" groups do not differ (P = 0.87), nor from the inverted scans as stored (P = 0.55, 0.77). So it is the
  header, not something else about that batch.
- **The 48 inverted scans that "did not fail" were harmed too.** In 9 tested, Dice was 0.155 as stored and 0.818 corrected
  (9 of 9 improve, P = 0.004).
- **Other causes exist:** the 5 failures on correct scans (100597, 100481, 101033, 101621, 100455) and the rotated scan stay
  empty after a flip. Unexplained, and few.
- **Remedies compared on inverted scans:** corrected header Dice 0.84; neighbouring organs, header left wrong, 0.68 (0.89 on
  correct scans); PANORAMA baseline 0.58 on both, so it does not depend on this.

### Q2. Where did it come from? Unknown
See section 3. One group of files, one likely mechanism, no way to confirm without the DICOMs or the organisers.

### Q3. How much does the 2.5D classifier depend on it? Measurably, but it does not change the results
`tools/orientation_flip_all.py`: all 1,939 evaluation scans reversed, scored by the held-out fold models, 3 seeds averaged.
- Paired change: correct non-PDAC 0.023 -> 0.232 (98.8 % up); correct PDAC 0.896 -> 0.958; inverted non-PDAC 0.620 -> 0.173
  (all down); inverted PDAC 0.896 -> 0.654.
- Score separating inverted from correct non-PDAC scans, within scanner x `level`: 0.725 [0.613, 0.878] as stored, **0.484
  [0.351, 0.654] with inverted scans upright**, so nothing is left once corrected. 3D model as stored: 0.529 [0.370, 0.653].
- Headline numbers: inverted upright, AUROC 0.928 -> 0.923 (difference [-0.011, -0.001]), tax +0.051 -> +0.050. Every scan
  upside down: 0.908, tax +0.059. Tax here uses seed-averaged scores, hence +0.051 against the panel's seed-level +0.056.
- Not settled: whether the model learned a cue (inverted scans are PDAC-rich in the data as a whole) or reversed anatomy is
  simply unfamiliar. Inference cannot separate these; retraining would.

### Q4. Why was it linked to cancer? It is the batch, not the disease
- Crude: 61.3 % vs 26.6 % PDAC (OR 4.36). This was the "four times" claim.
- The group is PDAC-rich: 60.0 % vs 25.1 % of other files.
- Inside the group: 61.3 % inverted vs 58.3 % correct (OR 1.13, P = 0.68). Mantel-Haenszel by batch 1.16 [0.68, 1.94]; by
  `level` 0.84 [0.54, 1.31]; by scanner and `level` 0.76 [0.44, 1.33]. Wide intervals, so a small link is not excluded.
- Failures still fall more on cancers (11.6 % vs 3.7 %), which matters for evaluation whatever the reason.

## 5. Recommendations
- Check header orientation against anatomy before running orientation-dependent tools, and correct the header if it
  contradicts. Requesting neighbouring organs prevents the empty masks but gives poorer masks on these scans.
- The affected case list is in `orientation_adjudicated.csv`. Reporting it to the organisers is the author's decision.

## 6. What the first version got wrong
- **155 / 7.9 %:** the lung rule's count. Reviewed: 160 / 8.1 %, all in one group.
- **"Curation artefact", "separate conversion run":** inference stated as fact. The labels are fine and the images are
  usable; only the header-to-slice-order relation is off.
- **"Four times more common in PDAC":** marginal, explained by the group.
- **"0.61 vs 0.02", AUROC 0.89:** compared against mostly `radiology` controls. Within scanner x `level` it is 0.725.
- **"Both detectors reach only 0.70 on inverted scans":** those streams are simply hard (non-PDAC lesions).
- Earlier still, "chest-heavy scans with the pancreas near the lower edge" came from geometry measured in header
  coordinates.
- `tools/orientation_analysis.py` and `tools/orientation_flip_test.py` (rule labels, 455 scans) are kept as the record but
  superseded by `orientation_adjudicate.py`, `orientation_flip_all.py` and `orientation_tests_summary.py`.
