# Inverted cranio-caudal orientation in PANORAMA (investigation, 2026-10-27)

**Finding.** 155 of the 1,964 Dutch PANORAMA development scans (7.9 %) are stored with the cranio-caudal axis inverted
relative to their NIfTI header: a reader that follows the header sees the body upside-down (lungs at the feet). The
external MSD and NIH scans are unaffected (0 / 274). The inversion is strongly associated with PDAC, explains nearly all
TotalSegmentator `roi_subset pancreas` failures, and the 2.5D classifier uses it as a shortcut.

Because a widely used public dataset should not carry an error of this size, each link in the chain was checked
separately.

## 1. The files are PANORAMA's own, unmodified
- Images come from the four Zenodo batch zips, each MD5-verified against the published checksum and extracted with
  `tar` (no conversion; `data/raw/ct/panorama/INVENTORY.md`, `download.log`).
- File modification times are those stored in the zips (15–18 April 2024 for all 155 inverted files; September 2024
  for a few others), so no file was rewritten locally. No project tool writes into `data/raw/`.
- Labels are at the latest `panorama_labels` commit (`bf1d6ba`, 2024-10-24). The labels share the image grid, so the
  provided masks lie on the anatomy as stored; only the header-to-anatomy relation is wrong.

## 2. Anatomy disagrees with the header (`tools/orientation_check.py` -> `reports/deployment_roi/orientation_check.csv`)
- Rule: after reorienting by the header to LPS (index increasing = superior), lung-density voxels (−950 to −600 HU) in a
  central window should be at the high-index end. Inverted scans: lung at the low-index end (median 29 % of voxels,
  minimum 3.6 %), none at the other end (maximum 7.6 %). 140 scans had no lung at either end and are left "unclear".
- **Visual check:** mid-sagittal views of 8 random inverted and 4 random correct scans, laid out by the header, show the
  pelvis and bowel at the top and the liver and heart at the bottom for every inverted scan, and the reverse for every
  correct one; anterior–posterior and left–right agree in all twelve, so only the cranio-caudal axis is affected.
- **Threshold robustness:** with lung cut-offs from 1 % to 10 % (ratio 2 to 10) the inverted count is 115–165, inverted
  scans are 59–66 % PDAC against 27–28 % otherwise, and they hold 84–111 of the 118 TotalSegmentator failures.

## 3. The inverted files were written differently (header fields)
- Every file declares the same orientation (qform: x −1, y −1, z +1). **152 of the 155 inverted files have
  `sform_code = 0` with an empty sform**, against 79 of 1,669 correctly oriented files; so the inverted files came from a
  different conversion run whose header did not encode the reversed slice order. There is no hidden correct matrix.
- The inversion follows PANORAMA's `level` column (how the diagnosis was confirmed, a data-collection-stream marker
  excluded from all models): inverted in 97 / 210 "pathology", 51 / 183 "cytology" and 5 / 6 "radiology / 3yFU" scans,
  but 0 / 353 "histopathology" and 2 / 1,212 "radiology" scans. The inversion is therefore tied to one part of the data
  curation, which is PDAC-rich; this explains its association with the label.
- Not reported in the `panorama_labels` issue tracker or commit history (checked 2026-10-27); the organisers' forum
  thread on dataset fixes was not accessible.

## 4. Consequences (`tools/orientation_analysis.py` -> `reports/deployment_roi/orientation_summary.csv`)
- **Label association:** 60.0 % PDAC among inverted vs 26.8 % (OR 4.1, Fisher p = 2 × 10⁻¹⁶); within Siemens 60 % vs 13 %,
  Toshiba 61 % vs 18 %.
- **Segmenter failures:** `roi_subset pancreas` fails on 108 / 155 inverted (70 %) vs 5 / 1,669 correct (0.3 %) and
  5 / 140 unclear; `robust_crop` 59 / 155 vs 2 / 1,669; with neighbouring organs 0 everywhere. This replaces the earlier
  explanation ("chest-heavy scans with the pancreas near the lower edge"), whose geometry was measured in header
  coordinates and so described these scans upside-down.
- **Detectors (patient-disjoint evaluation set):** both reach AUROC 0.697 on the inverted scans. The 3D nnU-Net (trained
  with mirroring) barely separates inverted from correct non-PDAC scans by its score (AUROC 0.63); the 2.5D classifier
  does (0.89; median score 0.61 vs 0.02).
- **Causal flip test** (`tools/orientation_flip_test.py`, inference with the saved ERM models; `orientation_flip_test.csv`):
  correct non-PDAC scans flipped upside down rise from median 0.023 to 0.235; inverted non-PDAC scans flipped upright fall
  from 0.607 to 0.163 (inverted PDAC 0.882 -> 0.590). The 2.5D classifier reads the inverted orientation as PDAC: a
  label-correlated shortcut that nobody planted. Its pooled AUROC is not inflated by it (0.928 all vs 0.937 on correct
  scans), because the shortcut mainly raises false-positive scores on inverted non-PDAC scans.

## 5. Recommendations
- Report the affected case list to the PANORAMA organisers (author's decision).
- Any pipeline on PANORAMA should check orientation against anatomy (or train with cranio-caudal mirroring), and
  orientation-dependent tools such as TotalSegmentator should be run on corrected volumes.
