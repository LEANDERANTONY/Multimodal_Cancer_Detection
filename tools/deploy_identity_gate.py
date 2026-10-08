"""Identity gate before the Dutch deployment run: the pod's data + code must reproduce our oracle pipeline.

    python tools/deploy_identity_gate.py masks <cases.txt>        # write oracle_rebuild masks (labels 4+5)
    python tools/deploy_identity_gate.py check <cases.txt> <Dataset701 imagesTr dir>

`masks` writes data/processed/ct/stage1_masks/oracle_rebuild/<case>.nii.gz from the provided pancreas+duct
labels, so `tools/deploy_infer_oof.py oracle_rebuild` cuts crops through the SAME code path as the predicted
arms. `check` then requires, for every sample case:
  1. the rebuilt crop is voxel-identical to the training crop in Dataset701 (and the affine matches);
  2. the out-of-fold scores reproduce tight_battery/cv_scores.csv (|diff| <= 1e-3 for p_max and cc_psz,
     <= 1e-2 relative for p_sum).
Exit code 0 = PASS, 1 = FAIL (the driver stops the pod on FAIL).
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/tools")
from build_roi_dataset import mask_path  # noqa: E402


def write_masks(cases):
    out = f"{ROOT}/data/processed/ct/stage1_masks/oracle_rebuild"
    os.makedirs(out, exist_ok=True)
    for c in cases:
        ref = nib.load(f"{ROOT}/{mask_path(c)}")
        m = np.isin(np.asarray(ref.dataobj), (4, 5)).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, ref.affine, ref.header), f"{out}/{c}.nii.gz")
    print(f"wrote {len(cases)} oracle_rebuild masks", flush=True)


def check(cases, ds701):
    ok = True
    crops = f"{ROOT}/data/processed/ct/deploy_crops/oracle_rebuild_dutch/imagesTs"
    for c in cases:
        a, b = nib.load(f"{crops}/{c}_0000.nii.gz"), nib.load(f"{ds701}/{c}_0000.nii.gz")
        same = a.shape == b.shape and np.array_equal(np.asarray(a.dataobj), np.asarray(b.dataobj))
        aff = np.allclose(a.affine, b.affine, atol=1e-4)
        if not (same and aff):
            ok = False
            print(f"  CROP MISMATCH {c}: shape {a.shape} vs {b.shape}, voxels_equal={same}, affine_equal={aff}")
    print(f"crop identity: {'PASS' if ok else 'FAIL'} ({len(cases)} cases)", flush=True)

    new = pd.read_csv(f"{ROOT}/reports/deployment_roi/detect_oracle_rebuild_dutch.csv").set_index("case")
    ref = pd.read_csv(f"{ROOT}/reports/nnunet_summaries/tight_battery/cv_scores.csv").set_index("case").loc[new.index]
    dmax = (new.p_max - ref.p_max).abs().max()
    dcc = (new.cc_psz - ref.cc_psz).abs().max()
    dsum = ((new.p_sum - ref.p_sum).abs() / ref.p_sum.abs().clip(lower=1)).max()
    sok = dmax <= 1e-3 and dcc <= 1e-3 and dsum <= 1e-2 and (new.fold == ref.fold).all()
    print(f"score identity: {'PASS' if sok else 'FAIL'} (max |dp_max| {dmax:.2e}, |dcc_psz| {dcc:.2e}, "
          f"rel |dp_sum| {dsum:.2e}, n={len(new)})", flush=True)
    return ok and sok


def main():
    mode, cases = sys.argv[1], open(sys.argv[2]).read().split()
    if mode == "masks":
        write_masks(cases)
    else:
        sys.exit(0 if check(cases, sys.argv[3]) else 1)


if __name__ == "__main__":
    main()
