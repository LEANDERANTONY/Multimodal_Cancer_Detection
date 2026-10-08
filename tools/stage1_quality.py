"""Deployment-ROI: segmentation quality and crop geometry of a stage-1 arm vs the reference masks.

Per case: pancreas Dice (vs labels 4+5 and vs 1+4+5), fraction of the lesion inside the predicted mask,
centroid offset (mm), tight-crop box IoU with the oracle crop, and lesion containment in the predicted
crop. The crop rule is the oracle builder's (tools/build_roi_dataset.py: bbox + 100x50x15 mm per side,
clipped to the image), applied to the predicted mask instead of labels 4+5.

    python tools/stage1_quality.py <baseline_oof|totalseg> [external|dutch]
Writes reports/deployment_roi/stage1_quality_<arm>_<cohort>.csv and prints a summary by source/scanner.
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_roi_dataset import bbox_from_roi, expand_clip, mask_path  # noqa: E402

MARGIN_MM = (100.0, 50.0, 15.0)  # tight crop, per side (x, y, z)
MASKS = "data/processed/ct/stage1_masks"
OUT = "reports/deployment_roi"


def crop_box(roi, sp):
    margin_vox = [int(np.ceil(mm / s)) for mm, s in zip(MARGIN_MM, sp)]
    return expand_clip(bbox_from_roi(roi), margin_vox, roi.shape)


def box_iou(a, b):
    inter = np.prod([max(0, min(ah, bh) - max(al, bl)) for (al, ah), (bl, bh) in zip(a, b)])
    vol = lambda x: np.prod([hi - lo for lo, hi in x])
    return inter / (vol(a) + vol(b) - inter)


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def one_case(case, arm):
    ref_img = nib.load(mask_path(case))
    sp = np.array(ref_img.header.get_zooms()[:3], float)
    ref = np.asarray(ref_img.dataobj)
    pred = np.asarray(nib.load(f"{MASKS}/{arm}/{case}.nii.gz").dataobj) > 0
    assert pred.shape == ref.shape, (case, pred.shape, ref.shape)
    panc, les = np.isin(ref, (4, 5)), ref == 1
    row = {"case": case, "pred_vox": int(pred.sum()), "empty": not pred.any(),
           "dice_panc": dice(pred, panc), "dice_panc_lesion": dice(pred, panc | les),
           "lesion_vox": int(les.sum())}
    row["lesion_in_mask"] = (pred & les).sum() / les.sum() if les.any() else np.nan
    if pred.any():
        row["centroid_offset_mm"] = float(np.linalg.norm(
            (np.argwhere(pred).mean(0) - np.argwhere(panc).mean(0)) * sp))
        pb, ob = crop_box(pred, sp), crop_box(panc, sp)
        row["crop_iou"] = box_iou(pb, ob)
        if les.any():
            row["lesion_in_crop"] = les[tuple(slice(lo, hi) for lo, hi in pb)].sum() / les.sum()
    return row


def main():
    arm = sys.argv[1]
    cohort = sys.argv[2] if len(sys.argv) > 2 else "external"
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    ext = pd.read_csv("reports/nnunet_summaries/external/external_cases.csv")
    if cohort == "external":
        meta = ext.rename(columns={"source": "group"})[["case", "group", "y"]]
    else:
        meta = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")
        meta = meta.rename(columns={"scanner": "group"})[["case", "group", "y"]]
    done = {f[:12] for f in os.listdir(f"{MASKS}/{arm}") if f.endswith(".nii.gz") and not f.startswith("_")}
    meta = meta[meta.case.isin(done)]
    rows = []
    for i, c in enumerate(meta.case, 1):
        rows.append(one_case(c, arm))
        if i % 50 == 0:
            print(f"  {i}/{len(meta)}", flush=True)
    df = meta.merge(pd.DataFrame(rows), on="case")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/stage1_quality_{arm}_{cohort}.csv", index=False)
    cols = ["dice_panc", "dice_panc_lesion", "centroid_offset_mm", "crop_iou"]
    summ = df.groupby("group")[cols].median().round(3)
    summ["n"] = df.groupby("group").size()
    summ["empty"] = df.groupby("group").empty.sum()
    pos = df[df.lesion_vox > 0]
    summ["lesion_in_crop_mean"] = pos.groupby("group").lesion_in_crop.mean().round(3)
    summ["lesion_fully_in_crop"] = pos.groupby("group").lesion_in_crop.apply(lambda s: (s >= 0.999).mean()).round(3)
    print(f"{arm} {cohort}: n={len(df)} (medians; lesion columns over cases with a lesion)")
    print(summ.to_string())


if __name__ == "__main__":
    main()
