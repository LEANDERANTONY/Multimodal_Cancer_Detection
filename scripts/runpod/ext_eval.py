"""Per-case external-validation scores (MSD + NIH, never trained on) from an nnU-Net prediction folder.

For each case: source, y (lesion in label), Dice of the predicted mask, n_pred voxels, and the
detection scores p_max / p_sum / cc_psz from the saved softmax. Analysis (AUROC with CI on MSD,
NIH specificity at a threshold frozen on the Dutch CV scores) is done locally from the CSV.
Usage: python ext_eval.py <pred_dir> <label_dir> <cases.csv> <out.csv>
"""
import csv, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import nibabel as nib
from scipy import ndimage

PRED, LAB, CASES, OUT = sys.argv[1:5]


def one(row):
    c = row["case"]
    try:
        gt = np.asarray(nib.load(f"{LAB}/{c}.nii.gz").dataobj) == 1
        pr = np.asarray(nib.load(f"{PRED}/{c}.nii.gz").dataobj) == 1
        fg = np.load(f"{PRED}/{c}.npz")["probabilities"][1]
    except Exception as e:
        print(f"  skip {c}: {e}", flush=True); return None
    inter = np.logical_and(gt, pr).sum(); den = gt.sum() + pr.sum()
    dice = 2 * inter / den if den else float("nan")
    pmax = float(fg.max()); psum = float(fg.sum()); m = fg > 0.5
    if m.any():
        lab, n = ndimage.label(m); idx = range(1, n + 1)
        peak = np.array(ndimage.maximum(fg, lab, idx), float); size = np.array(ndimage.sum(m, lab, idx), float)
        ccpsz = float((peak * size ** (1.0 / 15.0)).max())
    else:
        ccpsz = pmax
    return [c, row["source"], int(gt.any()), int(gt.sum()), int(pr.sum()), dice, pmax, psum, ccpsz]


rows = list(csv.DictReader(open(CASES)))
with ThreadPoolExecutor(8) as ex:
    res = [r for r in ex.map(one, rows) if r]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "source", "y", "n_ref", "n_pred", "dice", "p_max", "p_sum", "cc_psz"]); w.writerows(res)
for src in ("MSD", "NIH"):
    r = [x for x in res if x[1] == src]; pos = [x for x in r if x[2]]
    d = np.nanmean([x[5] for x in pos]) if pos else float("nan")
    print(f"  {src}: n={len(r)} pos={len(pos)} posDice={d:.3f} neg_with_pred={sum(1 for x in r if not x[2] and x[4] > 0)}/{len(r)-len(pos)}", flush=True)
print(f"wrote {OUT} ({len(res)} cases)")
