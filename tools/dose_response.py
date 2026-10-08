"""Deployment-ROI dose-response: tight-model detection on MSD as the oracle crop is degraded on purpose.

Conditions (docs/deployment_and_mitigation_design.md): the oracle tight crop (bbox of labels 4+5 +
100x50x15 mm per side) shifted by a fixed random 3D direction per case at 10/20/30/45 mm, or with its
margins scaled by 0.6/0.8/1.3/1.6, plus the unperturbed reference. Tight 5-fold ensemble WITHOUT mirroring
TTA (9 conditions x 194 cases); the curve is read as the change vs the no-TTA reference, so it is
internally consistent. Per case and condition: crop IoU with the oracle crop, lesion containment, Dice,
p_max / p_sum / cc_psz. Softmax files are deleted after scoring.

    data/envs/nnunet/Scripts/python.exe tools/dose_response.py
Writes reports/deployment_roi/dose_response_msd.csv (appended per case; resumable). Refuses to run twice,
below-normal priority, DOSE_THREADS (default 8). DOSE_LIMIT=n runs n cases per condition (smoke test).
"""
import csv
import os
import shutil
import sys
import time

THREADS = int(os.environ.get("DOSE_THREADS", "8"))  # <=8 keeps 4 of 12 cores free (CLAUDE.md)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(THREADS)

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/tools")
from build_roi_dataset import IMG, bbox_from_roi, mask_path  # noqa: E402

MARGIN_MM = np.array([100.0, 50.0, 15.0])
MODEL = f"{ROOT}/models/nnunet/tight_cv/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
CASES = f"{ROOT}/reports/nnunet_summaries/external/external_cases.csv"
OUT = f"{ROOT}/reports/deployment_roi/dose_response_msd.csv"
WORK = f"{ROOT}/data/processed/ct/dose_response"
CONDITIONS = [("ref", 0, 1.0)] + [(f"shift{s}", s, 1.0) for s in (10, 20, 30, 45)] + \
             [(f"scale{k}", 0, k) for k in (0.6, 0.8, 1.3, 1.6)]
LIMIT = int(os.environ.get("DOSE_LIMIT", "0"))
FIELDS = ["condition", "shift_mm", "margin_scale", "case", "y", "crop_iou", "lesion_in_crop", "dice",
          "p_max", "p_sum", "cc_psz"]


def box(bb_vox, sp, shift_mm, scale, shape):
    """Oracle bbox -> crop box with margins x scale, moved by shift_mm (vector), clipped to the image."""
    margin = np.ceil(MARGIN_MM * scale / sp).astype(int)
    off = np.round(shift_mm / sp).astype(int)
    out = []
    for (lo, hi), m, o, s in zip(bb_vox, margin, off, shape):
        a, b = lo - m + o, hi + m + o
        out.append((int(max(0, min(a, s - 1))), int(min(s, max(b, a + 1, 1)))))
    return out


def iou(a, b):
    inter = np.prod([max(0, min(ah, bh) - max(al, bl)) for (al, ah), (bl, bh) in zip(a, b)])
    vol = lambda x: np.prod([hi - lo for lo, hi in x])
    return inter / (vol(a) + vol(b) - inter)


def scores(pred_dir, case, gt):
    pr = np.asarray(nib.load(f"{pred_dir}/{case}.nii.gz").dataobj) == 1
    fg = np.load(f"{pred_dir}/{case}.npz")["probabilities"][1]
    den = gt.sum() + pr.sum()
    dice = 2 * np.logical_and(gt, pr).sum() / den if den else np.nan
    m = fg > 0.5
    if m.any():  # same definitions as scripts/runpod/ext_eval.py
        lab, n = ndimage.label(m)
        idx = range(1, n + 1)
        peak = np.array(ndimage.maximum(fg, lab, idx), float)
        size = np.array(ndimage.sum(m, lab, idx), float)
        ccpsz = float((peak * size ** (1.0 / 15.0)).max())
    else:
        ccpsz = float(fg.max())
    return dice, float(fg.max()), float(fg.sum()), ccpsz


def main():
    lock = f"{WORK}/.running.pid"
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(lock):
        pid = int(open(lock).read().strip() or 0)
        if pid and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            sys.exit(f"already running (pid {pid}); refusing to start a second copy")
    open(lock, "w").write(str(os.getpid()))
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(THREADS)
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        pred = nnUNetPredictor(device=torch.device("cuda"), use_mirroring=False, verbose=False,
                               verbose_preprocessing=False)
        pred.initialize_from_trained_model_folder(MODEL, use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_final.pth")
        msd = pd.read_csv(CASES).query("source == 'MSD'")
        done = set()
        if os.path.exists(OUT):
            d = pd.read_csv(OUT)
            done = set(zip(d.condition, d.case))
        new = not os.path.exists(OUT)
        fh = open(OUT, "a", newline="")
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for name, shift, scale in CONDITIONS:
            todo = [r for r in msd.itertuples() if (name, r.case) not in done]
            if LIMIT:
                todo = todo[:LIMIT]
            print(f"{name}: {len(todo)} to do", flush=True)
            t0 = time.time()
            for i, r in enumerate(todo, 1):
                c = r.case
                ref = nib.load(f"{ROOT}/{mask_path(c)}")
                sp = np.array(ref.header.get_zooms()[:3], float)
                lab = np.asarray(ref.dataobj)
                bb = bbox_from_roi(np.isin(lab, (4, 5)))
                rng = np.random.default_rng(int(c[:6]))  # same direction for a case across shift levels
                v = rng.normal(size=3)
                cb = box(bb, sp, shift * v / np.linalg.norm(v), scale, lab.shape)
                ob = box(bb, sp, np.zeros(3), 1.0, lab.shape)
                les = lab == 1
                sl = tuple(slice(lo, hi) for lo, hi in cb)
                im = nib.load(f"{ROOT}/{IMG}/{c}_0000.nii.gz")
                aff = im.affine.copy()
                aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([cb[0][0], cb[1][0], cb[2][0]])
                tmp = f"{WORK}/_tmp"
                os.makedirs(tmp, exist_ok=True)
                nib.save(nib.Nifti1Image(np.asarray(im.dataobj)[sl].astype(np.int16), aff), f"{tmp}/{c}_0000.nii.gz")
                pred.predict_from_files([[f"{tmp}/{c}_0000.nii.gz"]], [f"{tmp}/{c}"], save_probabilities=True,
                                        overwrite=True, num_processes_preprocessing=1,
                                        num_processes_segmentation_export=1)
                dice, pmax, psum, ccpsz = scores(tmp, c, les[sl])
                w.writerow({"condition": name, "shift_mm": shift, "margin_scale": scale, "case": c, "y": r.y,
                            "crop_iou": iou(cb, ob),
                            "lesion_in_crop": les[sl].sum() / les.sum() if les.any() else "",
                            "dice": dice, "p_max": pmax, "p_sum": psum, "cc_psz": ccpsz})
                fh.flush()
                shutil.rmtree(tmp, ignore_errors=True)
                rate = (time.time() - t0) / i
                print(f"  {name} {i}/{len(todo)} {c} ({rate:.0f}s/case, ETA {rate * (len(todo) - i) / 60:.0f} min)",
                      flush=True)
        fh.close()
        print("DOSE_DONE", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
