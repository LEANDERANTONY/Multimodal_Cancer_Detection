"""Thesis-cohort control, 3D part: run the deployment pipeline on the thesis's two-source TCIA set.

The thesis set (AUROC 0.9999) was cancer = Pancreatic-CT-CBCT-SEG (40 patients) vs control = NIH Pancreas-CT, i.e.
class tied to source. The NIH controls are PANORAMA's NIH external cases (80), already run through this pipeline
(data/processed/ct/deploy_crops/baseline_oof). Here the 40 cancers go through the same pipeline:
    native diagnostic CT per patient (the Philips / GE series; the Varian CBCT and MIM-aligned copies are skipped)
    -> official PANORAMA baseline pancreas nnU-Net (Dataset103, 5-fold ensemble: none of these scans was in its data)
    -> tight crop (bbox + 100 x 50 x 15 mm, native resolution) -> tight 3D nnU-Net 5-fold ensemble with mirroring TTA
    -> p_max, p_sum, cc_psz (scripts/runpod/ext_eval.py definitions); an empty pancreas mask is a failure.

    data/envs/nnunet/Scripts/python.exe tools/thesis_control.py
Writes data/processed/ct/thesis_cohort/{raw,masks,crops,pred}/ and reports/thesis_control/detect_3d_cbct_seg.csv.
Resumable per case, refuses to run twice, below-normal priority, THESIS_THREADS (default 8).
"""
import os
import sys
import time

THREADS = int(os.environ.get("THESIS_THREADS", "8"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(THREADS)

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_infer import MARGIN_MM, MODEL, bbox_from_roi, expand_clip  # noqa: E402
from stage1_segment import BASE_MODEL, to_uint8  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/data/raw/ct/cancer"
WORK = f"{ROOT}/data/processed/ct/thesis_cohort"
OUT = f"{ROOT}/reports/thesis_control/detect_3d_cbct_seg.csv"


def native_series():
    m = pd.read_csv(f"{SRC}/metadata.csv")
    m = m[(m.Modality == "CT") & m.Manufacturer.str.strip().isin(["Philips", "GE MEDICAL SYSTEMS"])]
    assert m["Subject ID"].is_unique and len(m) == 40, m["Subject ID"].value_counts().head()
    return {r["Subject ID"]: f"{SRC}/{r['File Location'].replace(chr(92), '/').lstrip('./')}" for _, r in m.iterrows()}


def convert(subject, folder, dst):
    if os.path.exists(dst):
        return
    rd = sitk.ImageSeriesReader()
    rd.SetFileNames(rd.GetGDCMSeriesFileNames(folder))
    sitk.WriteImage(sitk.Cast(rd.Execute(), sitk.sitkInt16), dst + ".tmp.nii.gz")
    os.replace(dst + ".tmp.nii.gz", dst)


def crop(raw, mask, dst):
    roi = np.asarray(nib.load(mask).dataobj) > 0
    if not roi.any():
        return False
    im = nib.load(raw)
    sp = np.array(im.header.get_zooms()[:3], float)
    cb = expand_clip(bbox_from_roi(roi), [int(np.ceil(mm / s)) for mm, s in zip(MARGIN_MM, sp)], roi.shape)
    aff = im.affine.copy()
    aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([cb[0][0], cb[1][0], cb[2][0]])
    nib.save(nib.Nifti1Image(np.asarray(im.dataobj)[tuple(slice(lo, hi) for lo, hi in cb)].astype(np.int16), aff), dst)
    return True


def scores(npz):
    fg = np.load(npz)["probabilities"][1]
    pmax, psum, m = float(fg.max()), float(fg.sum()), fg > 0.5
    if m.any():
        lab, n = ndimage.label(m)
        idx = range(1, n + 1)
        peak = np.array(ndimage.maximum(fg, lab, idx), float)
        size = np.array(ndimage.sum(m, lab, idx), float)
        return pmax, psum, float((peak * size ** (1.0 / 15.0)).max())
    return pmax, psum, pmax


def main():
    for d in ("raw", "masks", "crops", "pred"):
        os.makedirs(f"{WORK}/{d}", exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lock = f"{WORK}/.running.pid"
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read() or 0)):
        sys.exit(f"already running (pid {open(lock).read()})")
    open(lock, "w").write(str(os.getpid()))
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(THREADS)
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        series = native_series()
        cases = sorted(series)
        for i, c in enumerate(cases, 1):
            convert(c, series[c], f"{WORK}/raw/{c}_0000.nii.gz")
        print(f"converted {len(cases)} native CTs", flush=True)
        seg = nnUNetPredictor(device=torch.device("cuda"), verbose=False, verbose_preprocessing=False)
        seg.initialize_from_trained_model_folder(BASE_MODEL, use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_final.pth")
        for c in cases:
            out = f"{WORK}/masks/{c}.nii.gz"
            if not os.path.exists(out):
                t = time.time()
                seg.predict_from_files([[f"{WORK}/raw/{c}_0000.nii.gz"]], [f"{WORK}/masks/_raw_{c}"], save_probabilities=False,
                                       overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)
                to_uint8(f"{WORK}/masks/_raw_{c}.nii.gz", out)
                os.remove(f"{WORK}/masks/_raw_{c}.nii.gz")
                print(f"  stage 1 {c} ({time.time() - t:.0f}s)", flush=True)
        del seg
        torch.cuda.empty_cache()
        det = nnUNetPredictor(device=torch.device("cuda"), use_mirroring=True, verbose=False, verbose_preprocessing=False)
        det.initialize_from_trained_model_folder(MODEL, use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_final.pth")
        rows = []
        for c in cases:
            img = f"{WORK}/crops/{c}_0000.nii.gz"
            ok = os.path.exists(img) or crop(f"{WORK}/raw/{c}_0000.nii.gz", f"{WORK}/masks/{c}.nii.gz", img)
            if not ok:
                rows.append({"case": c, "source": "Pancreatic-CT-CBCT-SEG", "y": 1, "stage1_failed": True})
                print(f"  {c}: empty pancreas mask", flush=True)
                continue
            if not os.path.exists(f"{WORK}/pred/{c}.npz"):
                t = time.time()
                det.predict_from_files([[img]], [f"{WORK}/pred/{c}"], save_probabilities=True, overwrite=True,
                                       num_processes_preprocessing=1, num_processes_segmentation_export=1)
                print(f"  stage 2 {c} ({time.time() - t:.0f}s)", flush=True)
            p_max, p_sum, cc_psz = scores(f"{WORK}/pred/{c}.npz")
            rows.append({"case": c, "source": "Pancreatic-CT-CBCT-SEG", "y": 1, "stage1_failed": False,
                         "p_max": p_max, "p_sum": p_sum, "cc_psz": cc_psz})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"THESIS_3D_DONE {OUT}", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
