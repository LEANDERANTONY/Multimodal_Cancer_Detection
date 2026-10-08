"""Deployment-ROI stage 2: tight-model detection on crops cut from PREDICTED pancreas masks.

For each arm (stage-1 masks in data/processed/ct/stage1_masks/<arm>/), cut the tight crop with the oracle
builder's rule (bbox + 100x50x15 mm per side, native resolution; label = reference lesion), run the tight
5-fold ensemble with mirroring TTA (same protocol as the oracle external run), then score per case with
scripts/runpod/ext_eval.py. Arm "oracle" re-runs the provided-mask crops locally, so oracle vs predicted is
compared within one pipeline (paired, same machine).

    data/envs/nnunet/Scripts/python.exe tools/deploy_infer.py <arm> [<arm> ...]     # e.g. oracle baseline_oof totalseg

Writes data/processed/ct/deploy_crops/<arm>/{imagesTs,labelsTs,pred} and
reports/deployment_roi/detect_<arm>_external.csv. Resumable per case; refuses to run twice; below-normal
priority; DEPLOY_THREADS (default 8). DEPLOY_LIMIT=n runs only n cases (smoke test).
"""
import os
import subprocess
import sys
import time

THREADS = int(os.environ.get("DEPLOY_THREADS", "8"))  # <=8 keeps 4 of 12 cores free (CLAUDE.md)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(THREADS)

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/tools")
from build_roi_dataset import IMG, bbox_from_roi, expand_clip, mask_path  # noqa: E402

MARGIN_MM = (100.0, 50.0, 15.0)
MODEL = f"{ROOT}/models/nnunet/tight_cv/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
ORACLE = f"{ROOT}/data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight"
CASES = f"{ROOT}/reports/nnunet_summaries/external/external_cases.csv"
LIMIT = int(os.environ.get("DEPLOY_LIMIT", "0"))


def build_crop(case, arm, d):
    img_p, lab_p = f"{d}/imagesTs/{case}_0000.nii.gz", f"{d}/labelsTs/{case}.nii.gz"
    if os.path.exists(img_p) and os.path.exists(lab_p):
        return img_p
    if arm == "oracle":  # the provided-mask crop the model was evaluated on
        for src, dst in ((f"{ORACLE}/imagesTr/{case}_0000.nii.gz", img_p), (f"{ORACLE}/labelsTr/{case}.nii.gz", lab_p)):
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
        return img_p
    ref = nib.load(f"{ROOT}/{mask_path(case)}")
    sp = np.array(ref.header.get_zooms()[:3], float)
    les = np.asarray(ref.dataobj) == 1
    roi = np.asarray(nib.load(f"{ROOT}/data/processed/ct/stage1_masks/{arm}/{case}.nii.gz").dataobj) > 0
    margin_vox = [int(np.ceil(mm / s)) for mm, s in zip(MARGIN_MM, sp)]
    cb = expand_clip(bbox_from_roi(roi), margin_vox, roi.shape)
    sl = tuple(slice(lo, hi) for lo, hi in cb)
    im = nib.load(f"{ROOT}/{IMG}/{case}_0000.nii.gz")
    aff = im.affine.copy()
    aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([cb[0][0], cb[1][0], cb[2][0]])
    nib.save(nib.Nifti1Image(np.asarray(im.dataobj)[sl].astype(np.int16), aff), img_p)
    nib.save(nib.Nifti1Image(les[sl].astype(np.uint8), aff), lab_p)
    return img_p


def run_arm(arm, cases, predictor):
    d = f"{ROOT}/data/processed/ct/deploy_crops/{arm}"
    for sub in ("imagesTs", "labelsTs", "pred"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    todo = [c for c in cases if not os.path.exists(f"{d}/pred/{c}.npz")]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"{arm}: {len(cases)} cases, {len(todo)} to predict", flush=True)
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        img = build_crop(c, arm, d)
        predictor.predict_from_files([[img]], [f"{d}/pred/{c}"], save_probabilities=True, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1)
        rate = (time.time() - t0) / i
        print(f"  {arm} {i}/{len(todo)} {c} ({rate:.0f}s/case, ETA {rate * (len(todo) - i) / 60:.0f} min)", flush=True)
    done = pd.read_csv(CASES)
    done = done[done.case.apply(lambda c: os.path.exists(f"{d}/pred/{c}.npz"))]
    sub_cases = f"{d}/cases.csv"
    done.to_csv(sub_cases, index=False)
    out = f"{ROOT}/reports/deployment_roi/detect_{arm}_external.csv"
    subprocess.run([sys.executable, f"{ROOT}/scripts/runpod/ext_eval.py", f"{d}/pred", f"{d}/labelsTs", sub_cases, out],
                   check=True)


def main():
    arms = sys.argv[1:]
    assert arms and all(a in ("oracle", "baseline_oof", "totalseg") for a in arms)
    lock = f"{ROOT}/data/processed/ct/deploy_crops/.running.pid"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    if os.path.exists(lock):
        pid = int(open(lock).read().strip() or 0)
        if pid and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            sys.exit(f"already running (pid {pid}); refusing to start a second copy")
    open(lock, "w").write(str(os.getpid()))
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(THREADS)
    os.makedirs(f"{ROOT}/reports/deployment_roi", exist_ok=True)
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        pred = nnUNetPredictor(device=torch.device("cuda"), use_mirroring=True, verbose=False,
                               verbose_preprocessing=False)
        pred.initialize_from_trained_model_folder(MODEL, use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_final.pth")
        cases = pd.read_csv(CASES).case.tolist()
        for arm in arms:
            run_arm(arm, cases, pred)
        print("DEPLOY_DONE", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
