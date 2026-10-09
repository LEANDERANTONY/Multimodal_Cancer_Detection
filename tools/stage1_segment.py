"""Deployment-ROI stage 1: predicted pancreas masks on the raw PANORAMA scans.

Arms (docs/deployment_and_mitigation_design.md, both used as published):
  totalseg  TotalSegmentator 2.18, --roi_subset pancreas, full-resolution model
  baseline  official PANORAMA baseline pancreas nnU-Net (Dataset103), out-of-fold: each case is
            segmented by the one fold model that held it out

    data/envs/nnunet/Scripts/python.exe tools/stage1_segment.py <arm> <external|dutch>

Writes data/processed/ct/stage1_masks/<arm>/<case>.nii.gz (uint8 0/1) and _log.csv (case, seconds,
status, voxels). Resumable (skips cases already written), refuses to run twice, below-normal priority,
STAGE1_THREADS (default 8) caps CPU threads. STAGE1_LIMIT=n runs only n cases (smoke test).
STAGE1_SHARD=i/n runs every n-th case starting at i (parallel workers; processing is unchanged).
STAGE1_MAX_VOXELS=n defers scans larger than n voxels (listed in _deferred.txt) - for a RAM-limited machine;
the export of a 1024x1024x331 scan needs ~20 GB RAM. A later run without the limit fills them in.
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time

THREADS = int(os.environ.get("STAGE1_THREADS", "8"))  # <=8 keeps 4 of 12 cores free (CLAUDE.md)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(THREADS)

import numpy as np
import pandas as pd
import psutil
import SimpleITK as sitk
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw/ct/panorama/images"
OUT_ROOT = f"{ROOT}/data/processed/ct/stage1_masks"
BASE = f"{ROOT}/models/panorama_baseline"
BASE_MODEL = f"{BASE}/Dataset103_PANORAMA_baseline_Pancreas_Segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres"
BASE_FOLDS = f"{BASE}/Dataset103_PANORAMA_baseline_Pancreas_Segmentation_folds.json"
os.environ.setdefault("TOTALSEG_HOME_DIR", f"{ROOT}/data/envs/totalseg")
LIMIT = int(os.environ.get("STAGE1_LIMIT", "0"))
SHARD = os.environ.get("STAGE1_SHARD", "")  # "i/n"
MAX_VOX = int(os.environ.get("STAGE1_MAX_VOXELS", "0"))


def case_list(cohort):
    ext = pd.read_csv(f"{ROOT}/reports/nnunet_summaries/external/external_cases.csv").case.tolist()
    if cohort == "external":
        return ext
    every = sorted(f[:12] for f in os.listdir(RAW) if f.endswith("_0000.nii.gz"))
    return [c for c in every if c not in set(ext)]


def log_row(log, case, sec, status, out):
    vox = int((sitk.GetArrayFromImage(sitk.ReadImage(out)) > 0).sum()) if os.path.exists(out) else 0
    if status == "ok" and vox == 0:
        status = "empty"
    new = not os.path.exists(log)
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["case", "seconds", "status", "voxels"])
        w.writerow([case, round(sec, 1), status, vox])


def to_uint8(src, dst):
    img = sitk.ReadImage(src)
    out = sitk.Cast(img > 0, sitk.sitkUInt8)
    out.CopyInformation(img)
    sitk.WriteImage(out, dst, useCompression=True)


def run_totalseg(cases, out_dir, log):
    from totalsegmentator.python_api import totalsegmentator
    for i, c in enumerate(cases, 1):
        out = f"{out_dir}/{c}.nii.gz"
        t = time.time()
        tmp = tempfile.mkdtemp(dir=out_dir, prefix="_tmp_")
        try:
            totalsegmentator(f"{RAW}/{c}_0000.nii.gz", tmp, roi_subset=["pancreas"], nr_thr_resamp=4,
                             nr_thr_saving=2, quiet=True)
            to_uint8(f"{tmp}/pancreas.nii.gz", out)
            status = "ok"
        except Exception as e:  # keep going; a failure is a result to report
            print(f"  {c} FAILED: {e}", flush=True)
            status = "error"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        log_row(log, c, time.time() - t, status, out)
        print(f"  {i}/{len(cases)} {c} {status} ({time.time() - t:.0f}s)", flush=True)


def run_baseline(cases, out_dir, log):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    folds = json.load(open(BASE_FOLDS))
    fold_of = {c: i for i, k in enumerate(folds) for c in folds[k]}
    done = 0
    for f in range(5):
        todo = [c for c in cases if fold_of[c] == f]
        if not todo:
            continue
        pred = nnUNetPredictor(device=torch.device("cuda"), verbose=False, verbose_preprocessing=False)
        pred.initialize_from_trained_model_folder(BASE_MODEL, use_folds=(f,), checkpoint_name="checkpoint_final.pth")
        for c in todo:  # one case at a time so progress and the log stay per case
            out = f"{out_dir}/{c}.nii.gz"
            t = time.time()
            try:
                pred.predict_from_files([[f"{RAW}/{c}_0000.nii.gz"]], [f"{out_dir}/_raw_{c}"],
                                              save_probabilities=False, overwrite=True,
                                              num_processes_preprocessing=1,
                                              num_processes_segmentation_export=1)
                to_uint8(f"{out_dir}/_raw_{c}.nii.gz", out)
                os.remove(f"{out_dir}/_raw_{c}.nii.gz")
                status = "ok"
            except Exception as e:
                print(f"  {c} FAILED: {e}", flush=True)
                status = "error"
            done += 1
            log_row(log, c, time.time() - t, status, out)
            print(f"  fold {f} {done}/{len(cases)} {c} {status} ({time.time() - t:.0f}s)", flush=True)


def main():
    arm, cohort = sys.argv[1], sys.argv[2]
    assert arm in ("totalseg", "baseline") and cohort in ("external", "dutch")
    out_dir = f"{OUT_ROOT}/{'baseline_oof' if arm == 'baseline' else 'totalseg'}"
    os.makedirs(out_dir, exist_ok=True)
    tag = SHARD.replace("/", "of") if SHARD else ""
    lock = f"{out_dir}/.running{tag}.pid"
    if os.path.exists(lock):
        pid = int(open(lock).read().strip() or 0)
        if pid and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            sys.exit(f"already running (pid {pid}); refusing to start a second copy")
    open(lock, "w").write(str(os.getpid()))
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))  # Windows class / Unix niceness
    torch.set_num_threads(THREADS)
    try:
        cases = case_list(cohort)
        if SHARD:
            i, n = map(int, SHARD.split("/"))
            cases = cases[i::n]
        todo = [c for c in cases if not os.path.exists(f"{out_dir}/{c}.nii.gz")]
        if MAX_VOX:
            import nibabel as nib
            big = [c for c in todo if np.prod(nib.load(f"{RAW}/{c}_0000.nii.gz").shape[:3]) > MAX_VOX]
            open(f"{out_dir}/_deferred.txt", "w").write("".join(c + chr(10) for c in big))
            todo = [c for c in todo if c not in set(big)]
            print(f"deferred {len(big)} scans > {MAX_VOX} voxels (see _deferred.txt)", flush=True)
        if LIMIT:
            todo = todo[:LIMIT]
        print(f"{arm} {cohort}: {len(cases)} cases, {len(todo)} to do", flush=True)
        (run_totalseg if arm == "totalseg" else run_baseline)(todo, out_dir, f"{out_dir}/_log{tag}.csv")
        print("STAGE1_DONE", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
