"""Deployment-ROI stage 2 on the Dutch cohort, OUT-OF-FOLD: each case is scored by the tight-CV fold model
that never trained on it, with mirroring TTA - the exact protocol of the oracle Dutch CV scores
(reports/nnunet_summaries/tight_battery/cv_scores.csv: nnUNetv2_predict -f <fold>, default TTA).

Crops are cut by tools/deploy_infer.py:build_crop (same code as the external run) from
data/processed/ct/stage1_masks/<arm>/<case>.nii.gz. Arm "oracle_rebuild" = masks rebuilt from the provided
labels 4+5 (tools/deploy_identity_gate.py) - it must reproduce the training crops and the oracle scores.

    python tools/deploy_infer_oof.py <arm>           # baseline_oof | totalseg | oracle_rebuild
Writes reports/deployment_roi/detect_<arm>_dutch.csv (fold, case, scanner, y, n_ref, n_pred, dice, p_max,
p_sum, cc_psz), appended per case; softmax files are deleted after scoring; cases without a usable stage-1
mask are listed in <crop dir>/skipped.txt (segmenter failures). Resumable; refuses to run twice.
DEPLOY_CASES=<file> restricts to the listed cases (gate sample); DEPLOY_THREADS (default 8).
"""
import csv
import glob
import os
import sys
import time

THREADS = int(os.environ.get("DEPLOY_THREADS", "8"))
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
from deploy_infer import MODEL, build_crop  # noqa: E402  (same crop code as the external run)

META = f"{ROOT}/reports/nnunet_summaries/tight_battery/cv_scores.csv"
FIELDS = ["fold", "case", "scanner", "y", "n_ref", "n_pred", "dice", "p_max", "p_sum", "cc_psz"]


def score(pred_base, lab_path):
    """Same definitions as scripts/runpod/tight_battery.py:score_case and ext_eval.py."""
    gt = np.asarray(nib.load(lab_path).dataobj) == 1
    pr = np.asarray(nib.load(f"{pred_base}.nii.gz").dataobj) == 1
    fg = np.load(f"{pred_base}.npz")["probabilities"][1]
    den = gt.sum() + pr.sum()
    dice = 2 * np.logical_and(gt, pr).sum() / den if den else float("nan")
    pmax, psum, m = float(fg.max()), float(fg.sum()), fg > 0.5
    if m.any():
        lab, n = ndimage.label(m)
        idx = range(1, n + 1)
        peak = np.array(ndimage.maximum(fg, lab, idx), float)
        size = np.array(ndimage.sum(m, lab, idx), float)
        ccpsz = float((peak * size ** (1.0 / 15.0)).max())
    else:
        ccpsz = pmax
    return int(gt.sum()), int(pr.sum()), dice, pmax, psum, ccpsz


def main():
    arm = sys.argv[1]
    assert arm in ("baseline_oof", "totalseg", "oracle_rebuild")
    d = f"{ROOT}/data/processed/ct/deploy_crops/{arm}_dutch"
    for sub in ("imagesTs", "labelsTs", "pred"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    lock = f"{d}/.running.pid"
    if os.path.exists(lock):
        pid = int(open(lock).read().strip() or 0)
        if pid and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            sys.exit(f"already running (pid {pid}); refusing to start a second copy")
    open(lock, "w").write(str(os.getpid()))
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(THREADS)
    out = f"{ROOT}/reports/deployment_roi/detect_{arm}_dutch.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        meta = pd.read_csv(META)
        if os.environ.get("DEPLOY_CASES"):
            keep = set(open(os.environ["DEPLOY_CASES"]).read().split())
            meta = meta[meta.case.isin(keep)]
        done = set(pd.read_csv(out).case) if os.path.exists(out) else set()
        new = not os.path.exists(out)
        fh = open(out, "a", newline="")
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        for f in sorted(meta.fold.unique()):
            todo = meta[(meta.fold == f) & ~meta.case.isin(done)]
            print(f"{arm} fold {f}: {len(todo)} to do", flush=True)
            if todo.empty:
                continue
            pred = nnUNetPredictor(device=torch.device("cuda"), use_mirroring=True, verbose=False,
                                   verbose_preprocessing=False)
            pred.initialize_from_trained_model_folder(MODEL, use_folds=(int(f),), checkpoint_name="checkpoint_final.pth")
            t0 = time.time()
            for i, r in enumerate(todo.itertuples(), 1):
                try:  # a missing / empty stage-1 mask is a segmenter failure: log it, keep going
                    img = build_crop(r.case, arm, d)
                except Exception as e:
                    print(f"  SKIP {r.case}: no usable stage-1 mask ({type(e).__name__})", flush=True)
                    with open(f"{d}/skipped.txt", "a") as sk:
                        sk.write(f"{r.case}\n")
                    continue
                base = f"{d}/pred/{r.case}"
                pred.predict_from_files([[img]], [base], save_probabilities=True, overwrite=True,
                                        num_processes_preprocessing=1, num_processes_segmentation_export=1)
                n_ref, n_pred, dice, pmax, psum, ccpsz = score(base, f"{d}/labelsTs/{r.case}.nii.gz")
                w.writerow({"fold": f, "case": r.case, "scanner": r.scanner, "y": r.y, "n_ref": n_ref,
                            "n_pred": n_pred, "dice": dice, "p_max": pmax, "p_sum": psum, "cc_psz": ccpsz})
                fh.flush()
                for p in glob.glob(f"{base}.*"):
                    os.remove(p)
                rate = (time.time() - t0) / i
                print(f"  {arm} f{f} {i}/{len(todo)} {r.case} ({rate:.1f}s/case, ETA {rate * (len(todo) - i) / 60:.0f} min)",
                      flush=True)
        fh.close()
        print("DEPLOY_OOF_DONE", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
