"""Re-infer the LOOSE 5-fold CV models (Dataset700) on their own validation cases, locally,
and save per-case detection scores -> lets us put a CI on the loose confound tax
(tight: +0.031 [+0.012,+0.054]; loose +0.019 had no CI because its CV softmax was deleted).

Machine-safe (see D:/Documents/Projects/CLAUDE.md): below-normal priority, 4 threads, one GPU
process, refuses to start if a copy is running, resumable (appends to the CSV, skips done cases),
prints progress + ETA per case. Probabilities are scored in memory and never written to disk.

Run with the separate nnU-Net env (torch 2.8; nnU-Net excludes torch 2.9):
  data/envs/nnunet/Scripts/python.exe tools/loose_cv_reinfer.py
Stop: Ctrl+C, or kill the python process — rerunning resumes where it left off.
Then: .venv/Scripts/python.exe tools/confound_tax_ci.py reports/nnunet_summaries/loose_battery/cv_scores.csv
"""
import os
THREADS = int(os.environ.get("REINFER_THREADS", "4"))  # <=8 keeps 4 of 12 cores free (CLAUDE.md)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS)); os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("nnUNet_raw", "unused"); os.environ.setdefault("nnUNet_preprocessed", "unused")
os.environ.setdefault("nnUNet_results", "unused")
import csv, json, sys, time
import numpy as np
import psutil
import torch
from scipy import ndimage

ROOT = "D:/Documents/Projects/Multimodal_Cancer_Detection"
MODEL = f"{ROOT}/models/nnunet/loose_cv/Dataset700_PanoramaPDAC/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
SUMM = f"{ROOT}/reports/nnunet_summaries/nnunet_results/Dataset700_PanoramaPDAC/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
IMG = f"{ROOT}/data/processed/ct/nnunet_raw/Dataset700_PanoramaPDAC/imagesTr"
SCANNER_CSV = f"{ROOT}/reports/nnunet_summaries/tight_battery/lomo_scores.csv"
TTA = os.environ.get("REINFER_TTA", "1") == "1"  # mirroring TTA (nnU-Net default; ~8x slower in 3D)
OUT = f"{ROOT}/reports/nnunet_summaries/loose_battery/cv_scores{'' if TTA else '_notta'}.csv"
LOCK = f"{ROOT}/reports/nnunet_summaries/loose_battery/.running.pid"
FOLDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
LIMIT = int(os.environ.get("REINFER_LIMIT", "0"))  # >0: smoke test, only this many cases per fold


def score(fg):
    pmax = float(fg.max()); psum = float(fg.sum()); m = fg > 0.5
    if m.any():
        lab, n = ndimage.label(m); idx = range(1, n + 1)
        peak = np.array(ndimage.maximum(fg, lab, idx), float); size = np.array(ndimage.sum(m, lab, idx), float)
        return pmax, psum, float((peak * size ** (1.0 / 15.0)).max())
    return pmax, psum, pmax


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(LOCK):
        pid = int(open(LOCK).read().strip() or 0)
        if pid and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            sys.exit(f"already running (pid {pid}); refusing to start a second copy")
    open(LOCK, "w").write(str(os.getpid()))
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(THREADS)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    scanner = {r["case"]: r["scanner"] for r in csv.DictReader(open(SCANNER_CSV))}
    done = set()
    if os.path.exists(OUT):
        done = {(int(r["fold"]), r["case"]) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline=""); w = csv.writer(fh)
    if new:
        w.writerow(["fold", "case", "scanner", "y", "p_max", "p_sum", "cc_psz"]); fh.flush()

    io = SimpleITKIO()
    for f in FOLDS:
        s = json.load(open(f"{SUMM}/fold_{f}/validation/summary.json"))
        cases = sorted((os.path.basename(c["prediction_file"]).replace(".nii.gz", ""), int(c["metrics"]["1"]["n_ref"] > 0))
                       for c in s["metric_per_case"])
        if LIMIT: cases = cases[:LIMIT]
        todo = [(c, y) for c, y in cases if (f, c) not in done]
        print(f"fold {f}: {len(cases)} cases, {len(todo)} to do", flush=True)
        if not todo: continue
        pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=TTA,
                               perform_everything_on_device=True, device=torch.device("cuda"), allow_tqdm=False)
        pred.initialize_from_trained_model_folder(MODEL, use_folds=(f,), checkpoint_name="checkpoint_final.pth")
        t0 = time.time()
        for i, (c, y) in enumerate(todo):
            img, props = io.read_images([f"{IMG}/{c}_0000.nii.gz"])
            _, prob = pred.predict_single_npy_array(img, props, None, None, True)
            w.writerow([f, c, scanner.get(c, "Other"), y, *score(prob[1])]); fh.flush()
            el = time.time() - t0; eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"  fold {f} {i+1}/{len(todo)} {c} ({el/(i+1):.1f}s/case, fold ETA {eta/60:.0f} min)", flush=True)
        del pred; torch.cuda.empty_cache()
    fh.close(); os.remove(LOCK)
    print("LOOSE_REINFER_DONE", flush=True)


if __name__ == "__main__":
    main()
