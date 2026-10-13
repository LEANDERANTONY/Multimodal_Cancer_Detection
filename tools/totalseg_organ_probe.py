"""Neighbouring-organ probe on the Dutch scans where TotalSegmentator returned an empty pancreas
(status "empty" in the stage-1 logs; all Dutch scans incl. the large ones). Same TotalSegmentator 2.18 and settings as stage 1.

  subset  roi_subset = pancreas + liver, spleen, kidneys, stomach, duodenum (the stage-1 call plus
          neighbours): does it still find the organs around the pancreas on these scans?
  full    the whole `total` task without roi_subset (no region-cropping step): does it find the pancreas
          when it segments the entire scan?
  robust  roi_subset pancreas with robust_crop=True (the developers' documented fix: a 3 mm cropping model)
If neighbours are found but not the pancreas, the failure is pancreas-specific; if `full` finds the pancreas,
the roi_subset cropping step causes it.

    python tools/totalseg_organ_probe.py subset|full|robust
Appends reports/deployment_roi/totalseg_organ_probe.csv (case, mode, organ, voxels); resumable per case.
"""
import csv
import glob
import os
import shutil
import sys
import tempfile
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "8"

import numpy as np
import pandas as pd
import psutil
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOTALSEG_HOME_DIR", f"{ROOT}/data/envs/totalseg")
RAW = f"{ROOT}/data/raw/ct/panorama/images"
MASKS = f"{ROOT}/data/processed/ct/stage1_masks/totalseg"
EXT = f"{ROOT}/reports/nnunet_summaries/external/external_cases.csv"
OUT = f"{ROOT}/reports/deployment_roi/totalseg_organ_probe.csv"
ORGANS = ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "stomach", "duodenum"]


def main():
    mode = sys.argv[1]
    assert mode in ("subset", "full", "robust")
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    from totalsegmentator.python_api import totalsegmentator
    logs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{MASKS}/_log*.csv")])
    last = logs.drop_duplicates("case", keep="last")
    dutch = set(c[:12] for c in os.listdir(RAW)) - set(pd.read_csv(EXT).case)
    cases = sorted(c for c in last[last.status == "empty"].case if c in dutch)
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) == 0:
        os.remove(OUT)  # left by a run killed before its header was flushed
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        done = set(d[d["mode"] == mode].case)
    new = not os.path.exists(OUT)
    tmp_root = f"{ROOT}/data/processed/ct/stage1_masks/_probe_tmp"
    os.makedirs(tmp_root, exist_ok=True)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["case", "mode", "organ", "voxels"])
        todo = [c for c in cases if c not in done]
        print(f"probe {mode}: {len(todo)} to do", flush=True)
        for i, c in enumerate(todo, 1):
            t = time.time()
            tmp = tempfile.mkdtemp(dir=tmp_root)
            try:
                totalsegmentator(f"{RAW}/{c}_0000.nii.gz", tmp, roi_subset=ORGANS if mode == "subset" else ["pancreas"] if mode == "robust" else None,
                                 robust_crop=mode == "robust",
                                 nr_thr_resamp=4, nr_thr_saving=2, quiet=True)
                for o in ORGANS:
                    p = f"{tmp}/{o}.nii.gz"
                    vox = int((sitk.GetArrayFromImage(sitk.ReadImage(p)) > 0).sum()) if os.path.exists(p) else -1
                    w.writerow([c, mode, o, vox])
            except Exception as e:
                print(f"  {c} FAILED: {e}", flush=True)
                w.writerow([c, mode, "error", -1])
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            fh.flush()
            print(f"  {mode} {i}/{len(todo)} {c} ({time.time() - t:.0f}s)", flush=True)
    print(f"PROBE_{mode.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
