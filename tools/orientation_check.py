"""Does each scan's header orientation agree with its anatomy along the cranio-caudal axis?

After reorienting to LPS by the header (index increasing = superior), the lungs should sit at the high-index end. For
each Dutch scan, the fraction of lung-density voxels (-950 to -600 HU) in a central 256 x 256 window is measured over
the first and the last 10 % of slices:
    header_ok   lungs at the high-index (superior) end, as the header says
    inverted    lungs at the low-index end: the image is stored head-to-foot relative to its header
    unclear     no lung at either end (e.g. scan starts below the lung bases), or both ends similar
Joined with the TotalSegmentator `roi_subset pancreas` failures to test whether inverted headers explain them.

    python tools/orientation_check.py [workers]
Writes reports/deployment_roi/orientation_check.csv (resumable: cases already in the file are skipped).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

OUT = "reports/deployment_roi/orientation_check.csv"
CASES = "reports/nnunet_summaries/tight_battery/cv_scores.csv"
IMG = "data/raw/ct/panorama/images/{}_0000.nii.gz"
MIN_LUNG, RATIO = 0.03, 3.0


def one(case):
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    im = sitk.DICOMOrient(sitk.ReadImage(IMG.format(case)), "LPS")
    a = sitk.GetArrayFromImage(im)
    n, h, w = a.shape
    k = max(1, n // 10)
    win = (slice(h // 2 - 128, h // 2 + 128), slice(w // 2 - 128, w // 2 + 128))

    def lung(s):
        return float(((s > -950) & (s < -600)).mean())

    lo, hi = lung(a[(slice(0, k),) + win]), lung(a[(slice(n - k, n),) + win])
    if max(lo, hi) < MIN_LUNG or max(lo, hi) < RATIO * min(lo, hi):
        state = "unclear"
    else:
        state = "header_ok" if hi > lo else "inverted"
    return {"case": case, "n_slices": n, "lung_low_end": lo, "lung_high_end": hi, "orientation": state}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    cases = pd.read_csv(CASES).case.tolist()
    done = set(pd.read_csv(OUT).case) if os.path.exists(OUT) else set()
    todo = [c for c in cases if c not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    header = not os.path.exists(OUT)
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, todo, chunksize=4), 1):
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=header, index=False)
            header = False
            if i % 50 == 0:
                print(f"{len(done) + i} / {len(cases)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
