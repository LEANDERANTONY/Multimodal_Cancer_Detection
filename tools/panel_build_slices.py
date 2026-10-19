"""Mitigation panel: build the 2.5D slice dataset from the tight oracle crops (Dataset701).

Per case: reorient to LPS, resample to 1.0 x 1.0 x 2.0 mm (one z spacing for every scanner, so slice spacing can't leak the
scanner), apply the tight nnU-Net CT normalisation (clip [-34, 172], (x - 69.91) / 36.21; from the tight
plans.json), centre pad / crop in-plane to 352 x 192 (x, y), and save
    <case>_img.npy  float16 (Z, 192, 352)   axis order z, y, x
    <case>_les.npy  int32 (Z,)              lesion voxels per slice (per-slice auxiliary label)
Meta (case, cohort, group, y, fold, Z, lesion voxels kept / lost by the in-plane crop) is appended to
meta.csv one case at a time, so a killed run resumes where it stopped (power cuts).

    python tools/panel_build_slices.py [n_cases]   # n_cases: smoke test on the first n
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import psutil
import SimpleITK as sitk

SRC = "data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight"
OUT = "data/processed/ct/panel_slices"
SPACING = (1.0, 1.0, 2.0)  # x, y, z (mm)
FRAME = (352, 192)  # x, y
CLIP, MEAN, STD = (-34.0, 172.0), 69.91, 36.21
FIELDS = ["case", "cohort", "group", "y", "fold", "z", "nx", "ny", "lesion_vox", "lesion_lost"]


def resample(img, interp):
    sp, size = np.array(img.GetSpacing()), np.array(img.GetSize())
    new = [max(1, int(round(n))) for n in size * sp / SPACING]
    return sitk.Resample(img, new, sitk.Transform(), interp, img.GetOrigin(), SPACING, img.GetDirection(),
                         0, img.GetPixelID())


def fit(a, fill):
    """Centre pad / crop the last two axes (y, x) of a (z, y, x) array to FRAME."""
    out = np.full((a.shape[0], FRAME[1], FRAME[0]), fill, a.dtype)
    src, dst = [], []
    for n, m in zip(a.shape[1:], (FRAME[1], FRAME[0])):
        if n >= m:
            o = (n - m) // 2
            src.append(slice(o, o + m)); dst.append(slice(0, m))
        else:
            o = (m - n) // 2
            src.append(slice(0, n)); dst.append(slice(o, o + n))
    out[:, dst[0], dst[1]] = a[:, src[0], src[1]]
    return out


def one_case(case):
    # one orientation for every scan (LPS = identity direction, as in 1962 of 1964 Dutch scans); MSD is stored
    # rotated 180 degrees in-plane and NIH flipped front-back with the slices reversed
    img = sitk.DICOMOrient(sitk.ReadImage(f"{SRC}/imagesTr/{case}_0000.nii.gz", sitk.sitkFloat32), "LPS")
    les = sitk.DICOMOrient(sitk.ReadImage(f"{SRC}/labelsTr/{case}.nii.gz", sitk.sitkUInt8), "LPS")
    x = sitk.GetArrayFromImage(resample(img, sitk.sitkLinear))  # (z, y, x)
    m = sitk.GetArrayFromImage(resample(les, sitk.sitkNearestNeighbor)) > 0
    x = ((np.clip(x, *CLIP) - MEAN) / STD).astype(np.float16)
    fill = np.float16((CLIP[0] - MEAN) / STD)
    xf, mf = fit(x, fill), fit(m, False)
    per_slice = mf.sum((1, 2)).astype(np.int32)
    tmp = f"{OUT}/_{case}"
    np.save(f"{tmp}_img.npy", xf)
    np.save(f"{tmp}_les.npy", per_slice)
    os.replace(f"{tmp}_img.npy", f"{OUT}/{case}_img.npy")
    os.replace(f"{tmp}_les.npy", f"{OUT}/{case}_les.npy")
    return {"z": x.shape[0], "nx": x.shape[2], "ny": x.shape[1], "lesion_vox": int(m.sum()),
            "lesion_lost": int(m.sum() - mf.sum())}


def main():
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(os.environ.get("PANEL_THREADS", 4)))
    os.makedirs(OUT, exist_ok=True)
    lock = f"{OUT}/.running.pid"
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read() or 0)):
        sys.exit(f"already running (pid {open(lock).read()})")
    open(lock, "w").write(str(os.getpid()))
    try:
        cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")
        cv = cv.assign(cohort="dutch").rename(columns={"scanner": "group"})[["case", "cohort", "group", "y", "fold"]]
        ext = pd.read_csv("reports/nnunet_summaries/external/external_cases.csv")
        ext = ext.assign(cohort="external", fold=-1).rename(columns={"source": "group"})[cv.columns]
        meta = pd.concat([cv, ext]).sort_values("case")
        if len(sys.argv) > 1:
            meta = meta.head(int(sys.argv[1]))
        part = f"{OUT}/meta.csv"
        seen = set(pd.read_csv(part).case) if os.path.exists(part) and os.path.getsize(part) > 0 else set()
        todo = meta[~meta.case.isin(seen)]
        print(f"{len(meta)} cases, {len(seen)} already done, {len(todo)} to do", flush=True)
        with open(part, "a", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            if not seen:
                w.writeheader()
            for i, r in enumerate(todo.itertuples(index=False), 1):
                w.writerow({**r._asdict(), **one_case(r.case)})
                fh.flush()
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}  rss {psutil.Process().memory_info().rss / 2**30:.2f} GB", flush=True)
        print("DONE", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
