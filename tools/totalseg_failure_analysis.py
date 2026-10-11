"""Why does TotalSegmentator return an empty pancreas on some Dutch scans? Failed vs successful scans on
acquisition, anatomy and intensity features.

Per case (the scans stage 1 ran on, stage1_quality_totalseg_dutch.csv): scanner, label, age, sex, lesion-mask
source (manual / automatic); voxel spacing, matrix, field of view and scan length (mm), axis orientation; the provided pancreas
(4) + duct (5) volume and its distance to the scan's top and bottom edges; HU inside that pancreas (median,
a contrast-phase proxy since the metadata has no phase), and whole-scan HU minimum / median / maximum and
air fraction (catches non-standard intensity offsets).

    python tools/totalseg_failure_analysis.py
Writes reports/deployment_roi/totalseg_failure_features.csv (appended per case; resumable), then prints
failed-vs-ok medians with Mann-Whitney p-values and failure rates by feature quartile.
"""
import csv
import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import mannwhitneyu

sys.path.insert(0, "tools")
from build_roi_dataset import IMG, mask_path  # noqa: E402

QUAL = "reports/deployment_roi/stage1_quality_totalseg_dutch.csv"
CLIN = "data/raw/ct/panorama_labels/clinical_information.xlsx"
OUT = "reports/deployment_roi/totalseg_failure_features.csv"
FIELDS = ["case", "scanner", "y", "age", "sex", "mask_source", "empty", "orient", "sx", "sy", "sz", "nx", "ny", "nz",
          "fov_mm", "length_mm", "panc_ml", "panc_to_top_mm", "panc_to_bottom_mm", "panc_hu_median",
          "hu_min", "hu_median", "hu_max", "air_frac"]
NUM = ["sx", "sz", "nx", "nz", "fov_mm", "length_mm", "panc_ml", "panc_to_top_mm", "panc_to_bottom_mm",
       "panc_hu_median", "hu_min", "hu_median", "hu_max", "air_frac", "age"]


def features(case):
    m = nib.load(mask_path(case))
    sp = np.array(m.header.get_zooms()[:3], float)
    lab = np.isin(np.asarray(m.dataobj), (4, 5))
    im = np.asarray(nib.load(f"{IMG}/{case}_0000.nii.gz").dataobj)
    zs = np.where(lab.any(axis=(0, 1)))[0]
    sub = im[::2, ::2, ::2]
    return {"orient": "".join(nib.aff2axcodes(m.affine)), "sx": sp[0], "sy": sp[1], "sz": sp[2], "nx": im.shape[0], "ny": im.shape[1], "nz": im.shape[2],
            "fov_mm": im.shape[0] * sp[0], "length_mm": im.shape[2] * sp[2],
            "panc_ml": lab.sum() * sp.prod() / 1000,
            "panc_to_bottom_mm": zs.min() * sp[2] if zs.size else np.nan,
            "panc_to_top_mm": (im.shape[2] - 1 - zs.max()) * sp[2] if zs.size else np.nan,
            "panc_hu_median": float(np.median(im[lab])) if lab.any() else np.nan,
            "hu_min": float(sub.min()), "hu_median": float(np.median(sub)), "hu_max": float(sub.max()),
            "air_frac": float((sub < -900).mean())}


def summarise(d):
    d["failed"] = d["empty"].astype(str).str.lower() == "true"
    print(f"n={len(d)} failed={d.failed.sum()} ({d.failed.mean():.1%})\n")
    print(f"{'feature':18s} {'ok median':>10s} {'failed med':>10s} {'p':>8s}   failure rate by quartile (Q1..Q4)")
    for c in NUM:
        x = pd.to_numeric(d[c], errors="coerce")
        a, b = x[~d.failed].dropna(), x[d.failed].dropna()
        if len(b) < 3:
            continue
        p = mannwhitneyu(a, b).pvalue
        try:
            q = d.failed.groupby(pd.qcut(x, 4, duplicates="drop"), observed=True).mean()
            qs = " ".join(f"{v:.1%}" for v in q)
        except ValueError:
            qs = ""
        print(f"{c:18s} {a.median():10.2f} {b.median():10.2f} {p:8.1e}   {qs}")
    for c in ("scanner", "y", "sex", "mask_source", "orient"):
        print(f"\nfailure rate by {c}:")
        print(d.groupby(c).failed.agg(["size", "sum", "mean"]).round(3).to_string())


def main():
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    q = pd.read_csv(QUAL)[["case", "group", "y", "empty"]]
    clin = pd.read_excel(CLIN).set_index("PANORAMA_study_id")
    done = set(pd.read_csv(OUT).case) if os.path.exists(OUT) else set()
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        todo = q[~q.case.isin(done)]
        for i, r in enumerate(todo.itertuples(), 1):
            c = clin.loc[r.case]
            row = {"case": r.case, "scanner": r.group, "y": r.y, "empty": r.empty,
                   "age": pd.to_numeric(str(c.patient_age).rstrip("Y"), errors="coerce"), "sex": c.patient_sex,
                   "mask_source": "manual" if "manual" in mask_path(r.case) else "automatic"}
            row.update(features(r.case))
            w.writerow(row)
            fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    summarise(pd.read_csv(OUT))


if __name__ == "__main__":
    main()
