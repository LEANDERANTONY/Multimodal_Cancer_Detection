"""Causal test: is the header-anatomy mismatch what makes TotalSegmentator's `roi_subset pancreas` return nothing?

Same TotalSegmentator 2.18 call as stage 1 (tools/stage1_segment.py, arm totalseg). Voxels are never changed; only the
header's cranio-caudal direction is rewritten (direction column negated, origin moved to the other end of that axis), so
the output stays on the original voxel grid and is compared with the provided pancreas + duct mask directly.

Arms, in run order:
  sham        re-saved unchanged: 20 failed + 10 non-failed inverted scans; checks the method reproduces stage 1
  fix         all inverted scans, header corrected to match the anatomy
  other       the 10 failures not classed as inverted, header flipped
  break_win   correctly oriented scans of the Windows-written batch (sform_code 0), header deliberately inverted
  break_unix  correctly oriented Unix-written scans matched on scanner x label, header deliberately inverted

    data/envs/nnunet/Scripts/python.exe tools/orientation_totalseg_test.py [arm ...] [--limit n]
Appends reports/deployment_roi/orientation_totalseg_test.csv (resumable); masks in data/processed/ct/orientation_ts/.
"""
import csv
import glob
import os
import shutil
import sys
import tempfile
import time

THREADS = 8
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(THREADS)

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOTALSEG_HOME_DIR", os.path.join(os.path.dirname(ROOT), "..", "..", "data", "envs", "totalseg")
                      if ".claude" in ROOT else f"{ROOT}/data/envs/totalseg")
IMG = "data/raw/ct/panorama/images/{}_0000.nii.gz"
MASK_ROOT = "data/processed/ct/orientation_ts"
OUT = "reports/deployment_roi/orientation_totalseg_test.csv"
ORDER = ["sham", "fix", "other", "break_win", "break_unix"]
FIELDS = ["arm", "case", "scanner", "y", "orientation", "sform", "failed_stage1", "status", "voxels", "dice", "seconds"]


def cohort():
    o = pd.read_csv("reports/deployment_roi/orientation_check.csv")[["case", "orientation"]]
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["case", "scanner", "y"]]
    d = cv.merge(o, on="case")
    failed = set(open("data/processed/ct/deploy_crops/totalseg_dutch/skipped.txt").read().split())
    d["failed_stage1"] = d.case.isin(failed)
    d["sform"] = [int(nib.load(IMG.format(c)).header["sform_code"]) for c in d.case]
    return d


def arms(d):
    inv = d[d.orientation == "inverted"]
    a = {"sham": pd.concat([inv[inv.failed_stage1].sample(20, random_state=0), inv[~inv.failed_stage1].sample(10, random_state=0)]),
         "fix": inv,
         "other": d[d.failed_stage1 & (d.orientation != "inverted")]}
    ok = d[(d.orientation == "header_ok") & ~d.failed_stage1]
    a["break_win"] = ok[ok.sform == 0]
    pool, picks = ok[ok.sform == 1], []
    want = a["break_win"].groupby(["scanner", "y"]).size()
    for (s, y), n in want.items():  # matched on scanner x label; topped up below to the planned 80
        g = pool[(pool.scanner == s) & (pool.y == y)]
        picks.append(g.sample(min(n, len(g)), random_state=0))
    u = pd.concat(picks)
    if len(u) < 80:
        u = pd.concat([u, pool.drop(u.index).sample(80 - len(u), random_state=0)])
    a["break_unix"] = u.sample(min(80, len(u)), random_state=0)
    return a


def flip_header(im):
    """Negate the direction of the axis closest to cranio-caudal; move the origin so physical geometry stays rigid."""
    dirs = np.array(im.GetDirection()).reshape(3, 3)
    ax = int(np.argmax(np.abs(dirs[2])))
    n, sp = im.GetSize()[ax], im.GetSpacing()[ax]
    org = np.array(im.GetOrigin()) + dirs[:, ax] * sp * (n - 1)
    dirs[:, ax] *= -1
    out = sitk.Image(im)
    out.SetDirection(tuple(dirs.ravel()))
    out.SetOrigin(tuple(org))
    return out


def reference(case):
    for sub in ("manual_labels", "automatic_labels"):
        p = f"data/raw/ct/panorama_labels/{sub}/{case}.nii.gz"
        if os.path.exists(p):
            return np.isin(sitk.GetArrayFromImage(sitk.ReadImage(p)), (4, 5))
    return None


def main():
    argv = sys.argv[1:]
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0
    run = [a for a in argv if a in ORDER] or ORDER
    me = psutil.Process()
    me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    for p in psutil.process_iter(["pid", "cmdline"]):  # refuse to start a second copy
        cmd = " ".join(p.info["cmdline"] or [])
        if p.info["pid"] not in (me.pid, me.ppid()) and "orientation_totalseg_test" in cmd and "python" in cmd.lower():
            kids = {c.pid for c in me.parents()}
            if p.info["pid"] not in kids:
                sys.exit(f"already running: pid {p.info['pid']}")
    from totalsegmentator.python_api import totalsegmentator
    d = cohort()
    plan = arms(d)
    done = set()
    if os.path.exists(OUT):
        r = pd.read_csv(OUT)
        done = set(zip(r.arm, r.case))
    new = not os.path.exists(OUT)
    tmp_root = f"{MASK_ROOT}/_tmp"
    os.makedirs(tmp_root, exist_ok=True)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(FIELDS); fh.flush()
        for arm in run:
            todo = plan[arm][[(arm, c) not in done for c in plan[arm].case]]
            if limit:
                todo = todo.head(limit)
            os.makedirs(f"{MASK_ROOT}/{arm}", exist_ok=True)
            print(f"{arm}: {len(todo)} to do of {len(plan[arm])}", flush=True)
            for i, (_, c) in enumerate(todo.iterrows(), 1):
                t0, tmp = time.time(), tempfile.mkdtemp(dir=tmp_root)
                status, vox, dice = "error", 0, float("nan")
                try:
                    im = sitk.ReadImage(IMG.format(c.case))
                    src = f"{tmp}/in.nii.gz"
                    sitk.WriteImage(im if arm == "sham" else flip_header(im), src, useCompression=False)
                    totalsegmentator(src, f"{tmp}/out", roi_subset=["pancreas"], nr_thr_resamp=1, nr_thr_saving=1,
                                     robust_crop=False, quiet=True)
                    m = sitk.ReadImage(f"{tmp}/out/pancreas.nii.gz")
                    pred = sitk.GetArrayFromImage(m) > 0
                    vox = int(pred.sum())
                    status = "ok" if vox else "empty"
                    out = sitk.Cast(m > 0, sitk.sitkUInt8)
                    sitk.WriteImage(out, f"{MASK_ROOT}/{arm}/{c.case}.nii.gz", useCompression=True)
                    ref = reference(c.case)
                    if ref is not None and ref.shape == pred.shape and (ref.sum() + vox):
                        dice = 2 * float((ref & pred).sum()) / float(ref.sum() + vox)
                except Exception as e:  # a failure is a result to report; keep going
                    print(f"  {c.case} ERROR {e}", flush=True)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                w.writerow([arm, c.case, c.scanner, c.y, c.orientation, c.sform, c.failed_stage1, status, vox,
                            round(dice, 4), round(time.time() - t0, 1)])
                fh.flush()
                print(f"  {arm} {i}/{len(todo)} {c.case} {status} vox={vox} dice={dice:.3f} ({time.time() - t0:.0f}s) "
                      f"stage1_failed={c.failed_stage1}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
