"""Thesis-cohort control, 2.5D part and evaluation (run after tools/thesis_control.py).

Rebuilds the thesis's two-source comparison with the PANORAMA-trained models, both on crops cut from the PANORAMA
baseline segmenter's predicted pancreas masks:
    cancer  Pancreatic-CT-CBCT-SEG, 40 patients (tools/thesis_control.py)
    control NIH Pancreas-CT = PANORAMA's 80 NIH external cases (deploy_crops/baseline_oof, out-of-fold masks)
3D: tight nnU-Net scores (reports/thesis_control/detect_3d_cbct_seg.csv, reports/deployment_roi/detect_baseline_oof_external.csv).
2.5D: the crops are sliced exactly as tools/panel_build_slices.py does and scored by the 15 ERM soups (3 seeds x 5 CV
folds), averaged. A stage-1 failure counts as the lowest score. AUROC cancer vs control with a bootstrap CI; for
context, MSD PDAC vs the same NIH controls. The thesis model scored 0.9999 on this comparison.

    python tools/thesis_control_eval.py
Writes data/processed/ct/thesis_cohort/slices/, reports/thesis_control/detect_2p5d.csv and thesis_control.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
import psutil
import SimpleITK as sitk
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_train  # noqa: E402
from panel_build_slices import CLIP, MEAN, STD, fit, resample  # noqa: E402

WORK = "data/processed/ct/thesis_cohort"
SL = f"{WORK}/slices"
OUT = "reports/thesis_control"
N_BOOT = 2000


def slice_crop(case, path):
    if os.path.exists(f"{SL}/{case}_img.npy"):
        return
    img = sitk.DICOMOrient(sitk.ReadImage(path, sitk.sitkFloat32), "LPS")
    x = sitk.GetArrayFromImage(resample(img, sitk.sitkLinear))
    x = ((np.clip(x, *CLIP) - MEAN) / STD).astype(np.float16)
    xf = fit(x, np.float16((CLIP[0] - MEAN) / STD))
    np.save(f"{SL}/{case}_img.npy", xf)
    np.save(f"{SL}/{case}_les.npy", np.zeros(xf.shape[0], np.int32))


def score_2p5d(meta):
    panel_train.SLICES = SL  # workers=0 below, so the patched path is the one used
    args = type("A", (), {"depth": 20, "batch": 8, "workers": 0})()
    ps = []
    for seed in range(3):
        for k in range(5):
            model = panel_train.Net("resnet18").cuda()
            model.load_state_dict(torch.load(f"data/panel_runs/erm_resnet18_d20_s{seed}/cv{k}/soup.pt", map_location="cuda"))
            ps.append(panel_train.predict(model, meta.assign(group="x"), args, "cuda"))
            print(f"  2.5D model s{seed} cv{k} done", flush=True)
    return np.mean(ps, 0)


def auc_ci(y, p, rng):
    a = roc_auc_score(y, p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                        np.r_[p[rng.choice(pos, len(pos))], p[rng.choice(neg, len(neg))]]) for _ in range(N_BOOT)]
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(4)
    os.makedirs(SL, exist_ok=True)
    cb = pd.read_csv(f"{OUT}/detect_3d_cbct_seg.csv")
    ext = pd.read_csv("reports/deployment_roi/detect_baseline_oof_external.csv")
    nih, msd = ext[ext.source == "NIH"].assign(stage1_failed=False), ext[(ext.source == "MSD") & (ext.y == 1)]
    paths = {c: f"{WORK}/crops/{c}_0000.nii.gz" for c in cb.case[~cb.stage1_failed]}
    paths.update({c: f"data/processed/ct/deploy_crops/baseline_oof/imagesTs/{c}_0000.nii.gz" for c in pd.concat([nih, msd]).case})
    for c, p in paths.items():
        slice_crop(c, p)
    meta = pd.DataFrame({"case": list(paths), "y": 0})
    meta["p_2p5d"] = score_2p5d(meta)
    meta.to_csv(f"{OUT}/detect_2p5d.csv", index=False)
    s = meta.set_index("case").p_2p5d
    lo_fill = {"cc_psz": -1.0, "p_max": -1.0, "p_2p5d": -1.0}  # stage-1 failure = lowest score
    rng, rows = np.random.default_rng(0), []
    for pos_name, pos in (("CBCT-SEG cancer (thesis)", cb), ("MSD PDAC (context)", msd)):
        d = pd.concat([pos.assign(y=1), nih.assign(y=0)], ignore_index=True)
        d["p_2p5d"] = d.case.map(s)
        for col in ("cc_psz", "p_max", "p_2p5d"):
            p = d[col].fillna(lo_fill[col]).values
            a, lo, hi = auc_ci(d.y.values, p, rng)
            rows.append({"positives": pos_name, "controls": "NIH Pancreas-CT (PANORAMA external)", "model": col,
                         "n_pos": int(d.y.sum()), "n_neg": int((1 - d.y).sum()),
                         "stage1_failures": int(d.stage1_failed.fillna(False).sum()), "auroc": a, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/thesis_control.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
