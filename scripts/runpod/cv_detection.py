"""Naive (random-split) CV detection AUROC vs within-scanner -> the confound tax.

CV folds are random/mixed-scanner, so a single fold's model is validated on a
multi-scanner set where the between-scanner prevalence shortcut IS available.
  (A) per-fold mixed-scanner AUROC  -- shortcut available, single model (clean).
  (B) within-scanner AUROC          -- shortcut removed, same in-distribution models.
(A) - (B) isolates the between-scanner confound (domain-shift held fixed: both
in-distribution, unlike LOMO which also shifts domain).
Score = summed foreground softmax (p_sum). Reads only.
"""
import json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CV = "/workspace/nnunet_results/Dataset700_PanoramaPDAC/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
LOMO_SPLITS = "/workspace/splits_final_lomo.json"
NAMES = ["Siemens", "Toshiba", "Philips"]


def rankdata_avg(a):
    a = np.asarray(a, float); sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[sorter] = np.arange(len(a))
    a_s = a[sorter]; obs = np.r_[True, a_s[1:] != a_s[:-1]]
    dense = obs.cumsum()[inv]; count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auroc(y, s):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata_avg(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap(y, s):
    o = np.argsort(-np.asarray(s, float), kind="mergesort"); y = np.asarray(y)[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); prec = tp / (tp + fp); rec = tp / y.sum()
    return float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec))


def score_case(path):
    try:
        fg = np.load(path)["probabilities"][1]; return (float(fg.max()), float(fg.sum()))
    except Exception as e:
        return ("ERR", str(e))


sp = json.load(open(LOMO_SPLITS))
man = {}
for i, nm in enumerate(NAMES):
    for c in sp[i]["val"]:
        man[c] = nm

folds, cases_all, ys, psums = [], [], [], []
for f in range(5):
    d = f"{CV}/fold_{f}/validation"
    s = json.load(open(d + "/summary.json"))
    lab = {os.path.basename(c["prediction_file"]).replace(".nii.gz", ""):
           (1 if c["metrics"]["1"]["n_ref"] > 0 else 0) for c in s["metric_per_case"]}
    cs = sorted(lab); t = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(score_case, [f"{d}/{c}.npz" for c in cs]))
    for c, r in zip(cs, res):
        if r[0] == "ERR": continue
        folds.append(f); cases_all.append(c); ys.append(lab[c]); psums.append(r[1])
    print(f"  fold {f} read {len(cs)} cases ({time.time()-t:.0f}s)", flush=True)

folds = np.array(folds); y = np.array(ys); psum = np.array(psums)
manu = np.array([man.get(c, "Other") for c in cases_all])

print("\n(A) Per-fold MIXED-scanner detection AUROC (shortcut AVAILABLE, in-distribution):")
aucs = []
for f in range(5):
    m = folds == f; a = auroc(y[m], psum[m]); aucs.append(a)
    print(f"  fold {f}: n={m.sum():4d} pos={int(y[m].sum()):3d} AUROC={a:.3f} AP={ap(y[m],psum[m]):.3f}")
print(f"  >>> MEAN per-fold mixed AUROC = {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")

print("\n(B) WITHIN-scanner detection AUROC (shortcut REMOVED, same in-distribution models):")
b_perfold_means = []
for nm in NAMES + ["Other"]:
    mm = manu == nm
    pooled = auroc(y[mm], psum[mm])
    # per-fold-averaged within this scanner (single-model basis), cells with >=8 pos & >=8 neg
    cell = []
    for f in range(5):
        m = mm & (folds == f)
        if int(y[m].sum()) >= 8 and int((1 - y[m]).sum()) >= 8:
            cell.append(auroc(y[m], psum[m]))
    pf = np.nanmean(cell) if cell else float("nan")
    if nm in NAMES: b_perfold_means.append(pf)
    print(f"  {nm:7s}: n={mm.sum():4d} pos={int(y[mm].sum()):3d} base={y[mm].mean():.3f}  "
          f"AUROC(pooled-OOF)={pooled:.3f}  AUROC(per-fold avg)={pf:.3f}  [{len(cell)} folds]")

mixed = np.nanmean(aucs); within = np.nanmean(b_perfold_means)
print(f"\n>>> CONFOUND TAX (in-distribution): mixed {mixed:.3f} - within-scanner {within:.3f} = {mixed-within:+.3f} AUROC")
print("Reference: LOMO within-scanner (out-of-distribution) Siemens 0.641 / Toshiba 0.711 / Philips 0.588;"
      "\nscanner-only metadata baseline 0.71.")
print("CV_DETECTION_DONE")
