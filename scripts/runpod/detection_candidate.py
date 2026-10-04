"""Detection AUROC under several scoring methods, from nnU-Net --npz softmax.

Shows how much of the apparent 'gap' vs PanDx is just the detection score:
  p_max   : global max foreground prob (blunt)
  p_sum   : total foreground prob mass (blunt; what we used before)
  cc_peak : max over connected components of the component peak prob
  cc_psz  : PanDx-style peak x size^(1/15) candidate score (peak-scaled)
Per-fold mixed AUROC + pooled. Usage: python detection_candidate.py <RESULTS_BASE>
  (loose)  /workspace/nnunet_results/Dataset700_PanoramaPDAC/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
  (tight)  /workspace/nnunet_results_tight/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
Reads only.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage

BASE = sys.argv[1]
NFOLD = 5


def rankdata_avg(a):
    a = np.asarray(a, float); s = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[s] = np.arange(len(a)); asr = a[s]
    obs = np.r_[True, asr[1:] != asr[:-1]]; dense = obs.cumsum()[inv]; cnt = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def auroc(y, sc):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata_avg(sc); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def score_case(path):
    try:
        fg = np.load(path)["probabilities"][1]
    except Exception as e:
        return None
    pmax = float(fg.max()); psum = float(fg.sum())
    m = fg > 0.5
    if m.any():
        lab, n = ndimage.label(m)
        idx = range(1, n + 1)
        peak = np.array(ndimage.maximum(fg, lab, idx), float)
        size = np.array(ndimage.sum(m, lab, idx), float)
        cc_peak = float(peak.max())
        cc_psz = float((peak * size ** (1.0 / 15.0)).max())
    else:
        cc_peak = pmax; cc_psz = pmax
    return pmax, psum, cc_peak, cc_psz


SCORES = ["p_max", "p_sum", "cc_peak", "cc_psz"]
per_fold = {s: [] for s in SCORES}
pool = {s: [] for s in SCORES}; pool_y = []
for f in range(NFOLD):
    d = f"{BASE}/fold_{f}/validation"
    if not os.path.exists(d + "/summary.json"):
        print(f"fold {f}: no summary"); continue
    s = json.load(open(d + "/summary.json"))
    lab = {os.path.basename(c["prediction_file"]).replace(".nii.gz", ""):
           (1 if c["metrics"]["1"]["n_ref"] > 0 else 0) for c in s["metric_per_case"]}
    cases = sorted(lab); t = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(score_case, [f"{d}/{c}.npz" for c in cases]))
    y = np.array([lab[c] for c, r in zip(cases, res) if r]); arr = [r for r in res if r]
    cols = {s: np.array([a[i] for a in arr]) for i, s in enumerate(SCORES)}
    line = f"fold {f} (n={len(y)}, {time.time()-t:.0f}s): " + " ".join(f"{s}={auroc(y,cols[s]):.3f}" for s in SCORES)
    print(line, flush=True)
    for s in SCORES:
        per_fold[s].append(auroc(y, cols[s])); pool[s] += list(cols[s])
    pool_y += list(y)

print("\nMEAN per-fold AUROC:")
for s in SCORES:
    v = np.array(per_fold[s]); print(f"  {s:8s} {np.nanmean(v):.3f} +/- {np.nanstd(v):.3f}")
print("POOLED AUROC (cross-fold, calibration caveat):")
y = np.array(pool_y)
for s in SCORES:
    print(f"  {s:8s} {auroc(y, np.array(pool[s])):.3f}")
print("DETECTION_CANDIDATE_DONE")
