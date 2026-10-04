"""Confound battery for the TIGHT model (Dataset701) — same tests that settled the loose model.

  lomo   : per-held-out-scanner detection AUROC from the tight-LOMO --npz (OOD, never-seen manufacturer)
  cv     : in-distribution confound tax = per-fold MIXED-scanner AUROC - WITHIN-scanner AUROC,
           from re-inferred tight-CV softmax (CV npz was deleted; run_battery701.sh re-predicts it)
  feat   : feature-space probe on the tight-CV fold-0 encoder (linear/RF/MLP; scanner vs cancer)

Scores: p_max, p_sum, cc_psz (PanDx peak x size^(1/15)). Per-case scores saved as CSV so nothing
needs re-inference again. Usage: python tight_battery.py {lomo|cv|feat}. Writes to /workspace/tight_battery.
"""
import json, os, sys, time, csv
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage

OUT = "/workspace/tight_battery"
LOMO_BASE = "/workspace/nnunet_results_tight_lomo/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
CV_BASE = "/workspace/nnunet_results_tight/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
CV_PRED = "/root/cvout"
IMG = "/root/nnunet_raw/Dataset701_PanoramaPDAC_tight/imagesTr"
NAMES = ["Siemens", "Toshiba", "Philips"]
SCORES = ["p_max", "p_sum", "cc_psz"]
os.makedirs(OUT, exist_ok=True)


def rankdata_avg(a):
    a = np.asarray(a, float); s = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[s] = np.arange(len(a)); asr = a[s]
    obs = np.r_[True, asr[1:] != asr[:-1]]; dense = obs.cumsum()[inv]; cnt = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def auroc(y, sc):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata_avg(sc); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(y, sc, pat, n=1000, seed=0):
    # patient-level bootstrap
    rng = np.random.default_rng(seed); up = np.unique(pat); idx = {p: np.where(pat == p)[0] for p in up}
    v = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        a = auroc(y[ii], sc[ii])
        if a == a: v.append(a)
    return np.percentile(v, [2.5, 97.5])


def score_case(path):
    try:
        fg = np.load(path)["probabilities"][1]
    except Exception:
        return None
    pmax = float(fg.max()); psum = float(fg.sum())
    m = fg > 0.5
    if m.any():
        lab, n = ndimage.label(m); idx = range(1, n + 1)
        peak = np.array(ndimage.maximum(fg, lab, idx), float); size = np.array(ndimage.sum(m, lab, idx), float)
        ccpsz = float((peak * size ** (1.0 / 15.0)).max())
    else:
        ccpsz = pmax
    return pmax, psum, ccpsz


def scanner_map():
    sp = json.load(open("/workspace/splits_final_lomo.json"))
    return {c: NAMES[i] for i in range(3) for c in sp[i]["val"]}


def labels_from(summary):
    s = json.load(open(summary))
    return {os.path.basename(c["prediction_file"]).replace(".nii.gz", ""): int(c["metrics"]["1"]["n_ref"] > 0)
            for c in s["metric_per_case"]}


def score_dir(d, cases):
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(score_case, [f"{d}/{c}.npz" for c in cases]))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "case", "scanner", "y"] + SCORES); w.writerows(rows)


def run_lomo():
    man = scanner_map(); rows = []
    print("=== TIGHT LOMO: per-held-out-scanner detection AUROC (OOD) ===", flush=True)
    for f, nm in enumerate(NAMES):
        d = f"{LOMO_BASE}/fold_{f}/validation"
        lab = labels_from(f"{d}/summary.json"); cs = sorted(lab); t = time.time()
        res = score_dir(d, cs)
        R = [(f, c, man.get(c, "Other"), lab[c]) + r for c, r in zip(cs, res) if r]
        rows += R
        y = np.array([r[3] for r in R]); pat = np.array([r[1].split("_")[0] for r in R])
        line = f"  {nm:8s} n={len(y)} pos={y.sum()} base={y.mean():.3f}"
        for i, s in enumerate(SCORES):
            sc = np.array([r[4 + i] for r in R]); a = auroc(y, sc)
            line += f"  {s}={a:.3f}"
            if s == "cc_psz":
                lo, hi = boot_ci(y, sc, pat); line += f" [{lo:.2f},{hi:.2f}]"
        print(line + f"  ({time.time()-t:.0f}s)", flush=True)
    write_csv(f"{OUT}/lomo_scores.csv", rows)
    print("LOMO_BATTERY_DONE", flush=True)


def run_cv():
    man = scanner_map(); rows = []
    print("=== TIGHT CV: confound tax (mixed-scanner vs within-scanner, in-distribution) ===", flush=True)
    for f in range(5):
        lab = labels_from(f"{CV_BASE}/fold_{f}/validation/summary.json"); cs = sorted(lab)
        res = score_dir(f"{CV_PRED}/fold_{f}", cs)
        miss = sum(r is None for r in res)
        rows += [(f, c, man.get(c, "Other"), lab[c]) + r for c, r in zip(cs, res) if r]
        print(f"  fold {f}: scored {len(cs)-miss}/{len(cs)}", flush=True)
    write_csv(f"{OUT}/cv_scores.csv", rows)
    fo = np.array([r[0] for r in rows]); y = np.array([r[3] for r in rows]); sn = np.array([r[2] for r in rows])
    for i, s in enumerate(SCORES):
        sc = np.array([r[4 + i] for r in rows])
        mixed = np.nanmean([auroc(y[fo == f], sc[fo == f]) for f in range(5)])
        within = []
        line = f"  {s:7s} mixed(per-fold)={mixed:.3f} | within:"
        for nm in NAMES:
            cell = [auroc(y[(sn == nm) & (fo == f)], sc[(sn == nm) & (fo == f)]) for f in range(5)
                    if y[(sn == nm) & (fo == f)].sum() >= 8 and (1 - y[(sn == nm) & (fo == f)]).sum() >= 8]
            w = np.nanmean(cell) if cell else float("nan"); within.append(w)
            line += f" {nm}={w:.3f}"
        line += f" | TAX = {mixed - np.nanmean(within):+.3f}"
        print(line, flush=True)
    print("CV_BATTERY_DONE", flush=True)


def run_feat():
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    pred = nnUNetPredictor(device=torch.device("cuda"), allow_tqdm=False)
    pred.initialize_from_trained_model_folder(CV_BASE, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    net = pred.network.to("cuda").eval(); pm, cm, dj = pred.plans_manager, pred.configuration_manager, pred.dataset_json
    ps = cm.patch_size; pp = DefaultPreprocessor()
    man = scanner_map(); label = {}
    for f in range(5):
        label.update(labels_from(f"{CV_BASE}/fold_{f}/validation/summary.json"))
    rng = np.random.default_rng(20261001); sel = []
    for nm in NAMES:
        cs = sorted(c for c in man if man[c] == nm and os.path.exists(f"{IMG}/{c}_0000.nii.gz"))
        rng.shuffle(cs); sel += cs[:int(os.environ.get("FEAT_N", 250))]
    X, S, L, P = [], [], [], []
    for i, c in enumerate(sel):
        data, _, _ = pp.run_case([f"{IMG}/{c}_0000.nii.gz"], None, pm, cm, dj)
        sl, pads = [slice(None)], []
        for k, p in enumerate(ps):
            dd = data.shape[k + 1]
            if dd >= p: lo = (dd - p) // 2; sl.append(slice(lo, lo + p)); pads.append((0, 0))
            else: sl.append(slice(0, dd)); b = (p - dd) // 2; pads.append((b, p - dd - b))
        patch = np.pad(data[tuple(sl)], [(0, 0)] + pads, mode="constant", constant_values=float(data.min()))
        with torch.no_grad():
            v = net.encoder(torch.from_numpy(patch[None]).float().cuda())[-1].mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()
        X.append(v); S.append(man[c]); L.append(label.get(c, -1)); P.append(c.split("_")[0])
        if (i + 1) % 150 == 0: print(f"  feat {i+1}/{len(sel)}", flush=True)
    X, S, L, P = map(np.array, (X, S, L, P))
    np.savez(f"{OUT}/feature_diag_tight.npz", X=X, scanner=S, label=L, patient=P)

    def probe(Xp, y, g, name, mk, shuffle=False):
        yy = np.random.default_rng(0).permutation(y) if shuffle else y.copy(); cl = np.unique(yy)
        oof = np.zeros((len(yy), len(cl)))
        for tr, te in StratifiedGroupKFold(5).split(Xp, yy, g):
            clf = mk(); clf.fit(Xp[tr], yy[tr]); oof[te] = clf.predict_proba(Xp[te])
        return roc_auc_score(yy, oof[:, 1]) if len(cl) == 2 else roc_auc_score(yy, oof, multi_class="ovr", average="macro")

    mks = {"linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
           "RF": lambda: RandomForestClassifier(500, min_samples_leaf=3, n_jobs=8, random_state=0),
           "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, max_iter=1000, early_stopping=True, random_state=0))}
    m = L >= 0
    print(f"=== TIGHT feature probe (fold-0 encoder bottleneck, n={len(X)}, patient-grouped 5-fold) ===", flush=True)
    for k, mk in mks.items():
        print(f"  {k:6s} scanner={probe(X, S, P, k, mk):.3f}  scanner-shuffled={probe(X, S, P, k, mk, True):.3f}  "
              f"cancer={probe(X[m], L[m], P[m], k, mk):.3f}", flush=True)
    print("FEAT_BATTERY_DONE", flush=True)


if __name__ == "__main__":
    {"lomo": run_lomo, "cv": run_cv, "feat": run_feat}[sys.argv[1]]()
