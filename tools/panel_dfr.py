"""Mitigation panel, DFR arm (deep feature reweighting, Kirichenko et al. 2023): keep each ERM soup's backbone and
attention pooling, retrain only the last layer on group x label balanced data held out from ERM training.

Per ERM run (seed) and split (cv0-4, lomo0-2): pooled scan features of the inner-validation set (never trained on),
the test fold and MSD/NIH, cached in data/panel_runs/<erm run>/<split>/feats.npz. A standardised L2 logistic
regression is fitted on balanced subsamples of the inner validation set (equal cases per group x label cell; group =
Siemens/Toshiba/Philips, or the marker in the planted-shortcut control); C is chosen by the pre-registered rule (mean
over scanners of AUROC) with a stratified 5-fold CV inside the inner validation set; the final layer is the mean of
20 balanced-subsample fits. Scores go to reports/panel/dfr_<...>/ in the same format as tools/panel_train.py, so
tools/panel_eval.py reports DFR next to the other arms.

    python tools/panel_dfr.py [--planted p] [seed ...]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import psutil
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_train import REPORTS, RUNS, SCANNERS, Net, Scans, mean_scanner_auc, splits, worker_init  # noqa: E402

BACKBONE, DEPTH = "resnet18", 20
SPLITS = [("cv", k) for k in range(5)] + [("lomo", k) for k in range(3)]
CS = [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_SUB = 20


def features(model, meta, args):
    dl = torch.utils.data.DataLoader(Scans(meta, args.depth, False), batch_size=args.batch,
                                     num_workers=args.workers, worker_init_fn=worker_init)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, *_ in dl:
            x = x.cuda(non_blocking=True)
            b, d = x.shape[:2]
            f = model.backbone(x.flatten(0, 1)).view(b, d, -1)
            a = torch.softmax(model.attn(f), 1)
            out.append((a * f).sum(1).float().cpu())
    return torch.cat(out).numpy()


def groups(meta):
    g = meta.marker.astype(int).astype(str) if "marker" in meta.columns else meta.scanner
    keep = np.ones(len(meta), bool) if "marker" in meta.columns else np.isin(meta.scanner.values, SCANNERS)
    return (g + meta.y.astype(str)).values, keep


def balanced_fit(X, meta, C, rng):
    cell, keep = groups(meta)
    cells = np.unique(cell[keep])
    n = min((cell == c).sum() for c in cells)
    ws, bs = [], []
    for _ in range(N_SUB):
        idx = np.concatenate([rng.choice(np.where(cell == c)[0], n, replace=False) for c in cells])
        lr = LogisticRegression(C=C, max_iter=5000).fit(X[idx], meta.y.values[idx])
        ws.append(lr.coef_[0]); bs.append(lr.intercept_[0])
    return np.mean(ws, 0), np.mean(bs)


def score(X, wb):
    return 1 / (1 + np.exp(-(X @ wb[0] + wb[1])))


def choose_c(X, meta, seed):
    strat = groups(meta)[0]
    res = {}
    for C in CS:
        p = np.zeros(len(meta))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
            sc = StandardScaler().fit(X[tr])
            p[te] = score(sc.transform(X[te]), balanced_fit(sc.transform(X[tr]), meta.iloc[tr], C,
                                                             np.random.default_rng(seed)))
        res[C] = mean_scanner_auc(meta, p)
    return max(res, key=res.get), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planted", type=float, default=0.0)
    ap.add_argument("seeds", type=int, nargs="*", default=[0, 1, 2])
    cli = ap.parse_args()
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(2)
    warnings.filterwarnings("ignore")
    plant = f"_plant{cli.planted:g}" if cli.planted else ""
    splits_todo = SPLITS[:5] if cli.planted else SPLITS
    for seed in cli.seeds:
        src, run = f"erm_{BACKBONE}_d{DEPTH}{plant}_s{seed}", f"dfr_{BACKBONE}_d{DEPTH}{plant}_s{seed}"
        out = f"{REPORTS}/{run}"
        os.makedirs(out, exist_ok=True)
        for split, k in splits_todo:
            if os.path.exists(f"{out}/summary_{split}{k}.json"):
                print(f"{run} {split}{k}: already done", flush=True)
                continue
            t0 = time.time()
            args = argparse.Namespace(split=split, fold=k, seed=seed, smoke=False, depth=DEPTH, batch=8, workers=2,
                                      planted=cli.planted)
            tr, va, test, ext = splits(args)  # same inner-validation split (and markers) as the ERM run
            sets = [("test", test), ("ext", ext)]
            if cli.planted:
                sets.append(("test_flip", test.assign(marker=test.y == 0)))
            cache = f"{RUNS}/{src}/{split}{k}/feats.npz"
            if os.path.exists(cache):
                F = dict(np.load(cache))
            else:
                model = Net(BACKBONE).cuda()
                model.load_state_dict(torch.load(f"{RUNS}/{src}/{split}{k}/soup.pt", map_location="cuda"))
                F = {name: features(model, m, args) for name, m in [("val", va)] + sets}
                np.savez(cache, **F)
                del model
            t1 = time.time()
            va = va.reset_index(drop=True)
            C, cres = choose_c(F["val"], va, seed)
            sc = StandardScaler().fit(F["val"])
            wb = balanced_fit(sc.transform(F["val"]), va, C, np.random.default_rng(seed))
            for name, m in sets:
                m.assign(p=score(sc.transform(F[name]), wb), split=f"{split}{k}")[
                    ["split", "fold", "case", "scanner", "y", "p"]].to_csv(f"{out}/{name}_{split}{k}.csv", index=False)
            summ = {"run": run, "split": f"{split}{k}", "source": src, "C": C, "c_search": cres, "n_val": len(va)}
            json.dump(summ, open(f"{out}/summary_{split}{k}.json", "w"), indent=1)
            print(f"{run} {split}{k}: C={C} inner-CV mean-scanner AUROC {cres[C]:.3f} "
                  f"(features {t1 - t0:.0f} s, fit {time.time() - t1:.0f} s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
