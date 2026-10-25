"""Thesis-cohort control, part 2: retrain our own pipeline ON the thesis's two-source set, then cross-test.

Question: was the thesis's AUROC 0.9999 a property of the old model (full slices, no crop) or of the data (all
cancers from Pancreatic-CT-CBCT-SEG, all controls from NIH Pancreas-CT)? Our 2.5D ERM recipe is retrained on the
same predicted-mask tight crops as tools/thesis_control_eval.py (40 cancers vs 80 NIH controls), then:
    train      patient-level stratified 5-fold CV x 3 seeds -> out-of-fold AUROC on the thesis set
               (recipe as tools/panel_train.py: ResNet18 2.5D, lr 3e-4, batch 8, last-3-epoch soup; epochs 40 so the
               optimisation steps roughly match the PANORAMA runs; the per-slice lesion loss is off, no lesion masks)
    crosstest  the 15 thesis-trained models on PANORAMA (1,964 Dutch, PDAC vs non-PDAC) and MSD (PDAC vs other
               pancreatic tumours): if they learned the collection rather than cancer, they collapse here
    sizematched  the same recipe on 96 random PANORAMA Dutch scans (3 repeats) vs single thesis-trained models, on the
               same held-out Dutch scans and MSD: separates the shortcut from the small training set
    probe      can simple crop statistics alone (intensity percentiles, noise, z extent) tell the two collections
               apart? Same probe on PANORAMA for PDAC as a contrast
Run train and probe with PANEL_SLICES=data/processed/ct/thesis_cohort/slices; crosstest with the default slices.

    PANEL_SLICES=data/processed/ct/thesis_cohort/slices python tools/thesis_retrain.py train
    python tools/thesis_retrain.py crosstest
    PANEL_SLICES=data/processed/ct/thesis_cohort/slices python tools/thesis_retrain.py probe
Writes data/panel_runs/thesis_erm_s<seed>/cv<k>/soup.pt and reports/thesis_control/retrain_*.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_train as pt  # noqa: E402

OUT = "reports/thesis_control"
RUNS = "data/panel_runs"
EPOCHS, SOUP, N_BOOT = 40, 3, 2000


def thesis_cohort():
    cb = pd.read_csv(f"{OUT}/detect_3d_cbct_seg.csv")
    cb = cb[~cb.stage1_failed][["case"]].assign(y=1, source="CBCT-SEG")
    ext = pd.read_csv("reports/nnunet_summaries/external/external_cases.csv")
    nih = ext[ext.source == "NIH"][["case"]].assign(y=0, source="NIH")
    return pd.concat([cb, nih], ignore_index=True).assign(group="x", scanner="x")


def train_fold(tr, te, seed, k, dev, run="thesis_erm"):
    rdir = f"{RUNS}/{run}_s{seed}/cv{k}"
    os.makedirs(rdir, exist_ok=True)
    args = argparse.Namespace(depth=20, batch=8, workers=3)
    model = pt.Net("resnet18")
    if os.path.exists(f"{rdir}/soup.pt"):
        model.load_state_dict(torch.load(f"{rdir}/soup.pt", map_location="cpu"))
        return pt.predict(model.to(dev), te, args, dev)
    torch.manual_seed(seed * 100 + k)
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = EPOCHS * (len(tr) // 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    pos_w = torch.tensor((1 - tr.y.mean()) / tr.y.mean(), device=dev)
    ds, keep = pt.Scans(tr, 20, True), []
    for ep in range(EPOCHS):
        g = torch.Generator().manual_seed(seed * 1000 + k * 100 + ep)
        dl = torch.utils.data.DataLoader(ds, batch_size=8, sampler=torch.utils.data.RandomSampler(ds, generator=g),
                                         num_workers=3, worker_init_fn=pt.worker_init, drop_last=True, pin_memory=True)
        model.train()
        for x, y, _, _ in dl:
            x, y = pt.augment(x.to(dev, non_blocking=True)), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(x)[0].float(), y, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        if ep >= EPOCHS - SOUP:
            keep.append({kk: v.detach().float().cpu().clone() for kk, v in model.state_dict().items()})
    ref = model.state_dict()
    soup = {kk: (sum(s[kk] for s in keep) / len(keep)).to(ref[kk].dtype) for kk in ref}
    torch.save(soup, f"{rdir}/soup.pt")
    model.load_state_dict(soup)
    return pt.predict(model, te, args, dev)


def train():
    d, dev, rows = thesis_cohort(), "cuda", []
    for seed in range(3):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(d, d.y)):
            p = train_fold(d.iloc[tr], d.iloc[te], seed, k, dev)
            rows.append(d.iloc[te].assign(seed=seed, fold=k, p=p))
            print(f"seed {seed} fold {k}: fold AUROC {roc_auc_score(d.y.iloc[te], p):.3f}", flush=True)
    oof = pd.concat(rows)
    oof.to_csv(f"{OUT}/retrain_oof.csv", index=False)
    per_seed = oof.groupby("seed").apply(lambda g: roc_auc_score(g.y, g.p))
    print("out-of-fold AUROC per seed:", per_seed.round(4).to_dict(), flush=True)


def auc_ci(y, p, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                        np.r_[p[rng.choice(pos, len(pos))], p[rng.choice(neg, len(neg))]]) for _ in range(N_BOOT)]
    return roc_auc_score(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def crosstest():
    dev, args = "cuda", argparse.Namespace(depth=20, batch=8, workers=3)
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["case", "scanner", "y"]].assign(cohort="dutch")
    ext = pd.read_csv("reports/nnunet_summaries/external/external_cases.csv")
    msd = ext[ext.source == "MSD"].rename(columns={"source": "scanner"})[["case", "scanner", "y"]].assign(cohort="MSD")
    m = pd.concat([cv, msd], ignore_index=True).assign(group="x")
    ps = []
    for seed in range(3):
        for k in range(5):
            model = pt.Net("resnet18").to(dev)
            model.load_state_dict(torch.load(f"{RUNS}/thesis_erm_s{seed}/cv{k}/soup.pt", map_location=dev))
            ps.append(pt.predict(model, m, args, dev))
            print(f"  thesis model s{seed} cv{k} scored PANORAMA", flush=True)
    m["p_thesis_trained"] = np.mean(ps, 0)
    m.to_csv(f"{OUT}/retrain_crosstest_scores.csv", index=False)
    rng, rows = np.random.default_rng(0), []
    for name, g in [("PANORAMA Dutch, PDAC vs non-PDAC", m[m.cohort == "dutch"]),
                    *[(f"PANORAMA Dutch {s}", m[(m.cohort == "dutch") & (m.scanner == s)]) for s in ["Siemens", "Toshiba", "Philips"]],
                    ("MSD, PDAC vs other pancreatic tumour", m[m.cohort == "MSD"])]:
        a, lo, hi = auc_ci(g.y.values, g.p_thesis_trained.values, rng)
        rows.append({"test_set": name, "n": len(g), "pos": int(g.y.sum()), "auroc": a, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/retrain_crosstest.csv", index=False)
    print(res.round(3).to_string(index=False))


def sizematched():
    """Is 0.659 a shortcut or just 96 training scans? Train the same recipe on 96 random PANORAMA Dutch scans
    (32 PDAC + 64 non-PDAC, the thesis folds' mix), 3 repeats, and compare single models with single thesis-trained
    models (fold 0 of each seed, also 96 scans) on the same held-out Dutch scans and MSD."""
    dev, args = "cuda", argparse.Namespace(depth=20, batch=8, workers=3)
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["case", "scanner", "y"]]
    ext = pd.read_csv("reports/nnunet_summaries/external/external_cases.csv")
    msd = ext[ext.source == "MSD"].rename(columns={"source": "scanner"})[["case", "scanner", "y"]]
    rng, rows = np.random.default_rng(0), []
    for rep in range(3):
        r = np.random.default_rng(100 + rep)
        tr = pd.concat([cv[cv.y == 1].sample(32, random_state=int(r.integers(10**9))),
                        cv[cv.y == 0].sample(64, random_state=int(r.integers(10**9)))]).assign(group="x")
        test = pd.concat([cv[~cv.case.isin(tr.case)].assign(set="dutch_heldout"), msd.assign(set="MSD")]).assign(group="x")
        p_pan = train_fold(tr, test, rep, 0, dev, run="sizematched_panorama")
        model = pt.Net("resnet18").to(dev)
        model.load_state_dict(torch.load(f"{RUNS}/thesis_erm_s{rep}/cv0/soup.pt", map_location=dev))
        p_th = pt.predict(model, test, args, dev)
        for s in ("dutch_heldout", "MSD"):
            m = (test.set == s).values
            for name, p in (("panorama_96", p_pan), ("thesis_96", p_th)):
                a, lo, hi = auc_ci(test.y.values[m], p[m], rng)
                rows.append({"repeat": rep, "train": name, "test": s, "n": int(m.sum()), "auroc": a, "lo": lo, "hi": hi})
        print(pd.DataFrame(rows[-4:]).round(3).to_string(index=False), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/retrain_sizematched.csv", index=False)
    print(res.groupby(["train", "test"]).auroc.agg(["mean", "std"]).round(3))


def crop_stats(case):
    x = np.load(f"{pt.SLICES}/{case}_img.npy").astype(np.float32)
    fill = x.min()
    body = x[x > fill + 0.05]
    lap = np.abs(x[:, 1:-1, 1:-1] * 4 - x[:, :-2, 1:-1] - x[:, 2:, 1:-1] - x[:, 1:-1, :-2] - x[:, 1:-1, 2:])
    return [x.shape[0], (x <= fill + 0.05).mean(), *np.percentile(body, [10, 50, 90]), body.std(), np.median(lap)]


def probe():
    d = thesis_cohort()
    X = np.array([crop_stats(c) for c in d.case])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(lr, X, d.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    rows = [{"task": "thesis set: CBCT-SEG vs NIH from crop statistics only", "n": len(d), "auroc": roc_auc_score(d.y, p)}]
    pt.SLICES = "data/processed/ct/panel_slices"
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")
    Xp = np.array([crop_stats(c) for c in cv.case])
    pp = cross_val_predict(lr, Xp, cv.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    rows.append({"task": "PANORAMA Dutch: PDAC vs non-PDAC from crop statistics only", "n": len(cv), "auroc": roc_auc_score(cv.y, pp)})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/retrain_probe.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(2)
    {"train": train, "crosstest": crosstest, "probe": probe, "sizematched": sizematched}[sys.argv[1]]()
