"""Mitigation panel: 2.5D scan classifier on the tight crops (data from tools/panel_build_slices.py).

Model: ImageNet-pretrained 2D backbone on slice triplets (slices z-1, z, z+1 as RGB) at D evenly spaced
depths, attention pooling over depth -> scan logit, plus an auxiliary per-slice lesion logit.
Arms (scanner manufacturer = domain):
    erm   standard training (the bar)
    gbs   group-balanced sampling: every scanner x label cell drawn equally often
    grl   DANN: scanner head on the pooled feature behind a gradient-reversal layer
    mae   ERM from the CT masked-image-pretrained backbone (tools/panel_mae.py) instead of ImageNet
Splits: cv (the tight 5-fold CV folds; test fold k) or lomo (test = scanner k of Siemens/Toshiba/Philips).
Of the training cases, 15 % (stratified by scanner x label) are an inner validation set for the
pre-registered selection rule (mean over scanners of the inner-validation AUROC). The reported model is a
uniform soup of the last SOUP epochs. Test-fold and external (MSD/NIH) scores are written per case.
Planted-shortcut control (--planted p): a faint marker is added to training and inner-validation scans with
P(marker | PDAC) = p and P(marker | no PDAC) = 1 - p, and the marker (not the scanner) is the domain label of
gbs / grl. Test scans are scored clean (test_*.csv) and with the marker on exactly the non-PDAC scans
(test_flip_*.csv): a model that uses the marker collapses on the flipped set.

    python tools/panel_train.py --arm erm --fold 0 [--split cv] [--seed 0] [--smoke]
Checkpoints every epoch under data/panel_runs/<run>/ and resumes from the last one (power cuts).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SLICES = os.environ.get("PANEL_SLICES", "data/processed/ct/panel_slices")  # env, so DataLoader workers see it too
RUNS = "data/panel_runs"
REPORTS = "reports/panel"
SCANNERS = ["Siemens", "Toshiba", "Philips"]
DOMAINS = ["Siemens", "Toshiba", "Philips", "Other"]
SOUP = 3
# planted-shortcut control: a faint square (+1 SD of the normalised intensity, 16 x 16 mm) in the top-left corner
MARK_BOX, MARK_VALUE = slice(4, 20), 1.0


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["erm", "gbs", "grl", "mae"], default="erm")
    p.add_argument("--split", choices=["cv", "lomo"], default="cv")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--aux", type=float, default=0.5, help="weight of the per-slice lesion loss")
    p.add_argument("--grl", type=float, default=1.0, help="max weight of the reversed scanner loss")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--tag", default="")
    p.add_argument("--planted", type=float, default=0.0, help="planted-shortcut control: P(marker | PDAC), 0 = off")
    p.add_argument("--smoke", action="store_true", help="40 train / 20 val / 20 test cases, 2 epochs")
    p.add_argument("--rescore-ext", action="store_true", help="finished run: rewrite the external scores from soup.pt")
    return p.parse_args()


# ---------------------------------------------------------------- data
class Scans(torch.utils.data.Dataset):
    def __init__(self, meta, depth, train):
        self.m, self.depth, self.train = meta.reset_index(drop=True), depth, train

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        r = self.m.iloc[i]
        img = np.load(f"{SLICES}/{r.case}_img.npy", mmap_mode="r")
        les = np.load(f"{SLICES}/{r.case}_les.npy")
        z = img.shape[0]
        step = (z - 3) / self.depth
        off = np.random.uniform(0, step) if self.train else step / 2
        c = np.clip(np.round(1 + off + step * np.arange(self.depth)).astype(int), 1, z - 2)
        x = np.stack([img[c - 1], img[c], img[c + 1]], 1).astype(np.float32)  # (D, 3, H, W)
        planted = "marker" in self.m.columns
        if planted and r.marker:
            x[:, :, MARK_BOX, MARK_BOX] += MARK_VALUE
        dom = int(r.marker) if planted else (DOMAINS.index(r.group) if r.group in DOMAINS else 0)
        return (torch.from_numpy(x), torch.tensor(float(r.y)), torch.from_numpy((les[c] > 0).astype(np.float32)),
                torch.tensor(dom))


def worker_init(_):
    torch.set_num_threads(1)
    np.random.seed((torch.initial_seed() + _) % 2**32)


def augment(x):
    """GPU augmentation per scan (same for all its slices): small affine + contrast/brightness."""
    b = x.shape[0]
    ang = (torch.rand(b, device=x.device) - 0.5) * np.deg2rad(20)
    sc = 1 + (torch.rand(b, device=x.device) - 0.5) * 0.2
    tr = (torch.rand(b, 2, device=x.device) - 0.5) * 0.1
    th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tr[:, 0]], 1),
                      torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), tr[:, 1]], 1)], 1)
    d = x.shape[1]
    flat = x.flatten(0, 1)
    grid = F.affine_grid(th.repeat_interleave(d, 0), flat.shape, align_corners=False)
    fill = flat.amin()
    flat = F.grid_sample(flat - fill, grid, align_corners=False) + fill
    x = flat.view_as(x)
    gain = 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) - 0.5) * 0.2
    bias = (torch.rand(b, 1, 1, 1, 1, device=x.device) - 0.5) * 0.2
    return x * gain + bias


# ---------------------------------------------------------------- model
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        m = getattr(torchvision.models, backbone)(weights="DEFAULT")
        if hasattr(m, "fc"):
            dim, m.fc = m.fc.in_features, nn.Identity()
        else:
            dim, m.classifier = m.classifier[-1].in_features, nn.Identity()
        self.backbone = m
        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Linear(dim, 1)
        self.slice_head = nn.Linear(dim, 1)
        self.dom_head = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, len(DOMAINS)))

    def forward(self, x, lam=0.0):
        b, d = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(b, d, -1)
        a = torch.softmax(self.attn(f), 1)
        z = (a * f).sum(1)
        return self.head(z).squeeze(-1), self.slice_head(f).squeeze(-1), self.dom_head(GradReverse.apply(z, lam))


# ---------------------------------------------------------------- helpers
def splits(args):
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["fold", "case", "scanner", "y"]]
    meta = pd.read_csv(f"{SLICES}/meta.csv")[["case", "group"]]
    d = cv.merge(meta, on="case")
    test = d[d.fold == args.fold] if args.split == "cv" else d[d.scanner == SCANNERS[args.fold]]
    rest = d.drop(test.index)
    tr, va = train_test_split(rest, test_size=0.15, random_state=args.seed,
                              stratify=rest.scanner + rest.y.astype(str))
    ext = pd.read_csv("reports/nnunet_summaries/external/external_cases.csv")
    ext = ext.rename(columns={"source": "group"}).assign(scanner=lambda e: e.group, fold=-1)
    if args.smoke:
        tr, va, test, ext = (pd.concat([x[x.y == 0].head(n // 2), x[x.y == 1].head(n // 2)])
                             for x, n in ((tr, 40), (va, 20), (test, 20), (ext, 20)))
    if args.planted:
        rng = np.random.default_rng(1000 + args.seed)
        tr, va = (x.assign(marker=rng.random(len(x)) < np.where(x.y == 1, args.planted, 1 - args.planted))
                  for x in (tr, va))
    return tr, va, test, ext


def predict(model, meta, args, dev):
    dl = torch.utils.data.DataLoader(Scans(meta, args.depth, False), batch_size=args.batch,
                                     num_workers=args.workers, worker_init_fn=worker_init,
                                     persistent_workers=False)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, *_ in dl:
            out.append(torch.sigmoid(model(x.to(dev, non_blocking=True))[0]).float().cpu())
    return np.concatenate(out)


def mean_scanner_auc(meta, p):
    v = [roc_auc_score(meta.y[meta.scanner == s], p[(meta.scanner == s).values])
         for s in SCANNERS if meta.y[meta.scanner == s].nunique() == 2]
    return float(np.mean(v)) if v else float("nan")


def main():
    args = get_args()
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    plant = f"_plant{args.planted:g}" if args.planted else ""
    run = f"{args.arm}_{args.backbone}_d{args.depth}{plant}{args.tag}_s{args.seed}" + ("_smoke" if args.smoke else "")
    rdir = f"{RUNS}/{run}/{args.split}{args.fold}"
    if args.rescore_ext:
        ext = splits(args)[3]
        model = Net(args.backbone).cuda()
        model.load_state_dict(torch.load(f"{rdir}/soup.pt", map_location="cuda"))
        ext.assign(p=predict(model, ext, args, "cuda"), split=f"{args.split}{args.fold}")[
            ["split", "fold", "case", "scanner", "y", "p"]].to_csv(
            f"{REPORTS}/{run}/ext_{args.split}{args.fold}.csv", index=False)
        print(f"{run} {args.split}{args.fold}: external rescored", flush=True)
        return
    if os.path.exists(f"{REPORTS}/{run}/summary_{args.split}{args.fold}.json"):
        print(f"{run} {args.split}{args.fold}: already done", flush=True)
        return
    os.makedirs(rdir, exist_ok=True)
    lock = f"{rdir}/.running.pid"
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read() or 0)):
        sys.exit(f"already running (pid {open(lock).read()})")
    open(lock, "w").write(str(os.getpid()))
    try:
        train(args, run, rdir)
    finally:
        os.remove(lock)


def train(args, run, rdir):
    dev = "cuda"
    tr, va, test, ext = splits(args)
    print(f"{run} {args.split}{args.fold}: train {len(tr)} val {len(va)} test {len(test)} ext {len(ext)}", flush=True)
    model = Net(args.backbone)
    if args.arm == "mae":
        model.backbone.load_state_dict(torch.load(f"{RUNS}/mae_pretrain/encoder.pt", map_location="cpu"))
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    epochs = 2 if args.smoke else args.epochs
    steps = epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    start, hist = 0, []
    ck = f"{rdir}/last.pt"
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        start, hist = s["epoch"] + 1, s["hist"]
        print(f"resumed after epoch {s['epoch']}", flush=True)
    if args.arm == "gbs":
        cell = (tr.marker.astype(str) if args.planted else tr.scanner) + tr.y.astype(str)
        w = torch.tensor((1.0 / cell.map(cell.value_counts())).values, dtype=torch.double)
    pos_w = torch.tensor((1 - tr.y.mean()) / tr.y.mean(), device=dev)
    ds = Scans(tr, args.depth, True)
    step = start * (len(tr) // args.batch)
    for ep in range(start, epochs):
        g = torch.Generator().manual_seed(args.seed * 1000 + ep)
        sampler = (torch.utils.data.WeightedRandomSampler(w, len(tr), generator=g) if args.arm == "gbs"
                   else torch.utils.data.RandomSampler(ds, generator=g))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                                         worker_init_fn=worker_init, drop_last=True, pin_memory=True)
        model.train()
        t0, tot = time.time(), []
        for x, y, ys, dom in dl:
            x = augment(x.to(dev, non_blocking=True))
            y, ys, dom = y.to(dev), ys.to(dev), dom.to(dev)
            prog = step / steps
            lam = args.grl * (2 / (1 + np.exp(-10 * prog)) - 1) if args.arm == "grl" else 0.0
            with torch.autocast("cuda", dtype=torch.float16):
                logit, slog, dlog = model(x, lam)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_w)
                loss = loss + args.aux * F.binary_cross_entropy_with_logits(slog.float(), ys)
                if args.arm == "grl":
                    loss = loss + F.cross_entropy(dlog.float(), dom)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tot.append(loss.item())
        pv = predict(model, va, args, dev)
        rec = {"epoch": ep, "loss": float(np.mean(tot)), "val_auc": float(roc_auc_score(va.y, pv)),
               "val_mean_scanner_auc": mean_scanner_auc(va, pv), "sec": round(time.time() - t0),
               "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if ep >= epochs - SOUP:
            torch.save(model.state_dict(), f"{rdir}/ep{ep}.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "hist": hist}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
    # uniform soup of the last SOUP epochs
    sds = [torch.load(f"{rdir}/ep{e}.pt", map_location=dev) for e in range(max(0, epochs - SOUP), epochs)]
    soup = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) for k in sds[0]}
    model.load_state_dict(soup)
    torch.save(soup, f"{rdir}/soup.pt")
    pv = predict(model, va, args, dev)
    out = f"{REPORTS}/{run}"
    os.makedirs(out, exist_ok=True)
    evals = [("test", test), ("ext", ext)]
    if args.planted:
        evals.append(("test_flip", test.assign(marker=test.y == 0)))
    for name, m in evals:
        p = predict(model, m, args, dev)
        m.assign(p=p, split=f"{args.split}{args.fold}")[["split", "fold", "case", "scanner", "y", "p"]].to_csv(
            f"{out}/{name}_{args.split}{args.fold}.csv", index=False)
    summ = {"run": run, "split": f"{args.split}{args.fold}", "args": vars(args), "hist": hist,
            "soup_val_auc": float(roc_auc_score(va.y, pv)), "soup_val_mean_scanner_auc": mean_scanner_auc(va, pv)}
    json.dump(summ, open(f"{out}/summary_{args.split}{args.fold}.json", "w"), indent=1)
    print("DONE", json.dumps({k: summ[k] for k in ("soup_val_auc", "soup_val_mean_scanner_auc")}), flush=True)


if __name__ == "__main__":
    main()
