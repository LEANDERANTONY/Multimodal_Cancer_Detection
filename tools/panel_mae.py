"""Mitigation panel, MAE arm, stage 1: masked-image pretraining of the 2D backbone on CT (SimMIM-style, the masked-
autoencoder variant that suits CNNs), starting from the ImageNet weights.

Input: random slice triplets from every scan in data/processed/ct/panel_slices (Dutch, MSD and NIH; images only,
labels never read). 60 % of 32 x 32 patches are zeroed; a 1x1-conv + pixel-shuffle head on the last feature map
reconstructs the triplet, L1 loss on masked pixels only. Checkpoints every CKPT steps and resumes (power cuts).
Writes data/panel_runs/mae_pretrain/encoder.pt (backbone state dict), used by tools/panel_train.py --arm mae.

    python tools/panel_mae.py [--steps 20000] [--batch 48]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SLICES = "data/processed/ct/panel_slices"
OUT = "data/panel_runs/mae_pretrain"
PATCH, RATIO, CKPT = 32, 0.6, 1000


class Triplets(torch.utils.data.IterableDataset):
    def __init__(self, cases):
        self.cases = cases

    def __iter__(self):
        rng = np.random.default_rng(torch.initial_seed() % 2**32)
        while True:
            img = np.load(f"{SLICES}/{self.cases[rng.integers(len(self.cases))]}_img.npy", mmap_mode="r")
            c = rng.integers(1, img.shape[0] - 1)
            yield torch.from_numpy(np.ascontiguousarray(img[c - 1:c + 2]).astype(np.float32))


def one_thread(_):
    torch.set_num_threads(1)


class SimMIM(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights="DEFAULT")
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Conv2d(512, 3 * PATCH * PATCH, 1)

    def forward(self, x):
        b = self.backbone
        f = b.layer4(b.layer3(b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))))
        return F.pixel_shuffle(self.head(f), PATCH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/encoder.pt"):
        sys.exit("already done")
    lock = f"{OUT}/.running.pid"
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read() or 0)):
        sys.exit(f"already running (pid {open(lock).read()})")
    open(lock, "w").write(str(os.getpid()))
    try:
        cases = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{SLICES}/*_img.npy"))
        model = SimMIM().cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
        scaler = torch.amp.GradScaler()
        step, ck = 0, f"{OUT}/last.pt"
        if os.path.exists(ck):
            s = torch.load(ck, map_location="cuda", weights_only=False)
            model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
            sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"]); step = s["step"]
            print(f"resumed at step {step}", flush=True)
        dl = iter(torch.utils.data.DataLoader(Triplets(cases), batch_size=args.batch, num_workers=3,
                                              worker_init_fn=one_thread, pin_memory=True))
        print(f"{len(cases)} scans, steps {step}->{args.steps}, batch {args.batch}", flush=True)
        t0, losses = time.time(), []
        while step < args.steps:
            x = next(dl).cuda(non_blocking=True)
            b, _, h, w = x.shape
            mask = (torch.rand(b, 1, h // PATCH + 1, w // PATCH + 1, device="cuda") < RATIO).float()
            mask = F.interpolate(mask, scale_factor=PATCH, mode="nearest")[:, :, :h, :w]
            with torch.autocast("cuda", dtype=torch.float16):
                rec = model((x * (1 - mask)).contiguous(memory_format=torch.channels_last))
            rec = rec.float()[:, :, :h, :w]
            loss = (F.l1_loss(rec, x, reduction="none") * mask).sum() / (mask.sum() * 3 + 1e-6)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            losses.append(loss.item())
            if step % CKPT == 0 or step == args.steps:
                print(json.dumps({"step": step, "loss": round(float(np.mean(losses)), 4),
                                  "sec": round(time.time() - t0),
                                  "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
                losses, t0 = [], time.time()
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": step}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)
        torch.save(model.backbone.state_dict(), f"{OUT}/encoder.pt")
        print("DONE", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
