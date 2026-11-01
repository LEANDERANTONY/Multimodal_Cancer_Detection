"""Flip test on every evaluation scan: score each of the 1,939 single-examination patients head-to-foot reversed.

Inference only, with the saved ERM soups (3 seeds; each scan scored by the CV fold model that did not see it). Reversing
the stack turns reviewed-inverted scans upright and correct scans upside down. Compared with the original scores:
  - the within-scan change, per reviewed class and label (paired, so free of stream and scanner differences);
  - inverted-upright vs correct-as-stored non-PDAC scans within scanner x level: is any difference left without the
    orientation cue? (question: the residual after correcting)
  - AUROC with all inverted scans upright, and with every scan upside down.

    python tools/orientation_flip_all.py [--limit n]    (FLIP_DEVICE=cpu for a smoke test)
Writes reports/deployment_roi/orientation_flip_all.csv; resumes per seed/fold from data/processed/ct/orientation_flip_all/.
"""
import os
import sys

import numpy as np
import pandas as pd
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cohort import eval_set  # noqa: E402
from panel_train import SLICES, Net, worker_init  # noqa: E402

RUNS = "data/panel_runs"
OUT = "reports/deployment_roi/orientation_flip_all.csv"
PART = "data/processed/ct/orientation_flip_all"
DEPTH, BATCH = 20, 8
DEV = os.environ.get("FLIP_DEVICE", "cuda")


class Flipped(torch.utils.data.Dataset):
    """panel_train.Scans at evaluation, with the volume reversed along z."""

    def __init__(self, cases):
        self.cases = list(cases)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        img = np.load(f"{SLICES}/{self.cases[i]}_img.npy", mmap_mode="r")[::-1]
        z = img.shape[0]
        step = (z - 3) / DEPTH
        c = np.clip(np.round(1 + step / 2 + step * np.arange(DEPTH)).astype(int), 1, z - 2)
        return torch.from_numpy(np.stack([img[c - 1], img[c], img[c + 1]], 1).astype(np.float32))


def score(model, cases):
    dl = torch.utils.data.DataLoader(Flipped(cases), batch_size=BATCH, num_workers=2, worker_init_fn=worker_init)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16):
        for x in dl:
            out.append(torch.sigmoid(model(x.to(DEV, non_blocking=True))[0]).float().cpu())
    return np.concatenate(out)


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    a = pd.read_csv("reports/deployment_roi/orientation_adjudicated.csv")
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["case", "fold"]]
    d = eval_set(a.merge(cv, on="case"))
    assert len(d) == 1939, len(d)
    os.makedirs(PART, exist_ok=True)
    for s in range(3):
        for k in range(5):
            part = f"{PART}/s{s}_cv{k}.csv"
            todo = d[d.fold == k].case.tolist()
            if limit:
                todo = todo[:limit]
            elif os.path.exists(part):
                continue
            model = Net("resnet18").to(DEV)
            model.load_state_dict(torch.load(f"{RUNS}/erm_resnet18_d20_s{s}/cv{k}/soup.pt", map_location=DEV))
            p = score(model, todo)
            if not limit:
                pd.DataFrame({"case": todo, "p": p}).to_csv(part, index=False)
            print(f"seed {s} fold {k}: {len(todo)} scored; median {np.median(p):.3f}", flush=True)
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()
            if limit:
                print("smoke test ok", flush=True)
                return
    fl = pd.concat(pd.read_csv(f"{PART}/s{s}_cv{k}.csv") for s in range(3) for k in range(5)).groupby("case", as_index=False).p.mean()
    orig = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/test_cv{k}.csv") for s in range(3) for k in range(5))
    orig = orig.groupby("case", as_index=False).p.mean()
    m = d.merge(orig.rename(columns={"p": "p_original"}), on="case").merge(fl.rename(columns={"p": "p_flipped"}), on="case")
    m[["case", "scanner", "y", "level", "batch", "final", "p_original", "p_flipped"]].to_csv(OUT, index=False)
    print(OUT, len(m), flush=True)


if __name__ == "__main__":
    main()
