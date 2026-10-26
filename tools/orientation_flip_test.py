"""Causal test: does the 2.5D ERM classifier read the inverted cranio-caudal orientation as PDAC?

Inference only, with the saved ERM soups (3 seeds x 5 CV folds; each case scored by the fold model that did not see it):
  - the inverted scans (tools/orientation_check.py) are flipped head-to-foot, i.e. put the right way up;
  - a random sample of correctly oriented scans (150 PDAC, 150 non-PDAC) is flipped upside down.
If the orientation is used as a shortcut, flipping should move non-PDAC scores down (inverted -> fixed) and up
(correct -> flipped). Slice order does not matter to the model (attention pooling over depth); the flip reverses the
order of the three adjacent slices inside each input triplet.

    python tools/orientation_flip_test.py
Writes reports/deployment_roi/orientation_flip_test.csv (per case: original and flipped seed-averaged score).
"""
import os
import sys

import numpy as np
import pandas as pd

FLIP_DIR = "data/processed/ct/panel_slices_flip"
SRC = "data/processed/ct/panel_slices"
os.environ["PANEL_SLICES"] = FLIP_DIR  # read by panel_train at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psutil  # noqa: E402
import torch  # noqa: E402

from eval_cohort import eval_set  # noqa: E402
from panel_train import Net, predict  # noqa: E402

RUNS = "data/panel_runs"
OUT = "reports/deployment_roi/orientation_flip_test.csv"


class A:
    depth, batch, workers = 20, 8, 2


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2)
    o = pd.read_csv("reports/deployment_roi/orientation_check.csv")
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["fold", "case", "scanner", "y"]]
    d = eval_set(cv.merge(o[["case", "orientation"]], on="case"))
    inv = d[d.orientation == "inverted"].assign(test="inverted_flipped_upright")
    ok = d[d.orientation == "header_ok"]
    ok = pd.concat([ok[ok.y == c].sample(150, random_state=0) for c in (0, 1)]).assign(test="correct_flipped_upside_down")
    sel = pd.concat([inv, ok]).reset_index(drop=True).assign(group=lambda x: x.scanner)  # Scans() reads group
    os.makedirs(FLIP_DIR, exist_ok=True)
    for c in sel.case:
        for kind in ("img", "les"):
            dst = f"{FLIP_DIR}/{c}_{kind}.npy"
            if not os.path.exists(dst):
                np.save(dst, np.ascontiguousarray(np.load(f"{SRC}/{c}_{kind}.npy")[::-1]))
    print(f"{len(sel)} cases flipped into {FLIP_DIR}", flush=True)

    orig = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/test_cv{k}.csv")
                     for s in range(3) for k in range(5)).groupby("case").p.mean()
    flipped = {c: [] for c in sel.case}
    for s in range(3):
        for k in range(5):
            part = sel[sel.fold == k]
            if part.empty:
                continue
            model = Net("resnet18").cuda()
            model.load_state_dict(torch.load(f"{RUNS}/erm_resnet18_d20_s{s}/cv{k}/soup.pt", map_location="cuda"))
            for c, p in zip(part.case, predict(model, part, A, "cuda")):
                flipped[c].append(float(p))
            print(f"seed {s} fold {k}: {len(part)} scored", flush=True)
            del model
            torch.cuda.empty_cache()
    sel["p_original"] = sel.case.map(orig)
    sel["p_flipped"] = sel.case.map(lambda c: np.mean(flipped[c]))
    sel[["test", "case", "scanner", "y", "fold", "orientation", "p_original", "p_flipped"]].to_csv(OUT, index=False)
    for (t, y), g in sel.groupby(["test", "y"]):
        print(f"{t:28s} y={y} n={len(g):3d} median score {g.p_original.median():.3f} -> {g.p_flipped.median():.3f}")
    print(OUT)


if __name__ == "__main__":
    main()
