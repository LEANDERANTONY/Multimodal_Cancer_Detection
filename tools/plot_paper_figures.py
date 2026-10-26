"""Manuscript figures not covered by the per-experiment plotting tools.

    fig1_study_design.png               cohort -> region of interest -> models -> evaluation -> mitigation panel
    supp_totalseg_failure_geometry.png  TotalSegmentator (roi_subset pancreas) empty-mask rate vs scan coverage below
                                        the pancreas, PDAC vs non-PDAC
    supp_subgroups.png                  AUROC with DeLong CI by scanner, sex and age band for the three detectors
    supp_calibration.png                reliability curves: 2.5D ERM (Dutch CV, external) and 3D p_max (Dutch CV)
    supp_thesis_case_study.png          thesis two-collection set (not in paper 1; kept for a possible second paper)

    python tools/plot_paper_figures.py
Writes figures/panorama/. Reads only committed result tables in reports/. Dutch scores use the patient-disjoint
evaluation set (tools/eval_cohort.py).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cohort import eval_set  # noqa: E402
from panel_stats import ece  # noqa: E402

OUT = "figures/panorama"
INK, MUTED, GRID = "#2b2b29", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a7974"
FILL = {"data": "#e8f1fc", "roi": "#fdeee7", "model": "#e6f6ef", "eval": "#f4f3ef", "side": "#fdf4e0"}


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", lw=1,
                                ec=MUTED, fc=FILL[kind]))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.07, "\n".join(lines), ha="center", va="top", fontsize=7.6, color=INK, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=MUTED))


def fig1():
    fig, ax = plt.subplots(figsize=(15, 2.6))
    ax.set_xlim(-0.01, 1.51); ax.set_ylim(0.255, 0.495); ax.axis("off")
    box(ax, 0.01, 0.27, 0.25, 0.21, "Cohort", ["PANORAMA: 1,964 Dutch CTs (578 PDAC)", "Siemens / Toshiba / Philips / other",
        "External: 194 MSD, 80 NIH (held out)", "Scanner-only AUROC 0.70 (confound present)"], "data")
    box(ax, 0.31, 0.27, 0.25, 0.21, "Region of interest", ["Tight crop: pancreas + duct mask", "+ 100 x 50 x 15 mm per side",
        "Provided (oracle) masks for training", "Predicted masks at deployment:", "PANORAMA baseline; TotalSegmentator", "(as documented, robust_crop, + organs)"], "roi")
    box(ax, 0.61, 0.27, 0.25, 0.21, "Detection models", ["3D nnU-Net (lesion segmentation)", "score: cc_psz / p_max",
        "2.5D ResNet18 classifier", "(slice triplets, attention pooling)", "tuned ERM, 3 seeds"], "model")
    box(ax, 0.91, 0.27, 0.27, 0.21, "Evaluation", ["5-fold cross-validation; LOMO", "scored on 1,939 single-exam patients",
        "External MSD / NIH", "Confound tax = pooled - within-scanner AUROC", "DeLong / patient bootstrap CIs; ECE, Brier"], "eval")
    box(ax, 1.22, 0.27, 0.27, 0.21, "Mitigation panel (2.5D)", ["ERM vs group-balanced, DFR,", "GRL (DANN), MAE pretraining",
        "One selection rule, CV + LOMO + external", "Planted-shortcut positive control", "(marker on 80 % PDAC / 20 % non-PDAC)"], "model")
    for x in (0.26, 0.56, 0.86, 1.18):
        arrow(ax, x + 0.004, 0.375, x + 0.046, 0.375)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_study_design.png", dpi=200, facecolor="white")


def failure_geometry():
    d = pd.read_csv("reports/deployment_roi/totalseg_failure_features.csv")
    d["empty"] = d["empty"].astype(str).str.lower().eq("true")
    bins = [0, 50, 100, 150, 200, 300, 2000]
    labels = ["<50", "50-100", "100-150", "150-200", "200-300", ">300"]
    d["below"] = pd.cut(d.panc_to_bottom_mm, bins, labels=labels, right=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(labels))
    for k, (y, name, c) in enumerate(((1, "PDAC", ORANGE), (0, "non-PDAC", BLUE))):
        g = d[d.y == y].groupby("below", observed=False)["empty"].agg(["mean", "size"])
        ax[0].bar(x + (k - 0.5) * 0.38, 100 * g["mean"], 0.36, color=c, label=name)
        for xi, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax[0].text(xi + (k - 0.5) * 0.38, 100 * m + 0.8, f"n={n}", ha="center", fontsize=6.5, color=MUTED)
    ax[0].set_xticks(x, labels)
    ax[0].set_xlabel("scan coverage below the pancreas (mm)", fontsize=9, color=INK)
    ax[0].set_ylabel("% scans with an empty pancreas mask", fontsize=9, color=INK)
    ax[0].set_title("a  Failure rate by coverage below the pancreas", fontsize=10, color=INK, loc="left")
    ax[0].legend(fontsize=8, frameon=False)
    style(ax[0])
    ok, bad = d[~d["empty"]], d[d["empty"]]
    ax[1].scatter(ok.panc_to_bottom_mm, ok.panc_to_top_mm, s=6, color=GRAY, alpha=0.35, label="mask found")
    ax[1].scatter(bad.panc_to_bottom_mm, bad.panc_to_top_mm, s=14, color=ORANGE, edgecolor="white", lw=0.5,
                  label=f"empty mask (n={len(bad)})")
    ax[1].set_xlabel("scan coverage below the pancreas (mm)", fontsize=9, color=INK)
    ax[1].set_ylabel("scan coverage above the pancreas (mm)", fontsize=9, color=INK)
    ax[1].set_title("b  Scan coverage above and below the pancreas", fontsize=10, color=INK, loc="left")
    ax[1].set_xlim(0, 700)
    ax[1].legend(fontsize=8, frameon=False)
    style(ax[1])
    fig.suptitle("TotalSegmentator 2.18, roi_subset = pancreas, 1,964 Dutch scans", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/supp_totalseg_failure_geometry.png", dpi=160, facecolor="white")


def subgroups():
    s = pd.read_csv("reports/final_stats/subgroups.csv")
    s = s[s.auroc_2p5d_erm.notna() & (s.level != "Other")]
    names = {"3d_cc_psz": ("3D cc_psz", BLUE), "3d_p_max": ("3D p_max", AQUA), "2p5d_erm": ("2.5D ERM", ORANGE)}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    rows = list(s.itertuples())
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for k, (m, (lab, c)) in enumerate(names.items()):
            yy = y + (1 - k) * 0.24
            ax.plot([getattr(r, f"lo_{m}"), getattr(r, f"hi_{m}")], [yy, yy], color=c, lw=2)
            ax.plot(getattr(r, f"auroc_{m}"), yy, "o", color=c, ms=6, mec="white", mew=1.5, label=lab if i == 0 else None)
    ax.set_yticks(range(len(rows)), [f"{r.variable}: {r.level} (n={r.n})" for r in rows][::-1], fontsize=8, color=INK)
    ax.set_xlabel("AUROC (95% DeLong CI), Dutch out-of-fold", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/supp_subgroups.png", dpi=160, facecolor="white")


def reliability(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    xs, ys, ns = [], [], []
    for k in range(bins):
        m = b == k
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def calibration():
    cv = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/test_cv{k}.csv") for s in range(3) for k in range(5))
    cv = eval_set(cv.groupby(["case", "y"], as_index=False).p.mean())
    ex = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/ext_cv{k}.csv") for s in range(3) for k in range(5))
    ex = ex.groupby(["case", "y"], as_index=False).p.mean()
    d3 = eval_set(pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv"))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    for (y, p, lab, c) in ((cv.y.values, cv.p.values, f"2.5D ERM, Dutch CV (ECE {ece(cv.y.values, cv.p.values):.3f})", ORANGE),
                           (ex.y.values, ex.p.values, f"2.5D ERM, external MSD+NIH (ECE {ece(ex.y.values, ex.p.values):.3f})", BLUE),
                           (d3.y.values, d3.p_max.values, "3D p_max, Dutch CV (not a probability)", GRAY)):
        xs, ys, _ = reliability(y, p)
        ax.plot(xs, ys, "-o", color=c, lw=2, ms=6, mec="white", mew=1.5, label=lab)
    ax.set_xlabel("mean predicted probability", fontsize=9, color=INK)
    ax.set_ylabel("observed PDAC fraction", fontsize=9, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/supp_calibration.png", dpi=160, facecolor="white")


def thesis_case():
    tc = pd.read_csv("reports/thesis_control/thesis_control.csv")
    tc = tc[tc.positives.str.startswith("CBCT")].set_index("model")
    oof = pd.read_csv("reports/thesis_control/retrain_oof.csv")
    from sklearn.metrics import roc_auc_score
    retr = np.mean([roc_auc_score(g.y, g.p) for _, g in oof.groupby("seed")])
    sm = pd.read_csv("reports/thesis_control/retrain_sizematched.csv")
    pan96 = sm[(sm.train == "panorama_96") & (sm.test == "dutch_heldout")].auroc.mean()
    bars = [("thesis model\n(trained on the set)", 0.9999, GRAY), ("PANORAMA 3D\ncc_psz", tc.loc["cc_psz", "auroc"], BLUE),
            ("PANORAMA 3D\np_max", tc.loc["p_max", "auroc"], AQUA), ("PANORAMA 2.5D\nERM", tc.loc["p_2p5d", "auroc"], ORANGE),
            ("2.5D recipe retrained\non the set (OOF)", retr, YELLOW), ("same recipe, 96 PANORAMA\nscans (held-out Dutch)", pan96, YELLOW)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, 0.62, color=c, alpha=0.55 if i == 5 else 1)
        ax.text(i, v + 0.01, f"{v:.4f}" if v > 0.999 else f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7.5, color=INK)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("AUROC", fontsize=9, color=INK)
    ax.axvline(4.5, color=MUTED, lw=0.8, ls=":")
    ax.set_title("Thesis two-collection set (40 CBCT-SEG cancers vs 80 NIH controls); last bar: same training size on "
                 "clinical data", fontsize=9, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/supp_thesis_case_study.png", dpi=160, facecolor="white")


if __name__ == "__main__":
    fig1(); failure_geometry(); subgroups(); calibration(); thesis_case()
    print("figures written to", OUT)
