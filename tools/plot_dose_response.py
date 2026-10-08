"""Deployment-ROI dose-response figure + summary table (MSD, tight 5-fold, no TTA).

Panel A: change in MSD AUROC vs centroid shift of the oracle crop (paired bootstrap 95% CI vs the
unperturbed reference), with the two real segmenters' centroid-offset distributions underneath.
Panel B: the same vs margin scale. Labels give the share of tumours entirely inside the crop.

    python tools/plot_dose_response.py
Reads reports/deployment_roi/dose_response_msd.csv and stage1_quality_<arm>_external.csv;
writes reports/deployment_roi/dose_response_summary.csv and figures/panorama/deployment_dose_response.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

D = "reports/deployment_roi"
FIG = "figures/panorama/deployment_dose_response.png"
SCORES = {"p_max": "#2a78d6", "cc_psz": "#eb6834"}  # reference palette slots 1-2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_BOOT = 2000


def summarise(d):
    ref = d[d.condition == "ref"].set_index("case")
    rng = np.random.default_rng(0)
    cases = ref.index.values
    y = ref.y.values
    idx = [i for i in (rng.integers(0, len(cases), len(cases)) for _ in range(N_BOOT)) if 0 < y[i].sum() < len(i)]
    rows = []
    for cond, g in d.groupby("condition", sort=False):
        g = g.set_index("case").reindex(cases)
        if g.p_max.isna().any():
            continue  # condition not finished
        pos = g[g.y == 1]
        row = {"condition": cond, "shift_mm": g.shift_mm.iloc[0], "margin_scale": g.margin_scale.iloc[0],
               "crop_iou_median": g.crop_iou.median(), "lesion_fully_in": (pos.lesion_in_crop >= 0.999).mean(),
               "dice": pos.dice.fillna(0).mean()}
        for s in SCORES:
            a, r = g[s].values, ref[s].values
            row[f"auroc_{s}"] = roc_auc_score(y, a)
            diffs = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], r[i]) for i in idx]
            row[f"delta_{s}"] = row[f"auroc_{s}"] - roc_auc_score(y, r)
            row[f"delta_{s}_lo"], row[f"delta_{s}_hi"] = np.percentile(diffs, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def panel(ax, sub, x, xlabel, ref_x):
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(ref_x, color=GRID, lw=1, zorder=0)
    for s, c in SCORES.items():
        ax.fill_between(sub[x], sub[f"delta_{s}_lo"], sub[f"delta_{s}_hi"], color=c, alpha=0.15, lw=0)
        ax.plot(sub[x], sub[f"delta_{s}"], color=c, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=s)
    for _, r in sub.iterrows():
        ax.annotate(f"{r.lesion_fully_in:.0%}", (r[x], 0.045), ha="center",
                    fontsize=8, color=MUTED)
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_ylim(-0.16, 0.06)


def main():
    d = pd.read_csv(f"{D}/dose_response_msd.csv")
    summ = summarise(d)
    summ.to_csv(f"{D}/dose_response_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    shift = summ[summ.margin_scale == 1.0].sort_values("shift_mm")
    scale = summ[summ.shift_mm == 0].sort_values("margin_scale")
    off = {a: pd.read_csv(f"{D}/stage1_quality_{a}_external.csv").centroid_offset_mm
           for a in ("totalseg", "baseline_oof")}

    fig = plt.figure(figsize=(11, 4.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.1], hspace=0.08, wspace=0.22)
    a = fig.add_subplot(gs[0, 0])
    panel(a, shift, "shift_mm", "", 0)
    a.set_ylabel("Δ MSD AUROC vs unperturbed crop", color=INK)
    a.set_title("A  Crop moved off the pancreas", loc="left", color=INK, fontsize=11)
    a.legend(frameon=False, loc="lower left")
    a.text(0.99, 0.03, "% above points = tumours entirely inside the crop", transform=a.transAxes, ha="right",
           va="bottom", fontsize=8, color=MUTED)
    a.tick_params(labelbottom=False)
    s = fig.add_subplot(gs[1, 0], sharex=a)
    names = {"totalseg": "TotalSegmentator", "baseline_oof": "PANORAMA baseline"}
    bp = s.boxplot([off[k].values for k in names], orientation="horizontal", widths=0.55, showfliers=True,
                   patch_artist=True, medianprops={"color": INK}, flierprops={"ms": 3, "mfc": MUTED, "mec": MUTED})
    for p in bp["boxes"]:
        p.set(facecolor=GRID, edgecolor=MUTED)
    s.set_yticks([1, 2], list(names.values()), fontsize=8, color=MUTED)
    s.set_xlabel("centroid shift / segmenter centroid error (mm)", color=INK)
    for sp in ("top", "right"):
        s.spines[sp].set_visible(False)
    s.tick_params(colors=MUTED)

    b = fig.add_subplot(gs[0, 1])
    panel(b, scale, "margin_scale", "margin scale (× 100×50×15 mm per side)", 1.0)
    b.set_title("B  Crop margins shrunk / enlarged", loc="left", color=INK, fontsize=11)
    b.set_xlabel("margin scale (× 100×50×15 mm per side)", color=INK)
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
