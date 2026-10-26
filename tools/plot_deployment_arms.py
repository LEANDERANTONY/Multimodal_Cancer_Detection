"""Deployment-ROI on the Dutch cohort, all stage-1 arms side by side: deployed detection AUROC (intention-to-treat,
a segmenter failure = called negative), confound tax, and segmenter failures per scanner.

    python tools/plot_deployment_arms.py
Reads reports/deployment_roi/deploy_gap_dutch.csv and deploy_fail_dutch.csv (tools/deploy_gap_dutch.py);
writes figures/panorama/deployment_arms.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

GAP = "reports/deployment_roi/deploy_gap_dutch.csv"
FAIL = "reports/deployment_roi/deploy_fail_dutch.csv"
OUT = "figures/panorama/deployment_arms.png"
ARMS = [("baseline_oof", "PANORAMA baseline\nsegmenter"), ("totalseg", "TotalSegmentator\npancreas only"),
        ("totalseg_rc", "TotalSegmentator\n+ robust_crop"), ("totalseg_nb", "TotalSegmentator\n+ neighbour organs")]
SCANNERS = ["Siemens", "Toshiba", "Philips"]


def main():
    g = pd.read_csv(GAP)
    g = g[(g.population == "itt") & (g.score == "cc_psz")]
    f = pd.read_csv(FAIL)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = range(len(ARMS))
    for k, (metric, title) in enumerate([("mixed", "Detection AUROC (cc_psz), deployed"),
                                         ("tax", "Confound tax (mixed − within-scanner AUROC)")]):
        rows = [g[(g.arm == a) & (g.metric == metric)].iloc[0] for a, _ in ARMS]
        pred = [r.predicted for r in rows]
        lo = [r.predicted - (r["diff"] - r.diff_lo) for r in rows]
        hi = [r.predicted + (r.diff_hi - r["diff"]) for r in rows]
        ax[k].bar(x, pred, color="#4C72B0")
        ax[k].errorbar(x, pred, yerr=[[p - l for p, l in zip(pred, lo)], [h - p for p, h in zip(pred, hi)]],
                       fmt="none", ecolor="black", capsize=4)
        ax[k].axhline(rows[0].oracle, color="#C44E52", ls="--", label=f"oracle crop ({rows[0].oracle:.3f})")
        ax[k].set_xticks(list(x), [n for _, n in ARMS], fontsize=8)
        ax[k].set_title(title, fontsize=10)
        ax[k].legend(fontsize=8)
        for i, p in enumerate(pred):
            ax[k].text(i, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_ylim(0.6, 0.85)
    ax[1].set_ylim(0, 0.07)
    w = 0.25
    for j, s in enumerate(SCANNERS):
        vals = [100 * f[(f.arm == a) & (f.scanner == s)].eval("failed / n").iloc[0] for a, _ in ARMS]
        ax[2].bar([i + (j - 1) * w for i in x], vals, w, label=s)
    ax[2].set_xticks(list(x), [n for _, n in ARMS], fontsize=8)
    ax[2].set_ylabel("% scans with empty pancreas mask")
    ax[2].set_title("Segmenter failures per scanner", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.suptitle("Deployment-ROI, Dutch cohort (AUROC on 1,939 single-exam patients, failures on all 1,964 scans; "
                 "out-of-fold, failures called negative); "
                 "error bars = 95% patient bootstrap CI of the paired difference", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
