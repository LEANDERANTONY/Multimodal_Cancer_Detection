"""Graphical abstract (MedIA: landscape, at least 1328 px wide, readable at 13 cm).

Three panels, numbers read from the committed result tables:
  1. the confound in the data: PDAC share by scanner manufacturer; scanner-only AUROC and the two models' confound tax
  2. the header mismatch: TotalSegmentator failure rate as stored and with only the header corrected or inverted
  3. debiasing: AUROC lost to a planted shortcut (ERM vs debiasers) next to the change in the real scanner tax

    python tools/plot_graphical_abstract.py
Writes figures/panorama/graphical_abstract.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK, MUTED = "#2b2b29", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a7974"
OUT = "figures/panorama/graphical_abstract.png"


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def main():
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")
    share = cv[cv.scanner.isin(["Siemens", "Toshiba", "Philips"])].groupby("scanner").y.mean().mul(100)
    o = pd.read_csv("reports/deployment_roi/orientation_tests_summary.csv")
    o = o[o.block == "ts_failure"]
    fail = o.set_index("group").pct
    broken = 100 * o[o.group.str.contains("header inverted")].failed.sum() / o[o.group.str.contains("header inverted")].n.sum()
    adj = pd.read_csv("reports/deployment_roi/orientation_adjudicated_summary.csv")
    c = adj[(adj.block == "failures") & (adj.arm == "roi_subset pancreas") & (adj.final == "correct")].iloc[0]
    correct_failed = 100 * c.failed / c.n
    pl =pd.read_csv("reports/panel/planted_results.csv").set_index("arm")["drop"]
    st = pd.read_csv("reports/panel/panel_stats.csv")
    tax = st[st.setting == "cv"].set_index("arm").tax_diff_vs_erm

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    a = ax[0]
    order = ["Siemens", "Toshiba", "Philips"]
    a.bar(order, [share[s] for s in order], color=[BLUE, BLUE, ORANGE], width=0.6)
    for i, s in enumerate(order):
        a.text(i, share[s] + 2, f"{share[s]:.0f} %", ha="center", fontsize=10, color=INK)
    a.set_ylabel("PDAC share of scans (%)", fontsize=10, color=INK)
    a.set_ylim(0, 95)
    a.set_title("1  Confound present in the data", fontsize=12, loc="left", color=INK, weight="bold")
    a.text(0.02, 0.97, "Scanner alone: AUROC 0.70\nConfound tax: 3D +0.028, 2.5D +0.056\n-> weakly exploited",
           transform=a.transAxes, va="top", fontsize=9.5, color=INK)
    style(a)

    a = ax[1]
    x = [0, 1, 3, 4]
    vals = [correct_failed, fail["inverted, as stored"], fail["inverted, header corrected"], broken]
    a.bar(x, vals, color=[BLUE, ORANGE, BLUE, ORANGE], width=0.8)
    for xi, v in zip(x, vals):
        a.text(xi, v + 2, f"{v:.1f} %" if v < 1 else f"{v:.0f} %", ha="center", fontsize=10, color=INK)
    a.set_xticks(x, ["correct\nscans", "inverted\nscans", "inverted,\nheader\ncorrected", "correct,\nheader\ninverted"], fontsize=9)
    a.set_ylabel("TotalSegmentator: empty pancreas mask (%)", fontsize=10, color=INK)
    a.set_ylim(0, 115)
    a.legend(handles=[Patch(color=BLUE, label="header matches anatomy"),
                      Patch(color=ORANGE, label="header contradicts slice order")],
             fontsize=8.5, frameon=False, loc="upper left")
    a.set_title("2  A hidden header mismatch", fontsize=12, loc="left", color=INK, weight="bold")
    a.text(0.02, 0.84, "8.1 % of scans, one release batch\nVoxels never changed: the header alone\ncures and creates the failures",
           transform=a.transAxes, va="top", fontsize=9.5, color=INK)
    style(a)

    a = ax[2]
    arms = [("erm", "ERM"), ("gbs", "group-\nbalanced"), ("dfr", "DFR"), ("grl", "GRL")]
    a.bar(range(4), [pl[k] for k, _ in arms], color=[GRAY, AQUA, AQUA, AQUA], width=0.6)
    for i, (k, _) in enumerate(arms):
        a.text(i, pl[k] + 0.004, f"{pl[k]:.3f}", ha="center", fontsize=10, color=INK)
    a.set_xticks(range(4), [n for _, n in arms], fontsize=9.5)
    a.set_ylabel("AUROC lost to a planted shortcut", fontsize=10, color=INK)
    a.set_ylim(0, 0.2)
    a.set_title("3  Debiasing: works when planted, not here", fontsize=12, loc="left", color=INK, weight="bold")
    a.text(0.98, 0.97, "Real scanner tax vs ERM:\n" + "\n".join(f"{n.replace(chr(10), '').replace('-', '-', 1)} {tax[k]:+.3f} (n.s.)" for k, n in arms[1:])
           + f"\nMAE pretraining {tax['mae']:+.3f} (raised)", transform=a.transAxes, va="top", ha="right", fontsize=9, color=INK)
    style(a)
    fig.suptitle("Confounding in AI detection of pancreatic cancer on CT (PANORAMA, 1,964 development scans)",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
