"""Mitigation panel figure: each debiasing arm vs tuned ERM, plus the planted-shortcut control.

    python tools/plot_panel.py
Reads reports/panel/panel_stats.csv (tools/panel_stats.py) and planted_results.csv (tools/panel_planted_eval.py);
writes figures/panorama/mitigation_panel.png.
"""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STATS = "reports/panel/panel_stats.csv"
PLANT = "reports/panel/planted_results.csv"
OUT = "figures/panorama/mitigation_panel.png"
ARMS = [("dfr", "DFR"), ("gbs", "Group-balanced"), ("grl", "GRL")]
COLOR = {"erm": "#7a7974", "dfr": "#2a78d6", "gbs": "#eb6834", "grl": "#1baf7a"}
INK, MUTED = "#2b2b29", "#8a8984"


def forest(ax, rows, title, xlabel):
    """rows: list of (label, arm, value, lo, hi), drawn top to bottom."""
    for i, (label, arm, v, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=COLOR[arm], lw=2, solid_capstyle="round")
        ax.plot(v, y, "o", ms=8, color=COLOR[arm], mec="white", mew=2)
        ax.text(hi, y + 0.22, f"{v:+.3f}", fontsize=8, color=INK, ha="right")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    style(ax)


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


def main():
    s = pd.read_csv(STATS)
    p = pd.read_csv(PLANT)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.25, 1]})
    cv = s[s.setting == "cv"].set_index("arm")
    forest(ax[0], [(n, a, cv.auroc_diff_vs_erm[a], cv.diff_lo[a], cv.diff_hi[a]) for a, n in ARMS],
           "a  Detection, Dutch CV (DeLong)", "AUROC difference vs ERM")
    forest(ax[1], [(n, a, cv.tax_diff_vs_erm[a], cv.tax_diff_lo[a], cv.tax_diff_hi[a]) for a, n in ARMS],
           "b  Confound tax, Dutch CV", "tax difference vs ERM (< 0 = less confound use)")
    rows = []
    for sc in ["Siemens", "Toshiba", "Philips"]:
        d = s[s.setting == f"lomo_{sc}"].set_index("arm")
        rows += [(f"{n}, {sc} unseen", a, d.auroc_diff_vs_erm[a], d.diff_lo[a], d.diff_hi[a]) for a, n in ARMS]
    forest(ax[2], rows, "c  Unseen manufacturer (LOMO, DeLong)", "AUROC difference vs ERM")
    pr = p.set_index("arm")
    order = [("erm", "ERM")] + ARMS
    forest(ax[3], [(n, a, pr["drop"][a], pr.drop_lo[a], pr.drop_hi[a]) for a, n in order],
           "d  Planted shortcut, marker flipped", "AUROC lost, clean − flipped (0 = ignored)")
    fig.suptitle("Mitigation panel: 2.5D classifier, scores averaged over 3 seeds; 95% CIs (DeLong for AUROC, "
                 "patient bootstrap for tax and planted drop)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
