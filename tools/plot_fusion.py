"""Fusion figure: what combining urine biomarkers with CT could add, as a function of the unknown dependence
between the two tests, next to the pseudo-paired fusion number the thesis would have reported.

    python tools/plot_fusion.py
Reads reports/fusion/fusion_triage.csv and fusion_pseudo_pairs.csv (tools/fusion_sim.py);
writes figures/fusion_triage.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TRIAGE = "reports/fusion/fusion_triage.csv"
PAIRS = "reports/fusion/fusion_pseudo_pairs.csv"
OUT = "figures/fusion_triage.png"
INK, MUTED, GRID = "#2b2b29", "#8a8984", "#e6e5e0"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#7a7974"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    t = pd.read_csv(TRIAGE)
    t = t[t.ct_model == "ct_3d"]
    pp = pd.read_csv(PAIRS)
    pp = pp[pp.ct_model == "ct_3d"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    a = ax[0]
    a.fill_between(t.rho, t.combined_auroc_lo, t.combined_auroc_hi, color=BLUE, alpha=0.15, lw=0)
    a.plot(t.rho, t.combined_auroc, "-o", color=BLUE, lw=2, ms=8, mec="white", mew=2)
    a.text(t.rho.iloc[1] + 0.02, t.combined_auroc.iloc[1] + 0.004, "best combined score", color=INK, fontsize=8)
    for val, lab in ((t.biomarker_auroc.iloc[0], "urine biomarkers alone"), (t.ct_auroc.iloc[0], "CT (3D) alone")):
        a.axhline(val, color=GRAY, lw=1, ls="--")
        a.text(0.76, val + (0.004 if lab.startswith("CT") else -0.008), f"{lab} {val:.3f}", color=MUTED, fontsize=8, ha="right")
    m = pp[pp.pairing == "label_matched"].fusion_auroc
    a.plot(0, m.mean(), "D", color=ORANGE, ms=9, mec="white", mew=2)
    a.annotate(f"pseudo-paired fusion {m.mean():.3f}\n(what random label-matched pairing reports)",
               (0, m.mean()), (0.08, m.mean() + 0.012), fontsize=8, color=INK,
               arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    a.set_xlabel("within-class correlation of the two tests' scores (rho)", fontsize=9, color=INK)
    a.set_ylabel("AUROC, PDAC vs non-PDAC", fontsize=9, color=INK)
    a.set_title("a  Fusion AUROC depends on an untestable assumption", fontsize=10, color=INK, loc="left")
    a.set_xticks(t.rho)
    style(a)

    b = ax[1]
    for col, ref, lab, colr in (("triage_sens", "ct_alone_sens", "sensitivity", BLUE),
                                ("triage_spec", "ct_alone_spec", "specificity", ORANGE)):
        b.fill_between(t.rho, t[f"{col}_lo"], t[f"{col}_hi"], color=colr, alpha=0.15, lw=0)
        b.plot(t.rho, t[col], "-o", color=colr, lw=2, ms=8, mec="white", mew=2)
        b.axhline(t[ref].iloc[0], color=colr, lw=1, ls="--")
        dy = -0.035 if col == "triage_sens" else -0.05
        b.text(t.rho.iloc[-1], t[col].iloc[-1] + dy, f"urine -> CT {lab}", color=INK, fontsize=8, ha="right")
        b.text(0, t[ref].iloc[0] + 0.012, f"CT for everyone, {lab} {t[ref].iloc[0]:.2f}", color=MUTED, fontsize=8)
    av = t.ct_avoided_screening_1pct.iloc[0]
    b.set_title(f"b  Urine triage (95 % sens.) then CT: {av:.0%} of CTs avoided at 1 % prevalence",
                fontsize=10, color=INK, loc="left")
    b.set_xlabel("within-class correlation of the two tests' scores (rho)", fontsize=9, color=INK)
    b.set_ylabel("pathway sensitivity / specificity", fontsize=9, color=INK)
    b.set_xticks(t.rho)
    style(b)

    fig.suptitle("CT (PANORAMA) and urine biomarkers come from different patients: fusion simulated from each test's "
                 "real scores; 95 % bootstrap bands", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
