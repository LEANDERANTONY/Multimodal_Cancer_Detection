"""Results of the two orientation experiments, on the reviewed labels (tools/orientation_adjudicate.py).

TotalSegmentator (tools/orientation_totalseg_test.py): is the header-anatomy mismatch the cause of the empty masks?
  failure rate as stored vs header corrected vs header deliberately inverted on correct scans, per batch; paired Dice.
Classifier (tools/orientation_flip_all.py; skipped until its CSV exists): how much of the 2.5D score follows orientation?
  paired within-scan change; inverted-upright vs correct non-PDAC within scanner x level (the residual); AUROC and
  confound tax with the inverted scans upright, and with every scan upside down.

    python tools/orientation_tests_summary.py
Writes reports/deployment_roi/orientation_tests_summary.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confound_tax_ci import tax  # noqa: E402

TS = "reports/deployment_roi/orientation_totalseg_test.csv"
FLIP = "reports/deployment_roi/orientation_flip_all.csv"
ADJ = "reports/deployment_roi/orientation_adjudicated.csv"
OUT = "reports/deployment_roi/orientation_tests_summary.csv"
NATURAL = (112, 160)  # roi_subset pancreas, inverted as stored; orientation_adjudicated_summary.csv


def rate(k, n):
    lo, hi = proportion_confint(k, n, method="beta")
    return {"failed": int(k), "n": int(n), "pct": 100 * k / n, "lo": 100 * lo, "hi": 100 * hi}


def totalseg(rows):
    r = pd.read_csv(TS).merge(pd.read_csv(ADJ)[["case", "final", "batch"]], on="case")
    print("errors:", int((r.status == "error").sum()))
    print(pd.crosstab([r.arm, r.final, r.failed_stage1], r.status))

    fix = r[r.final.eq("inverted") & r.arm.isin(["fix", "other"])]
    groups = {"inverted, as stored": NATURAL,
              "inverted, header corrected": ((fix.status == "empty").sum(), len(fix))}
    for arm, nm in (("break_win", "correct Windows batch, header inverted"), ("break_unix", "correct Unix batch, header inverted")):
        g = r[r.arm == arm]
        groups[nm] = ((g.status == "empty").sum(), len(g))
    print()
    for nm, (k, n) in groups.items():
        x = rate(k, n)
        print(f"{nm}: {x['failed']}/{x['n']} empty = {x['pct']:.1f} % [{x['lo']:.1f}, {x['hi']:.1f}]")
        rows.append({"block": "ts_failure", "group": nm, **x})
    for a, b in (("correct Windows batch, header inverted", "correct Unix batch, header inverted"),
                 ("inverted, as stored", "correct Windows batch, header inverted"),
                 ("inverted, as stored", "correct Unix batch, header inverted")):
        (k1, n1), (k2, n2) = groups[a], groups[b]
        orr, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
        print(f"{a} vs {b}: OR {orr:.2f}, p {p:.2f}")
        rows.append({"block": "ts_failure_test", "group": f"{a} vs {b}", "or": orr, "p": p})

    rec = fix[fix.failed_stage1]
    print(f"\nstage-1 failures among inverted, header corrected: {(rec.status == 'ok').sum()}/{len(rec)} recovered; "
          f"Dice median {rec.dice.median():.3f}, min {rec.dice.min():.3f}")
    rows.append({"block": "ts_recovered", "failed": int((rec.status == "ok").sum()), "n": len(rec), "dice_median": rec.dice.median(), "dice_min": rec.dice.min()})
    print(f"all inverted, header corrected: Dice median {fix.dice.median():.3f}, IQR {fix.dice.quantile(.25):.3f}-{fix.dice.quantile(.75):.3f}")
    rows.append({"block": "ts_fixed_dice", "n": len(fix), "dice_median": fix.dice.median(), "lo": fix.dice.quantile(.25), "hi": fix.dice.quantile(.75)})

    s = r[(r.arm == "sham") & r.final.eq("inverted") & ~r.failed_stage1].merge(r[r.arm == "fix"][["case", "dice", "voxels"]], on="case", suffixes=("", "_fix"))
    print(f"inverted non-failures, same scans: Dice {s.dice.median():.3f} as stored -> {s.dice_fix.median():.3f} corrected; "
          f"{(s.dice_fix > s.dice).sum()}/{len(s)} improve, Wilcoxon p {wilcoxon(s.dice, s.dice_fix).pvalue:.3g}; "
          f"mask voxels {s.voxels.median():.0f} -> {s.voxels_fix.median():.0f}")
    rows.append({"block": "ts_nonfailures_paired", "n": len(s), "dice_stored": s.dice.median(), "dice_median": s.dice_fix.median(), "p": wilcoxon(s.dice, s.dice_fix).pvalue})

    for arm in ("break_win", "break_unix"):
        g = r[(r.arm == arm) & (r.status == "ok")]
        print(f"{arm}: the masks not empty: Dice median {g.dice.median():.3f}, n {len(g)}")
        rows.append({"block": "ts_broken_nonempty_dice", "group": arm, "n": len(g), "dice_median": g.dice.median()})
    for tag, nm in (("totalseg_nb", "neighbouring organs"), ("baseline_oof", "PANORAMA baseline")):
        q = pd.read_csv(f"reports/deployment_roi/stage1_quality_{tag}_dutch.csv").merge(r[["case", "final"]].drop_duplicates(), how="left", on="case")
        q = q.drop(columns="final").merge(pd.read_csv(ADJ)[["case", "final"]], on="case")
        for st in ("inverted", "correct"):
            v = q[q.final == st].dice_panc.median()
            print(f"{nm}, {st} as stored: Dice median {v:.3f}")
            rows.append({"block": "ts_other_segmenters_dice", "group": f"{nm}, {st}", "n": int((q.final == st).sum()), "dice_median": v})
    o = r[(r.final != "inverted") & r.failed_stage1 & r.arm.isin(["other", "fix"])]
    print("failures not inverted, header flipped:\n", o[["case", "final", "batch", "arm", "status"]].to_string(index=False))


def strat_auc(g, col, group="upright_inv"):
    num = den = 0.0
    for _, x in g.groupby(["scanner", "level"]):
        n1, n0 = x[group].sum(), (1 - x[group]).sum()
        if n1 >= 3 and n0 >= 3:
            w = n1 * n0 / (n1 + n0)
            num += w * roc_auc_score(x[group], x[col])
            den += w
    return num / den if den else float("nan")


def flip(rows):
    d = pd.read_csv(FLIP)
    d = d.merge(pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["case", "fold"]], on="case")
    d = d[d.final.isin(["inverted", "correct"])].copy()
    d["inv"] = (d.final == "inverted").astype(int)
    d["delta"] = d.p_flipped - d.p_original
    d["p_upright"] = np.where(d.inv == 1, d.p_flipped, d.p_original)  # every scan head-up, as the anatomy is
    d["p_inverted"] = np.where(d.inv == 1, d.p_original, d.p_flipped)  # every scan upside down
    d["upright_inv"] = d.inv
    print(f"\nflip test: {len(d)} scans, {d.inv.sum()} inverted")
    for (st, y), g in d.groupby(["final", "y"]):
        p = wilcoxon(g.delta).pvalue
        print(f"{st}, y={y}: n {len(g)}; {g.p_original.median():.3f} -> {g.p_flipped.median():.3f}; median change {g.delta.median():+.3f}, "
              f"{100 * (g.delta > 0).mean():.1f} % up, p {p:.2g}")
        rows.append({"block": "flip_paired", "group": f"{st}, y={y}", "n": len(g), "stored": g.p_original.median(), "flipped": g.p_flipped.median(),
                     "change": g.delta.median(), "pct_up": 100 * (g.delta > 0).mean(), "p": p})

    neg, rng, pre, post = d[d.y == 0], np.random.default_rng(0), [], []
    for _ in range(1000):
        b = pd.concat(x.sample(len(x), replace=True, random_state=int(rng.integers(10**9))) for _, x in neg.groupby(["scanner", "level"]))
        pre.append(strat_auc(b, "p_original"))
        post.append(strat_auc(b, "p_upright"))
    a0, a1 = strat_auc(neg, "p_original"), strat_auc(neg, "p_upright")
    print(f"score -> inverted, non-PDAC within scanner x level: as stored {a0:.3f} {np.nanpercentile(pre, [2.5, 97.5]).round(3)}; "
          f"inverted turned upright {a1:.3f} {np.nanpercentile(post, [2.5, 97.5]).round(3)}")
    rows.append({"block": "flip_score_vs_inverted", "group": "as stored", "auc": a0, "lo": np.nanpercentile(pre, 2.5), "hi": np.nanpercentile(pre, 97.5)})
    rows.append({"block": "flip_score_vs_inverted", "group": "inverted upright", "auc": a1, "lo": np.nanpercentile(post, 2.5), "hi": np.nanpercentile(post, 97.5)})

    d["pat"] = d.case.str.split("_").str[0]
    idx, ids = d.groupby("pat").indices, d.pat.unique()
    cols = {"as stored": "p_original", "inverted upright": "p_upright", "all upside down": "p_inverted"}
    res = {nm: (roc_auc_score(d.y, d[c]), tax(d, c)[0]) for nm, c in cols.items()}
    bs = {nm: [] for nm in cols}
    for _ in range(1000):
        b = d.iloc[np.concatenate([idx[p] for p in rng.choice(ids, len(ids))])]
        for nm, c in cols.items():
            bs[nm].append((roc_auc_score(b.y, b[c]), tax(b, c)[0]))
    base = np.array(bs["as stored"])
    for nm in cols:
        x = np.array(bs[nm])
        lo, hi = np.percentile(x[:, 1], [2.5, 97.5])
        dlo, dhi = np.percentile(x[:, 0] - base[:, 0], [2.5, 97.5])
        print(f"{nm}: pooled AUROC {res[nm][0]:.3f} (vs stored [{dlo:+.3f}, {dhi:+.3f}]); tax {res[nm][1]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        rows.append({"block": "flip_auroc_tax", "group": nm, "auc": res[nm][0], "dlo": dlo, "dhi": dhi, "tax": res[nm][1], "lo": lo, "hi": hi})


def main():
    rows = []
    totalseg(rows)
    if os.path.exists(FLIP):
        flip(rows)
    else:
        print(f"\n{FLIP} not there yet")
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
