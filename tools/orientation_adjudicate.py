"""Reviewed cranio-caudal orientation of each Dutch scan, and what it does and does not explain.

The lung rule of tools/orientation_check.py made errors (bowel gas read as lung; no lung in the end slices), so its
calls were reviewed on mid-sagittal contact sheets (coronal too where unclear). Reviewed: all 245 files of the batch
gzipped on Windows, all Unix-written scans the rule called inverted or unclear, and a random 80 of the 1,590 Unix-written
scans it called correct. Other Unix-written scans keep the rule's call. Review by the analysing AI assistant, not a
radiologist; the overrides are listed below so they can be checked.

Final classes: correct, inverted (voxels stored head-first while the header declares foot-first), rotated, uncertain.

Reports, against the file batch ("writer": gzip OS byte + sform_code), the `level` column and scanner:
  - inversion vs batch; inversion vs PDAC crude, within strata (Mantel-Haenszel) and inside the Windows batch
  - TotalSegmentator failures per setting by final class
  - the detection scores: inverted vs correct non-PDAC scans within scanner x level, AUROC inside the same streams

    python tools/orientation_adjudicate.py
Writes reports/deployment_roi/orientation_adjudicated.csv and orientation_adjudicated_summary.csv.
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cohort import eval_set  # noqa: E402

IMG = "data/raw/ct/panorama/images/{}_0000.nii.gz"
OVERRIDES = {  # rule call -> reviewed call
    "100312": "inverted", "100411": "inverted", "102208": "inverted", "101109": "inverted", "100138": "inverted",
    "100841": "inverted", "100043": "inverted", "100060": "inverted",  # Windows batch, rule "unclear"
    "101913": "correct", "100054": "correct", "100388": "correct", "101352": "correct", "101355": "correct",
    "100199": "uncertain",  # Windows batch, rule "unclear"
    "100350": "correct", "101710": "correct",  # Unix, rule "inverted": bowel gas read as lung
    "101104": "rotated",  # Unix, rule "inverted": stored rotated 90 degrees, a different problem
}
ARMS = {"totalseg": "roi_subset pancreas", "totalseg_rc": "+ robust_crop", "totalseg_nb": "+ neighbouring organs"}


def load():
    o = pd.read_csv("reports/deployment_roi/orientation_check.csv")[["case", "orientation"]].rename(columns={"orientation": "rule"})
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")[["case", "fold", "scanner", "y", "cc_psz"]]
    c = pd.read_excel("data/raw/ct/panorama_labels/clinical_information.xlsx").rename(columns={"PANORAMA_study_id": "case"})
    d = cv.merge(o, on="case").merge(c[["case", "level"]], on="case")
    d["final"] = d.rule.map({"header_ok": "correct", "inverted": "inverted", "unclear": "correct"})
    for k, v in OVERRIDES.items():
        d.loc[d.case.str.startswith(k + "_"), "final"] = v
    gz, sf = [], []
    for case in d.case:
        with open(IMG.format(case), "rb") as f:
            gz.append(f.read(10)[9])
        sf.append(int(nib.load(IMG.format(case)).header["sform_code"]))
    d["gz_os"], d["sform_code"] = gz, sf
    d["batch"] = np.where(d.gz_os == 11, "windows", "unix")
    d["inv"] = (d.final == "inverted").astype(int)
    for arm in ARMS:
        sk = f"data/processed/ct/deploy_crops/{arm}_dutch/skipped.txt"  # absent when an arm had no failures
        d[f"fail_{arm}"] = d.case.isin(open(sk).read().split() if os.path.exists(sk) else [])
    return d


def mh(g, keys):
    num = den = 0.0
    for _, x in g.groupby(keys):
        i, y = x.inv.values, x.y.values
        a, b, c, e = ((i == 1) & (y == 1)).sum(), ((i == 1) & (y == 0)).sum(), ((i == 0) & (y == 1)).sum(), ((i == 0) & (y == 0)).sum()
        num += a * e / len(x)
        den += b * c / len(x)
    return num / den if den else float("nan")


def mh_ci(g, keys, n_boot=1000):
    g = g.assign(pat=g.case.str.split("_").str[0])
    idx, pats, rng, bs = g.groupby("pat").indices, g.pat.unique(), np.random.default_rng(0), []
    for _ in range(n_boot):
        bs.append(mh(g.iloc[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])], keys))
    return np.nanpercentile(bs, [2.5, 97.5])


def strat_auc(g, col):
    num = den = 0.0
    for _, x in g.groupby(["scanner", "level"]):
        n1, n0 = x.inv.sum(), (1 - x.inv).sum()
        if n1 >= 3 and n0 >= 3:
            w = n1 * n0 / (n1 + n0)
            num += w * roc_auc_score(x.inv, x[col])
            den += w
    return num / den if den else float("nan")


def main():
    d = load()
    d[["case", "scanner", "y", "level", "batch", "sform_code", "gz_os", "rule", "final"]].to_csv(
        "reports/deployment_roi/orientation_adjudicated.csv", index=False)
    rows = []
    print(pd.crosstab([d.batch, d.sform_code], d.final, margins=True))
    print(pd.crosstab([d.batch, d.scanner], d.final))
    print(pd.crosstab(d.level, [d.batch, d.final]))
    inv, win = d.inv == 1, d.batch == "windows"
    rows.append({"block": "counts", "n": len(d), "inverted": int(inv.sum()), "inverted_pct": 100 * inv.mean(),
                 "inverted_in_windows": int((inv & win).sum()), "windows": int(win.sum())})

    tab = pd.crosstab(d.inv, d.y)
    orr, p = fisher_exact(tab.values)
    print(f"\ninverted vs PDAC: {100 * d.y[inv].mean():.1f} % vs {100 * d.y[~inv].mean():.1f} %; crude OR {orr:.2f}, p {p:.2g}")
    rows.append({"block": "pdac_crude", "pdac_inverted": 100 * d.y[inv].mean(), "pdac_other": 100 * d.y[~inv].mean(), "or": orr, "p": p})
    for keys in (["level"], ["scanner"], ["scanner", "level"], ["batch"]):
        lo, hi = mh_ci(d, keys)
        val = mh(d, keys)
        print(f"Mantel-Haenszel OR by {keys}: {val:.2f} [{lo:.2f}, {hi:.2f}]")
        rows.append({"block": "pdac_mh", "strata": "+".join(keys), "or": val, "lo": lo, "hi": hi})
    w = d[win & d.final.isin(["inverted", "correct"])]
    tab = pd.crosstab(w.inv, w.y)
    orr, p = fisher_exact(tab.values)
    print(f"inside the Windows batch: PDAC {100 * w.y[w.inv == 1].mean():.1f} % inverted vs {100 * w.y[w.inv == 0].mean():.1f} % correct; OR {orr:.2f}, p {p:.2f}")
    rows.append({"block": "pdac_in_windows", "pdac_inverted": 100 * w.y[w.inv == 1].mean(), "pdac_other": 100 * w.y[w.inv == 0].mean(), "or": orr, "p": p})
    print(f"PDAC share by batch: windows {100 * d.y[win].mean():.1f} % vs unix {100 * d.y[~win].mean():.1f} %")

    for arm, name in ARMS.items():
        print(f"\n{name}\n", pd.crosstab(d.final, d[f"fail_{arm}"], margins=True))
        for st, g in d.groupby("final"):
            rows.append({"block": "failures", "arm": name, "final": st, "n": len(g), "failed": int(g[f"fail_{arm}"].sum())})

    p25 = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/test_cv{k}.csv") for s in range(3) for k in range(5))
    e = eval_set(d.merge(p25.groupby("case", as_index=False).p.mean(), on="case"))
    neg = e[(e.y == 0) & e.final.isin(["inverted", "correct"])]
    rng = np.random.default_rng(0)
    print()
    for col, nm in (("p", "2.5D"), ("cc_psz", "3D")):
        bs = []
        for _ in range(1000):
            b = pd.concat(x.sample(len(x), replace=True, random_state=int(rng.integers(10**9))) for _, x in neg.groupby(["scanner", "level"]))
            bs.append(strat_auc(b, col))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        crude, st = roc_auc_score(neg.inv, neg[col]), strat_auc(neg, col)
        print(f"{nm}: score -> inverted among non-PDAC: crude {crude:.3f}; within scanner x level {st:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append({"block": "score_vs_inverted", "model": nm, "crude": crude, "stratified": st, "lo": lo, "hi": hi})
    streams = e[e.level.isin(["pathology", "cytology", "radiology / 3yFU"])]
    for st in ("correct", "inverted"):
        g = streams[streams.final == st]
        a2, a3 = roc_auc_score(g.y, g.p), roc_auc_score(g.y, g.cc_psz)
        print(f"same streams, {st}: n {len(g)} PDAC {int(g.y.sum())}: 2.5D {a2:.3f}, 3D {a3:.3f}")
        rows.append({"block": "auroc_same_streams", "final": st, "n": len(g), "pos": int(g.y.sum()), "auroc_2p5d": a2, "auroc_3d": a3})
    pd.DataFrame(rows).to_csv("reports/deployment_roi/orientation_adjudicated_summary.csv", index=False)


if __name__ == "__main__":
    main()
