"""Paper statistics for the two CT detectors: tight 3D nnU-Net (cc_psz, p_max) vs the 2.5D ERM classifier
(scores averaged over 3 seeds), on the same cases.

    compare.csv    DeLong AUROC difference 2.5D - 3D on the Dutch out-of-fold cohort (all, per scanner, automatic-
                   mask cases only) and on MSD/NIH; Brier and ECE for the probability-valued scores (3D p_max, 2.5D)
    subgroups.csv  AUROC with a DeLong 95% CI per model and subgroup: scanner, mask provenance, sex, age band
    provenance.csv mask-provenance leakage check: AUROC of PDAC with manual masks vs all non-PDAC against PDAC with
                   automatic masks vs the same non-PDAC, difference with a patient bootstrap CI (if manual-mask
                   crops gave the label away, manual-mask PDAC would be ranked higher)

    python tools/final_stats.py [n_boot]
Writes reports/final_stats/.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_stats import delong, ece  # noqa: E402

OUT = "reports/final_stats"
MODELS = [("3d_cc_psz", "cc_psz"), ("3d_p_max", "p_max"), ("2p5d_erm", "p_2p5d")]


def single_ci(y, p):
    """AUROC with a DeLong 95% CI: paired against a constant score (AUROC 0.5, zero variance), the CI of the
    difference is the single-model CI shifted by 0.5."""
    a, _, _, lo, hi, _ = delong(y, p, np.full(len(p), 0.5))
    return a, lo + 0.5, hi + 0.5


def load_dutch():
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")
    p = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/test_cv{k}.csv") for s in range(3) for k in range(5))
    p = p.groupby("case", as_index=False).p.mean().rename(columns={"p": "p_2p5d"})
    q = pd.read_csv("data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight/roi_build_qc.csv")[["case", "provenance"]]
    c = pd.read_excel("data/raw/ct/panorama_labels/clinical_information.xlsx")
    c = c.rename(columns={"PANORAMA_study_id": "case", "patient_sex": "sex"})
    c["age"] = pd.to_numeric(c.patient_age.astype(str).str.rstrip("Y"), errors="coerce")
    d = cv.merge(p, on="case").merge(q, on="case").merge(c[["case", "sex", "age"]], on="case", how="left")
    d["age_band"] = pd.cut(d.age, [0, 60, 70, 200], labels=["<60", "60-69", ">=70"], right=False).astype(str)
    assert len(d) == len(cv) == 1964, len(d)
    return d


def load_ext():
    t = pd.read_csv("reports/nnunet_summaries/external/tight_external.csv")
    p = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/ext_cv{k}.csv") for s in range(3) for k in range(5))
    p = p.groupby("case", as_index=False).p.mean().rename(columns={"p": "p_2p5d"})
    return t.merge(p, on="case")


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    os.makedirs(OUT, exist_ok=True)
    d, e = load_dutch(), load_ext()
    rows = []
    cohorts = [("dutch_all", d), *[(f"dutch_{s}", d[d.scanner == s]) for s in ["Siemens", "Toshiba", "Philips"]],
               ("dutch_auto_masks", d[d.provenance == "auto"]), ("external_all", e), ("external_MSD", e[e.source == "MSD"])]
    for name, g in cohorts:
        y = g.y.values
        r = {"cohort": name, "n": len(g), "pos": int(y.sum())}
        for m, col in MODELS:
            r[f"auroc_{m}"] = roc_auc_score(y, g[col])
        for m, col in MODELS[:2]:
            _, _, diff, lo, hi, pv = delong(y, g.p_2p5d.values, g[col].values)
            r.update({f"diff_2p5d_vs_{m}": diff, f"diff_lo_{m}": lo, f"diff_hi_{m}": hi, f"delong_p_{m}": pv})
        for m, col in (("3d_p_max", "p_max"), ("2p5d_erm", "p_2p5d")):
            r[f"brier_{m}"], r[f"ece_{m}"] = brier_score_loss(y, g[col]), ece(y, g[col].values)
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/compare.csv", index=False)

    sub = []
    for var in ["scanner", "provenance", "sex", "age_band"]:
        for lvl, g in d.groupby(var):
            if g.y.sum() < 8 or (1 - g.y).sum() < 8:
                sub.append({"variable": var, "level": lvl, "n": len(g), "pos": int(g.y.sum()), "note": "too few of one class"})
                continue
            r = {"variable": var, "level": lvl, "n": len(g), "pos": int(g.y.sum())}
            for m, col in MODELS:
                a, lo, hi = single_ci(g.y.values, g[col].values)
                r.update({f"auroc_{m}": a, f"lo_{m}": lo, f"hi_{m}": hi})
            sub.append(r)
    pd.DataFrame(sub).to_csv(f"{OUT}/subgroups.csv", index=False)

    neg = d[d.y == 0]
    prov = []
    for m, col in MODELS:
        def gap(dd):
            nn = dd[dd.y == 0][col].values
            man, auto = dd[(dd.y == 1) & (dd.provenance == "man")][col].values, dd[(dd.y == 1) & (dd.provenance == "auto")][col].values
            a_man = roc_auc_score(np.r_[np.ones(len(man)), np.zeros(len(nn))], np.r_[man, nn])
            a_auto = roc_auc_score(np.r_[np.ones(len(auto)), np.zeros(len(nn))], np.r_[auto, nn])
            return a_man, a_auto
        a_man, a_auto = gap(d)
        dd = d.assign(pat=d.case.str.split("_").str[0])
        idx, pats, rng = dd.groupby("pat").indices, dd.pat.unique(), np.random.default_rng(0)
        bs = []
        for _ in range(n_boot):
            b = dd.iloc[np.concatenate([idx[q] for q in rng.choice(pats, len(pats))])]
            am, aa = gap(b)
            bs.append(am - aa)
        prov.append({"model": m, "n_neg": len(neg), "n_pdac_manual": int(((d.y == 1) & (d.provenance == "man")).sum()),
                     "n_pdac_auto": int(((d.y == 1) & (d.provenance == "auto")).sum()), "auroc_manual_pdac": a_man,
                     "auroc_auto_pdac": a_auto, "diff": a_man - a_auto, "diff_lo": np.percentile(bs, 2.5),
                     "diff_hi": np.percentile(bs, 97.5)})
    pd.DataFrame(prov).to_csv(f"{OUT}/provenance.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for f in ("compare", "subgroups", "provenance"):
            print(f"--- {f}")
            print(pd.read_csv(f"{OUT}/{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
