"""Do Dutch-CV thresholds transfer to the external cohorts?

For each tight CV fold f, pick operating points on that fold's own Dutch out-of-fold scores
(Youden, and the threshold reaching 90% sensitivity), then apply them to fold f's single-model
predictions on MSD and NIH. Same thresholds pooled over all folds are applied to the 5-fold
ensemble for comparison. MSD and NIH are reported separately (never pooled).

    python tools/external_threshold_check.py
Writes reports/nnunet_summaries/external/threshold_check.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = "reports/nnunet_summaries"
SCORES = ["p_max", "cc_psz"]


def operating_points(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    youden = thr[np.argmax(tpr - fpr)]
    sens90 = thr[np.argmax(tpr >= 0.90)]
    return {"youden": youden, "sens90": sens90}


def rates(ext, score, t):
    msd, nih = ext[ext.source == "MSD"], ext[ext.source == "NIH"]
    pos, neg = msd[msd.y == 1], msd[msd.y == 0]
    return {
        "msd_sens": (pos[score] >= t).mean(),
        "msd_spec": (neg[score] < t).mean(),
        "nih_spec": (nih[score] < t).mean(),
    }


def main():
    cv = pd.read_csv(f"{ROOT}/tight_battery/cv_scores.csv")
    rows = []
    for score in SCORES:
        for f in sorted(cv.fold.unique()):
            dutch = cv[cv.fold == f]
            ext = pd.read_csv(f"{ROOT}/external/tight_external_fold{f}.csv")
            for op, t in operating_points(dutch.y, dutch[score]).items():
                d = rates(dutch.assign(source="MSD"), score, t)  # Dutch own-fold rates
                rows.append({"model": f"fold{f}", "score": score, "op": op, "thr": t,
                             "dutch_sens": d["msd_sens"], "dutch_spec": d["msd_spec"],
                             **rates(ext, score, t)})
        ens = pd.read_csv(f"{ROOT}/external/tight_external.csv")
        for op, t in operating_points(cv.y, cv[score]).items():
            rows.append({"model": "ensemble (pooled CV thr)", "score": score, "op": op, "thr": t,
                         **rates(ens, score, t)})
    out = pd.DataFrame(rows)
    out.to_csv(f"{ROOT}/external/threshold_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
