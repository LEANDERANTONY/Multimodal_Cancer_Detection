"""Patient-disjoint evaluation set for the Dutch PANORAMA cohort.

The CV folds (nnU-Net's default split, shared by the 3D and 2.5D models) were drawn by scan. 11 patients have more
than one scan (25 scans); for 9 of them the scans fall in different folds, and for 4 the manufacturer differs between
scans, so LOMO is affected too. Every detection metric on the Dutch cohort is therefore computed on the 1,939 scans
of patients with a single examination: no test patient then contributed any scan to training. The repeat scans stay
in the training data. The external MSD / NIH scans share no patient with the Dutch cohort and have one scan per
patient, so they are untouched.

    from eval_cohort import eval_set
    d = eval_set(d)          # drops the repeat-examination patients' scans (case ids "<patient>_<study>")

Set ALL_SCANS=1 in the environment to reproduce the earlier all-scan numbers.
"""
import os

import pandas as pd

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "reports/nnunet_summaries/tight_battery/cv_scores.csv")  # the 1,964 Dutch development scans


def _patient(case):
    return case.astype(str).str.split("_").str[0]


_pid = _patient(pd.read_csv(CASES, usecols=["case"]).case)
MULTI_EXAM = frozenset(_pid[_pid.duplicated(keep=False)])  # 11 patients
assert len(MULTI_EXAM) == 11, len(MULTI_EXAM)


def eval_set(d, col="case"):
    if os.environ.get("ALL_SCANS") == "1":
        return d
    return d[~_patient(d[col]).isin(MULTI_EXAM)].reset_index(drop=True)
