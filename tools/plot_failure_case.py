"""Image figure: one TotalSegmentator failure on a scan whose header inverts the cranio-caudal axis.

Case 100113_00001 (Siemens, PDAC): the file header places the lungs at the feet (tools/orientation_check.py), and
`roi_subset pancreas` returned an empty mask, so the deployed pipeline had no crop and scored the scan as negative.
Panels: (a) coronal slice as the header lays it out (what an orientation-dependent tool sees: liver at the bottom);
(b) the same slice in anatomical orientation, with the provided pancreas and lesion outlines, the oracle crop and the crop
from TotalSegmentator + neighbouring organs; (c) axial slice through the lesion (viewed from below) with both crops.
Crops follow tools/build_roi_dataset.py: pancreas bounding box + 100 x 50 x 15 mm per side (x y z), clipped.

    python tools/plot_failure_case.py [case]
Writes figures/panorama/fig_failure_case.png. Reads the local raw scan and masks under data/ (not tracked).
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "100113_00001"
IMG = f"data/raw/ct/panorama/images/{CASE}_0000.nii.gz"
LAB = f"data/raw/ct/panorama_labels/manual_labels/{CASE}.nii.gz"
NB = f"data/processed/ct/stage1_masks/totalseg_nb/{CASE}.nii.gz"
MARGIN_MM = (100, 50, 15)  # x y z, per side
OUT = "figures/panorama/fig_failure_case.png"
WL, WW = 40, 400
PANC, LES, ORACLE, PRED = "#eda100", "#eb6834", "white", "#1baf7a"


def load(p):
    im = sitk.DICOMOrient(sitk.ReadImage(p), "LPS")
    return sitk.GetArrayFromImage(im), im.GetSpacing()  # array z, y, x; header says z index increases superiorly


def crop_box(mask, sp):
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    m = [int(np.ceil(mm / s)) for mm, s in zip(MARGIN_MM[::-1], sp[::-1])]
    return [(max(0, a - k), min(n, b + k)) for a, b, k, n in zip(lo, hi, m, mask.shape)]


def window(x):
    return np.clip((x - (WL - WW / 2)) / WW, 0, 1)


def scalebar(ax, w, h, _):
    y = h * 0.05
    ax.plot([w * 0.05, w * 0.05 + 50], [y, y], color="white", lw=2)
    ax.text(w * 0.05 + 60, y, "50 mm", color="white", fontsize=7, ha="left", va="center")


def main():
    state = pd.read_csv("reports/deployment_roi/orientation_check.csv").set_index("case").orientation.get(CASE, "?")
    ct, sp = load(IMG)
    lab, _ = load(LAB)
    nb, _ = load(NB)
    sx, sy, sz = sp
    panc, les = np.isin(lab, (4, 5)), lab == 1
    oracle, pred = crop_box(panc, sp), crop_box(nb > 0, sp)
    zc, yc, _ = np.round(np.argwhere(les).mean(0)).astype(int)
    nz, ny, nx = ct.shape
    flip = state == "inverted"  # anatomical view: put the lungs at the top

    fig, ax = plt.subplots(1, 3, figsize=(13, 5.4), gridspec_kw={"width_ratios": [0.8, 0.8, 1]})
    ext_c = [0, nx * sx, 0, nz * sz]  # coronal, origin lower: row 0 (header-inferior) at the bottom
    ax[0].imshow(window(ct[:, yc, :]), cmap="gray", origin="lower", extent=ext_c)
    ax[0].text(0.03, 0.97, "header orientation\n(lungs at the bottom)", transform=ax[0].transAxes, color="white",
               fontsize=8, va="top")
    cor = ct[:, yc, :][::-1] if flip else ct[:, yc, :]
    cp, cl = (panc[:, yc, :][::-1], les[:, yc, :][::-1]) if flip else (panc[:, yc, :], les[:, yc, :])
    ax[1].imshow(window(cor), cmap="gray", origin="lower", extent=ext_c)
    ax[1].contour(cp, [0.5], colors=PANC, linewidths=1, origin="lower", extent=ext_c)
    ax[1].contour(cl, [0.5], colors=LES, linewidths=1.5, origin="lower", extent=ext_c)
    ext_a = [0, nx * sx, ny * sy, 0]  # axial, viewed from below: anterior up, patient right on the viewer's left
    ax[2].imshow(window(ct[zc]), cmap="gray", extent=ext_a)
    ax[2].contour(panc[zc], [0.5], colors=PANC, linewidths=1, origin="upper", extent=ext_a)
    ax[2].contour(les[zc], [0.5], colors=LES, linewidths=1.5, origin="upper", extent=ext_a)
    for box, col, ls in ((oracle, ORACLE, "--"), (pred, PRED, "-")):
        (z0, z1), (y0, y1), (x0, x1) = box
        if flip:
            z0, z1 = nz - z1, nz - z0
        ax[1].add_patch(Rectangle((x0 * sx, z0 * sz), (x1 - x0) * sx, (z1 - z0) * sz, fill=False, ec=col, ls=ls, lw=1.6))
        ax[2].add_patch(Rectangle((x0 * sx, y0 * sy), (x1 - x0) * sx, (y1 - y0) * sy, fill=False, ec=col, ls=ls, lw=1.6))
    titles = ["a  Coronal, as stored", "b  Coronal, anatomical orientation", "c  Axial, through the tumour"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=10, loc="left")
        a.set_xticks([]); a.set_yticks([])
    for lab_, kw in (("pancreas (provided mask)", {"color": PANC}), ("PDAC (provided mask)", {"color": LES}),
                     ("oracle crop", {"color": ORACLE, "ls": "--"}),
                     ("crop from TotalSegmentator\n+ neighbouring organs", {"color": PRED})):
        ax[2].plot([], [], label=lab_, **kw)
    ax[2].legend(loc="lower right", fontsize=7, framealpha=0.85)
    scalebar(ax[0], nx * sx, nz * sz, False)
    scalebar(ax[1], nx * sx, nz * sz, False)
    scalebar(ax[2], nx * sx, ny * sy, False)  # y axis inverted: the bar sits at the top
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, state, "oracle", oracle, "pred", pred)


if __name__ == "__main__":
    main()
