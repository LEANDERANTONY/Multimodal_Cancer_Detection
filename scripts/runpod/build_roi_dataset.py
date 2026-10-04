"""Parameterised ROI-crop builder for the PANORAMA nnU-Net dataset.

ROI = bounding box of the provided pancreas(4)+duct(5) masks (label-blind),
expanded by a fixed margin (mm, PER SIDE), cropped at native resolution.
Training label = PDAC lesion (mask==1). Never uses the lesion mask for the ROI.

Loose (current Dataset700): margin 150x100x40 mm.  Tight (field/PanDx): 100x50x15 mm.
One flag switches them -> both datasets regenerable (the original builder was lost).

Modes:
  --dry N        : calibration — process first N cases, WRITE NOTHING, print crop
                   sizes (mm/vox) and lesion-containment. Use to validate convention + gate.
  (full)         : write nnU-Net raw (imagesTr/labelsTr + dataset.json).
Machine-safe: one case at a time, capped threads, below-normal priority, progress, resumable.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import nibabel as nib

IMG = "data/raw/ct/panorama/images"
LAB_MAN = "data/raw/ct/panorama_labels/manual_labels"
LAB_AUTO = "data/raw/ct/panorama_labels/automatic_labels"


def mask_path(case):
    p = f"{LAB_MAN}/{case}.nii.gz"
    return p if os.path.exists(p) else f"{LAB_AUTO}/{case}.nii.gz"


def bbox_from_roi(roi):
    idx = np.where(roi)
    return [(int(a.min()), int(a.max()) + 1) for a in idx]  # [(lo,hi)] per axis, hi exclusive


def expand_clip(bb, margin_vox, shape):
    out = []
    for (lo, hi), m, s in zip(bb, margin_vox, shape):
        out.append((max(0, lo - m), min(s, hi + m)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-mm", nargs=3, type=float, required=True, help="per-side margin x y z (mm)")
    ap.add_argument("--out", default="", help="nnU-Net raw dataset dir (full mode)")
    ap.add_argument("--dry", type=int, default=0, help="calibration: first N cases, write nothing")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass  # priority best-effort

    cases = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(IMG) if f.endswith("_0000.nii.gz"))
    if args.dry:
        cases = cases[: args.dry]
    margin = tuple(args.margin_mm)
    print(f"margin(mm, per-side)={margin}  cases={len(cases)}  dry={bool(args.dry)}", flush=True)

    qc = None
    if not args.dry:
        assert args.out, "--out required in full mode"
        os.makedirs(f"{args.out}/imagesTr", exist_ok=True)
        os.makedirs(f"{args.out}/labelsTr", exist_ok=True)
        qc = open(f"{args.out}/roi_build_qc.csv", "w")
        qc.write("case,provenance,crop_x_mm,crop_y_mm,crop_z_mm,lesion_vox,containment,clipped\n")

    sizes, contain, clipped, t0 = [], [], 0, time.time()
    for i, c in enumerate(cases):
        if not args.dry and os.path.exists(f"{args.out}/imagesTr/{c}_0000.nii.gz") \
                and os.path.exists(f"{args.out}/labelsTr/{c}.nii.gz"):
            continue  # resume: already built
        mp = mask_path(c)
        m = nib.load(mp); sp = np.array(m.header.get_zooms()[:3], float)
        ma = np.asarray(m.dataobj)
        roi = (ma == 4) | (ma == 5)
        if not roi.any():
            print(f"  [{c}] WARN no pancreas/duct mask -> skip", flush=True); continue
        bb = bbox_from_roi(roi)
        margin_vox = [int(np.ceil(mm / s)) for mm, s in zip(margin, sp)]
        cb = expand_clip(bb, margin_vox, ma.shape)
        sz_mm = tuple(round((hi - lo) * s, 0) for (lo, hi), s in zip(cb, sp))
        sizes.append(sz_mm)
        # lesion containment: fraction of lesion voxels inside the crop box
        les = (ma == 1)
        nl = int(les.sum())
        if nl:
            sl = tuple(slice(lo, hi) for lo, hi in cb)
            inside = int(les[sl].sum())
            frac = inside / nl
            contain.append(frac)
            if frac < 0.999:
                clipped += 1
        if args.dry:
            prov = "man" if mp.startswith(LAB_MAN) else "auto"
            print(f"  [{c}] {prov} crop_mm={sz_mm} vox={tuple(hi-lo for lo,hi in cb)} "
                  f"lesion_vox={nl} contain={'%.3f'%(contain[-1]) if nl else 'NA(neg)'}", flush=True)
            continue
        # full write
        im = nib.load(f"{IMG}/{c}_0000.nii.gz")
        sl = tuple(slice(lo, hi) for lo, hi in cb)
        img_c = np.asarray(im.dataobj)[sl]
        les_c = les[sl].astype(np.uint8)
        aff = im.affine.copy()
        aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([cb[0][0], cb[1][0], cb[2][0]])
        nib.save(nib.Nifti1Image(img_c.astype(np.int16), aff), f"{args.out}/imagesTr/{c}_0000.nii.gz")
        nib.save(nib.Nifti1Image(les_c, aff), f"{args.out}/labelsTr/{c}.nii.gz")
        prov = "man" if mp.startswith(LAB_MAN) else "auto"
        frac = f"{contain[-1]:.4f}" if nl else ""
        clip = 1 if (nl and contain[-1] < 0.999) else 0
        qc.write(f"{c},{prov},{sz_mm[0]:.0f},{sz_mm[1]:.0f},{sz_mm[2]:.0f},{nl},{frac},{clip}\n"); qc.flush()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)} ({time.time()-t0:.0f}s)", flush=True)

    if not args.dry:
        import glob
        n = len(glob.glob(f"{args.out}/labelsTr/*.nii.gz"))
        json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "lesion": 1},
                   "numTraining": n, "file_ending": ".nii.gz"},
                  open(f"{args.out}/dataset.json", "w"), indent=2)
        qc.close()
        print(f"wrote dataset.json (numTraining={n}) + roi_build_qc.csv", flush=True)

    if sizes:
        arr = np.array(sizes)
        print(f"\ncrop size mm: median={np.median(arr,0)} min={arr.min(0)} max={arr.max(0)}")
    if contain:
        cc = np.array(contain)
        print(f"lesion containment: n_pos={len(cc)} mean={cc.mean():.4f} "
              f"min={cc.min():.3f} full(>=0.999)={int((cc>=0.999).sum())}/{len(cc)} clipped={clipped}")
    print("BUILD_DONE" if not args.dry else "DRY_DONE", flush=True)


if __name__ == "__main__":
    main()
