"""Assemble the model-weight release for Zenodo (local staging only; the author uploads).

Contents (licence CC BY-NC 4.0, following the PANORAMA data licence):
  nnunet_tight_cv/   the five tight 3D nnU-Net cross-validation models (checkpoint_final.pth per fold, plans, dataset
                     json), as used for every 3D result and the deployment experiment
  panel_erm_2p5d/    the 15 ERM 2.5D soups (3 seeds x 5 CV folds) of the mitigation panel
plus README.md and SHA256SUMS. Code, per-case scores and result tables are released from the GitHub repository at a
tagged commit; the list of scans stored head-to-foot is reports/deployment_roi/orientation_check.csv.

    python tools/build_release.py [--zip]
Writes data/release/panorama_confound_audit_weights/ (git-ignored) and, with --zip, one zip per part.
"""
import hashlib
import os
import shutil
import sys
import zipfile

OUT = "data/release/panorama_confound_audit_weights"
NNU = "models/nnunet/tight_cv/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
PANEL = "data/panel_runs/erm_resnet18_d20_s{s}/cv{k}/soup.pt"

README = """# Model weights: confounding audit of AI detection of pancreatic cancer on CT (PANORAMA)

Weights accompanying the paper "Present, weakly exploited, and fragile at deployment: an end-to-end audit of
confounding in AI detection of pancreatic cancer on CT" (A. Leander Antony). Code: [GitHub URL, tag].

Licence: CC BY-NC 4.0 (non-commercial), because the models were trained on the PANORAMA public training and development
dataset, which is released under CC BY-NC 4.0. Cite the PANORAMA study (Alves et al., Lancet Oncol 2026) and the
dataset when using them. Research use only; not a medical device.

## nnunet_tight_cv/
nnU-Net v2 (2.8.1) 3d_fullres models, trainer nnUNetTrainer_250epochs, five folds of nnU-Net's default split of the
1,964 Dutch PANORAMA scans. Input: a pancreas-centred crop (bounding box of the pancreas and duct masks + 100 x 50 x 15 mm
per side, native resolution); output: PDAC lesion segmentation. Place the folder as
nnUNet_results/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres and run
nnUNetv2_predict with -d 701 -c 3d_fullres -tr nnUNetTrainer_250epochs.

## panel_erm_2p5d/
2.5D ResNet18 classifiers (slice triplets, attention pooling), empirical risk minimisation, seeds 0-2 x CV folds 0-4,
uniform soup of the last three epochs. Load with tools/panel_train.py (class Net, backbone resnet18). Inputs are built by
tools/panel_build_slices.py (LPS, 1 x 1 x 2 mm, nnU-Net CT normalisation). Note: these models respond to the
cranio-caudal orientation of the input (see the paper); orient scans by anatomy before use.

SHA256SUMS lists a checksum for every file.
"""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = []
    for name in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        if os.path.exists(f"{NNU}/{name}"):
            files.append((f"{NNU}/{name}", f"nnunet_tight_cv/{name}"))
    for k in range(5):
        for name in ("checkpoint_final.pth", "debug.json"):
            files.append((f"{NNU}/fold_{k}/{name}", f"nnunet_tight_cv/fold_{k}/{name}"))
    for s in range(3):
        for k in range(5):
            files.append((PANEL.format(s=s, k=k), f"panel_erm_2p5d/seed{s}_cv{k}_soup.pt"))
    missing = [a for a, _ in files if not os.path.exists(a)]
    assert not missing, missing
    for src, dst in files:
        d = f"{OUT}/{dst}"
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(src):
            shutil.copy2(src, d)
    open(f"{OUT}/README.md", "w", encoding="utf-8").write(README)
    with open(f"{OUT}/SHA256SUMS", "w") as f:
        for _, dst in files:
            f.write(f"{sha(f'{OUT}/{dst}')}  {dst}\n")
    size = sum(os.path.getsize(f"{OUT}/{dst}") for _, dst in files) / 1e9
    print(f"{len(files)} files, {size:.2f} GB in {OUT}")
    if "--zip" in sys.argv:
        for part in ("nnunet_tight_cv", "panel_erm_2p5d"):
            with zipfile.ZipFile(f"{OUT}/../{part}.zip", "w", zipfile.ZIP_STORED) as z:
                for root, _, fs in os.walk(f"{OUT}/{part}"):
                    for fn in fs:
                        z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), OUT))
            print("zipped", part)


if __name__ == "__main__":
    main()
