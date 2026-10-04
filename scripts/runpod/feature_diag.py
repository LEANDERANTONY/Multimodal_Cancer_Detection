"""Feature-space scanner confound diagnostic for the trained 3D nnU-Net (Dataset700, fold_0).

Q: does the model's internal representation ENCODE scanner (acquisition signature),
even though its detection SCORE (p_sum) didn't ride the between-scanner shortcut?

Method: extract the encoder bottleneck (320-d, global-avg-pooled over the centre
patch) for a scanner-balanced subset of cases, then see how well a linear probe
recovers (a) scanner manufacturer and (b) the PDAC label from those features,
with PATIENT-GROUPED CV. High scanner-AUROC = features carry scanner; compare to
how well the SAME features carry cancer. A label-shuffle control checks the probe.
Reads only; nothing on the volume is modified.
"""
import json, os, sys, time
import numpy as np
import torch

MF = "/workspace/nnunet_results/Dataset700_PanoramaPDAC/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
IMG = "/root/d700/Dataset700_PanoramaPDAC/imagesTr"
LOMO = "/workspace/splits_final_lomo.json"
PER_SCANNER = 250   # cap per manufacturer for balance/speed
SEED = 20261001


def center_patch(data, ps):
    # data: (C, *spatial) -> centre crop/pad to patch size ps
    out = data
    pads = []
    sl = [slice(None)]
    for i, p in enumerate(ps):
        d = out.shape[i + 1]
        if d >= p:
            lo = (d - p) // 2
            sl.append(slice(lo, lo + p)); pads.append((0, 0))
        else:
            sl.append(slice(0, d)); before = (p - d) // 2; pads.append((before, p - d - before))
    out = out[tuple(sl)]
    out = np.pad(out, [(0, 0)] + pads, mode="constant", constant_values=out.min())
    return out


def main():
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    pred = nnUNetPredictor(device=torch.device("cuda"), allow_tqdm=False)
    pred.initialize_from_trained_model_folder(MF, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    net = pred.network.to("cuda").eval()
    pm, cm = pred.plans_manager, pred.configuration_manager
    dj = pred.dataset_json
    ps = cm.patch_size
    pp = DefaultPreprocessor()

    # scanner map (3 majors) from LOMO splits; label from the 5 CV fold summaries
    sp = json.load(open(LOMO))
    scanner = {}
    for i, nm in enumerate(["Siemens", "Toshiba", "Philips"]):
        for c in sp[i]["val"]:
            scanner[c] = nm
    label = {}
    for f in range(5):
        s = json.load(open(f"{MF}/fold_{f}/validation/summary.json"))
        for c in s["metric_per_case"]:
            cid = os.path.basename(c["prediction_file"]).replace(".nii.gz", "")
            label[cid] = 1 if c["metrics"]["1"]["n_ref"] > 0 else 0

    have = {c for c in scanner if os.path.exists(f"{IMG}/{c}_0000.nii.gz")}
    rng = np.random.default_rng(SEED)
    sel = []
    for nm in ["Siemens", "Toshiba", "Philips"]:
        cs = sorted(c for c in have if scanner[c] == nm)
        rng.shuffle(cs); sel += cs[:PER_SCANNER]
    print(f"selected {len(sel)} cases ({[ (nm, sum(scanner[c]==nm for c in sel)) for nm in ['Siemens','Toshiba','Philips'] ]})", flush=True)

    feats, scs, labs, pats, t0 = [], [], [], [], time.time()
    for i, c in enumerate(sel):
        try:
            data, _, _ = pp.run_case([f"{IMG}/{c}_0000.nii.gz"], None, pm, cm, dj)
            patch = center_patch(data, ps)
            x = torch.from_numpy(patch[None]).float().to("cuda")
            with torch.no_grad():
                bott = net.encoder(x)[-1]            # (1,320,d,h,w)
                v = bott.mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()  # (320,)
            feats.append(v); scs.append(scanner[c]); labs.append(label.get(c, -1)); pats.append(c.split("_")[0])
        except Exception as e:
            print(f"  skip {c}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sel)} ({time.time()-t0:.0f}s)", flush=True)

    X = np.array(feats); scs = np.array(scs); labs = np.array(labs); pats = np.array(pats)
    np.savez("/workspace/feature_diag_features.npz", X=X, scanner=scs, label=labs, patient=pats)
    print(f"\nfeatures: {X.shape}  (saved to volume)", flush=True)

    # ---- linear probes, patient-grouped CV ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    def probe(Xp, y, grp, name, shuffle=False):
        yy = y.copy()
        if shuffle:
            yy = np.random.default_rng(0).permutation(yy)
        classes = np.unique(yy)
        oof = np.zeros((len(yy), len(classes)))
        for tr, te in StratifiedGroupKFold(5).split(Xp, yy, grp):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(Xp[tr], yy[tr])
            oof[te] = clf.predict_proba(Xp[te])
        auc = (roc_auc_score(yy, oof[:, 1]) if len(classes) == 2
               else roc_auc_score(yy, oof, multi_class="ovr", average="macro"))
        print(f"  {name:40s} AUROC={auc:.3f}  (n={len(yy)}, classes={list(classes)})", flush=True)
        return auc

    print("\n=== linear probe on nnU-Net bottleneck features (patient-grouped 5-fold) ===")
    probe(X, scs, pats, "SCANNER (manufacturer, macro-OVR)")
    probe(X, scs, pats, "  scanner, label-shuffled control", shuffle=True)
    m = labs >= 0
    if m.sum() > 50:
        probe(X[m], labs[m], pats[m], "PDAC label (cancer, from same features)")
    print("FEATURE_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
