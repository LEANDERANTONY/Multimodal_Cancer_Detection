import json, os, numpy as np
base="/workspace/nnunet_results_lomo/Dataset700_PanoramaPDAC/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
names=["Siemens","Toshiba","Philips"]
print("=== LOMO holdout results (Dice on the held-out manufacturer, never trained on) ===")
for f in range(3):
    p=os.path.join(base,f"fold_{f}","validation","summary.json")
    if not os.path.exists(p):
        print(f"holdout {names[f]}: (summary not written yet)"); continue
    s=json.load(open(p))
    pos=np.array([c["metrics"]["1"]["Dice"] for c in s["metric_per_case"] if c["metrics"]["1"]["n_ref"]>0],float)
    print(f"holdout {names[f]:8s}: pos={len(pos):4d} meanDice={np.nanmean(pos):.4f} median={np.nanmedian(pos):.4f} >0.5={int((pos>0.5).sum())}")
print("Compare vs 5-fold CV mean ~0.33: a large DROP here = model was leaning on manufacturer signal (the confound).")
