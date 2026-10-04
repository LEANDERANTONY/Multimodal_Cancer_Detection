#!/bin/bash
# Post-LOMO confound battery for the tight model, then pod stop.
# Before launch, rest.runpod.io is blocked in /etc/hosts so run_lomo701.sh's own auto-stop
# fails harmlessly; this script unblocks and stops the pod when the battery is done.
# A 14h failsafe stops the pod regardless.
source /root/.runpod_api
LOG=/workspace/battery701.log
S=/root/tight_battery.py
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
stop_pod() {
  grep -v "rest.runpod.io" /etc/hosts > /root/h.tmp; cat /root/h.tmp > /etc/hosts
  for t in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: Mozilla/5.0")
    case "$code" in 200|201|204) echo "pod $POD_ID stopped ($code) $(date)" >> "$LOG"; return;; esac; sleep 10
  done
}
( sleep 50400; echo "FAILSAFE stop $(date)" >> "$LOG"; stop_pod ) &

echo "=== battery waiting for LOMO driver $(date) ===" > "$LOG"
while tmux has-session -t lomo 2>/dev/null; do sleep 60; done
if grep -q TIGHT_LOMO_COMPLETE /workspace/lomo701.log; then
  echo "=== [1/3] LOMO per-scanner detection $(date) ===" >> "$LOG"
  python3 $S lomo >> "$LOG" 2>&1

  echo "=== [2/3] tight-CV re-inference + confound tax $(date) ===" >> "$LOG"
  for f in 0 1 2 3 4; do
    mkdir -p /root/cvin/fold_$f /root/cvout/fold_$f
    python3 - "$f" <<'PY'
import json, os, sys
f = sys.argv[1]
s = json.load(open(f"/workspace/nnunet_results_tight/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_{f}/validation/summary.json"))
for c in s["metric_per_case"]:
    cid = os.path.basename(c["prediction_file"]).replace(".nii.gz", "")
    dst = f"/root/cvin/fold_{f}/{cid}_0000.nii.gz"
    if not os.path.exists(dst):
        os.symlink(f"/root/nnunet_raw/Dataset701_PanoramaPDAC_tight/imagesTr/{cid}_0000.nii.gz", dst)
PY
    echo "  fold $f: $(ls /root/cvin/fold_$f | wc -l) cases $(date)" >> "$LOG"
    nnUNet_results=/workspace/nnunet_results_tight nnUNet_raw=/root/nnunet_raw nnUNet_preprocessed=/root/nnunet_prep \
      nnUNetv2_predict -i /root/cvin/fold_$f -o /root/cvout/fold_$f -d 701 -c 3d_fullres \
      -tr nnUNetTrainer_250epochs -f $f --save_probabilities -npp 6 -nps 6 > /root/predict_$f.log 2>&1
    echo "  fold $f predicted: $(ls /root/cvout/fold_$f/*.npz 2>/dev/null | wc -l) npz" >> "$LOG"
  done
  python3 $S cv >> "$LOG" 2>&1

  echo "=== [3/3] feature probe $(date) ===" >> "$LOG"
  python3 $S feat >> "$LOG" 2>&1
  echo "TIGHT_BATTERY_COMPLETE $(date)" >> "$LOG"
else
  echo "LOMO did not complete; skipping battery" >> "$LOG"
fi
stop_pod
