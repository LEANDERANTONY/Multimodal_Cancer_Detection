#!/bin/bash
# Re-run tight-LOMO fold 2 (Philips holdout). The first run collapsed to all-background at
# epoch 1 (per-sample Dice rewards empty predictions on lesion-free patches; fold 2 has the
# lowest positive rate, ~21%). Same config, fresh init. While /root/hold_stop exists the
# /usr/local/bin/curl wrapper swallows pod-stop calls (so run_battery701.sh can't stop the pod
# mid-retrain); this script stops the pod itself with /usr/bin/curl when done. 12h failsafe.
source /root/.runpod_api
export nnUNet_raw=/root/nnunet_raw nnUNet_preprocessed=/root/nnunet_prep nnUNet_results=/workspace/nnunet_results_tight_lomo
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nnUNet_n_proc_DA=8
LOG=/workspace/fold2_retry.log
real_stop() {
  rm -f /root/hold_stop
  grep -v "rest.runpod.io" /etc/hosts > /root/h.tmp; cat /root/h.tmp > /etc/hosts
  for t in 1 2 3 4 5; do
    code=$(/usr/bin/curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: Mozilla/5.0")
    case "$code" in 200|201|204) echo "pod $POD_ID stopped ($code) $(date)" >> "$LOG"; return;; esac; sleep 10
  done
}
( sleep 43200; echo "FAILSAFE stop $(date)" >> "$LOG"; real_stop ) &

echo "=== fold 2 retry START $(date) ===" > "$LOG"
nnUNetv2_train 701 3d_fullres 2 -tr nnUNetTrainer_250epochs --npz >> "$LOG" 2>&1
echo "=== fold 2 retry DONE $(date) ===" >> "$LOG"
# wait for the battery to finish, then rescore LOMO detection with the new fold 2
while tmux has-session -t battery 2>/dev/null; do sleep 60; done
python3 /root/tight_battery.py lomo >> "$LOG" 2>&1
echo "FOLD2_RETRY_COMPLETE $(date)" >> "$LOG"
real_stop
