#!/bin/bash
# Wait for staging, verify the 1964-Dutch count, preprocess, train tight CV 5-fold, auto-stop.
source /root/.runpod_api
export nnUNet_raw=/root/nnunet_raw nnUNet_preprocessed=/root/nnunet_prep nnUNet_results=/workspace/nnunet_results_tight
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nnUNet_n_proc_DA=8
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"
LOG=/workspace/train701.log
D="$nnUNet_raw/Dataset701_PanoramaPDAC_tight"
echo "=== waiting for SETUP701_DONE ===" > "$LOG"
for i in $(seq 1 240); do grep -q SETUP701_DONE /root/setup701.log 2>/dev/null && break; sleep 15; done
n=$(ls "$D/labelsTr" 2>/dev/null | wc -l)
echo "=== setup done; numTraining=$n (expect 1964 Dutch) ===" >> "$LOG"
if [ "$n" -ne 1964 ]; then
  echo "ABORT: expected 1964, got $n — not training" >> "$LOG"
else
  echo "=== plan_and_preprocess $(date) ===" >> "$LOG"
  nnUNetv2_plan_and_preprocess -d 701 -c 3d_fullres --verify_dataset_integrity -np 8 >> "$LOG" 2>&1
  for f in 0 1 2 3 4; do
    echo "=== TRAIN fold $f START $(date) ===" >> "$LOG"
    nnUNetv2_train 701 3d_fullres $f -tr nnUNetTrainer_250epochs --npz >> "$LOG" 2>&1
    echo "=== TRAIN fold $f DONE $(date) ===" >> "$LOG"
  done
  echo "TIGHT_TRAIN_COMPLETE $(date)" >> "$LOG"
fi
# auto-stop regardless of success/abort so the pod cannot idle
for t in 1 2 3 4 5; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: Mozilla/5.0")
  case "$code" in 200|201|204) echo "pod $POD_ID stopped ($code)" >> "$LOG"; break;; esac; sleep 10
done
