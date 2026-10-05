#!/bin/bash
# Per-fold external predictions for the TIGHT model, so external operating points can be set from the
# same single model's Dutch CV scores (the 5-fold ensemble smooths probabilities, so CV-OOF thresholds
# don't transfer to ensemble scores). Waits for run_external.sh to finish (tmux `ext` gone).
# While /root/hold_stop exists, the /usr/local/bin/curl wrapper swallows pod-stop calls (so
# run_external.sh can't stop the pod first); this script stops it with /usr/bin/curl. 8h failsafe.
source /root/.runpod_api
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nnUNet_raw=/root/unused nnUNet_preprocessed=/root/unused
LOG=/workspace/ext_eval/run_external_perfold.log; E=/root/ext; T=Dataset701_PanoramaPDAC_tight
real_stop() {
  rm -f /root/hold_stop
  for t in 1 2 3 4 5; do
    code=$(/usr/bin/curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: Mozilla/5.0")
    case "$code" in 200|201|204) echo "pod $POD_ID stopped ($code) $(date)" >> "$LOG"; return;; esac; sleep 10
  done
}
( sleep 28800; echo "FAILSAFE stop $(date)" >> "$LOG"; real_stop ) &
echo "=== per-fold: waiting for run_external.sh $(date) ===" > "$LOG"
while tmux has-session -t ext 2>/dev/null; do sleep 60; done
for f in 0 1 2 3 4; do
  echo "=== tight fold $f $(date) ===" >> "$LOG"
  nnUNet_results=/workspace/nnunet_results_tight nnUNetv2_predict -i $E/$T/imagesTr -o $E/tight_f$f -d 701 \
    -c 3d_fullres -tr nnUNetTrainer_250epochs -f $f --save_probabilities -npp 6 -nps 6 > $E/predict_tight_f$f.log 2>&1
  python3 /root/ext_eval.py $E/tight_f$f $E/$T/labelsTr /root/external_cases.csv /workspace/ext_eval/tight_external_fold$f.csv >> "$LOG" 2>&1
  rm -rf $E/tight_f$f
done
echo "PERFOLD_COMPLETE $(date)" >> "$LOG"
real_stop
