#!/bin/bash
# External validation (MSD 194 + NIH 80, oracle ROI) for the tight and loose models, 5-fold ensemble, TTA.
# Tight data comes from the volume tar; loose test scans are uploaded to /root/Dataset700_Ts.tar
# (waits for /root/Dataset700_Ts.tar.done). Per-case CSVs -> /workspace/ext_eval/. Stops the pod at the end; 8h failsafe.
source /root/.runpod_api
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nnUNet_raw=/root/unused nnUNet_preprocessed=/root/unused
LOG=/workspace/ext_eval/run_external.log; E=/root/ext; mkdir -p /workspace/ext_eval $E
stop_pod() {
  for t in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: Mozilla/5.0")
    case "$code" in 200|201|204) echo "pod $POD_ID stopped ($code) $(date)" >> "$LOG"; return;; esac; sleep 10
  done
}
( sleep 28800; echo "FAILSAFE stop $(date)" >> "$LOG"; stop_pod ) &
PRED_ARGS="-c 3d_fullres -tr nnUNetTrainer_250epochs -f 0 1 2 3 4 --save_probabilities -npp 6 -nps 6"

echo "=== TIGHT: extract 274 external cases $(date) ===" > "$LOG"
T=Dataset701_PanoramaPDAC_tight
tail -n +2 /root/external_cases.csv | cut -d, -f1 | while read c; do echo "$T/imagesTr/${c}_0000.nii.gz"; echo "$T/labelsTr/${c}.nii.gz"; done > $E/tight_list.txt
tar xf /workspace/Dataset701_tight.tar -C $E --no-same-owner -T $E/tight_list.txt 2>>"$LOG"
echo "  images=$(ls $E/$T/imagesTr | wc -l) labels=$(ls $E/$T/labelsTr | wc -l) $(date)" >> "$LOG"
nnUNet_results=/workspace/nnunet_results_tight nnUNetv2_predict -i $E/$T/imagesTr -o $E/tight_out -d 701 $PRED_ARGS > $E/predict_tight.log 2>&1
echo "  predicted $(ls $E/tight_out/*.npz 2>/dev/null | wc -l) $(date)" >> "$LOG"
python3 /root/ext_eval.py $E/tight_out $E/$T/labelsTr /root/external_cases.csv /workspace/ext_eval/tight_external.csv >> "$LOG" 2>&1

echo "=== LOOSE: wait for upload $(date) ===" >> "$LOG"
while [ ! -f /root/Dataset700_Ts.tar.done ]; do sleep 30; done
tar xf /root/Dataset700_Ts.tar -C $E --no-same-owner
L=Dataset700_PanoramaPDAC
echo "  images=$(ls $E/$L/imagesTs | wc -l) labels=$(ls $E/$L/labelsTs | wc -l) $(date)" >> "$LOG"
cp /root/Dataset700_Ts.tar /workspace/ 2>>"$LOG"   # keep on the volume for future runs
nnUNet_results=/workspace/nnunet_results nnUNetv2_predict -i $E/$L/imagesTs -o $E/loose_out -d 700 $PRED_ARGS > $E/predict_loose.log 2>&1
echo "  predicted $(ls $E/loose_out/*.npz 2>/dev/null | wc -l) $(date)" >> "$LOG"
python3 /root/ext_eval.py $E/loose_out $E/$L/labelsTs /root/external_cases.csv /workspace/ext_eval/loose_external.csv >> "$LOG" 2>&1
echo "EXTERNAL_COMPLETE $(date)" >> "$LOG"
stop_pod
