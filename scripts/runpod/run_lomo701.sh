#!/bin/bash
# Tight-crop LOMO: wait upload, stage to 1964 Dutch, preprocess, install manufacturer-holdout
# splits (fold0=Siemens,1=Toshiba,2=Philips), train 3 folds, auto-stop. Count-gated.
source /root/.runpod_api
export nnUNet_raw=/root/nnunet_raw nnUNet_preprocessed=/root/nnunet_prep nnUNet_results=/workspace/nnunet_results_tight_lomo
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nnUNet_n_proc_DA=8
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
LOG=/workspace/lomo701.log
D="$nnUNet_raw/Dataset701_PanoramaPDAC_tight"
echo "=== waiting for /root/upload_done ===" > "$LOG"
for i in $(seq 1 600); do [ -f /root/upload_done ] && break; sleep 15; done
echo "=== extracting $(date) ===" >> "$LOG"
tar -xf /root/Dataset701_tight.tar -C "$nnUNet_raw" --no-same-owner 2>>"$LOG"
python3 - <<'PY' >> "$LOG" 2>&1
import os, glob, json
D = '/root/nnunet_raw/Dataset701_PanoramaPDAC_tight'
dutch = set(open('/root/dutch_cases.txt').read().split())
rm = 0
for p in glob.glob(f'{D}/imagesTr/*_0000.nii.gz'):
    c = os.path.basename(p)[:-len('_0000.nii.gz')]
    if c not in dutch:
        os.remove(p)
        lp = f'{D}/labelsTr/{c}.nii.gz'
        if os.path.exists(lp): os.remove(lp)
        rm += 1
n = len(glob.glob(f'{D}/labelsTr/*.nii.gz'))
dj = json.load(open(f'{D}/dataset.json')); dj['numTraining'] = n
json.dump(dj, open(f'{D}/dataset.json', 'w'), indent=2)
print(f'removed {rm} non-Dutch; numTraining={n}')
PY
n=$(ls "$D/labelsTr" 2>/dev/null | wc -l)
echo "=== staged numTraining=$n (expect 1964) ===" >> "$LOG"
if [ "$n" -ne 1964 ]; then
  echo "ABORT: expected 1964 Dutch, got $n" >> "$LOG"
else
  echo "=== preprocess $(date) ===" >> "$LOG"
  nnUNetv2_plan_and_preprocess -d 701 -c 3d_fullres --verify_dataset_integrity -np 8 >> "$LOG" 2>&1
  python3 - <<'PY' >> "$LOG" 2>&1
import json
dutch = set(open('/root/dutch_cases.txt').read().split())
sp = json.load(open('/workspace/splits_final_lomo.json'))
out = []
for i, fold in enumerate(sp):
    tr = [c for c in fold['train'] if c in dutch]
    va = [c for c in fold['val'] if c in dutch]
    out.append({'train': tr, 'val': va})
    print(f'LOMO fold {i}: train={len(tr)} val={len(va)} (dropped tr {len(fold["train"])-len(tr)}, val {len(fold["val"])-len(va)})')
pp = '/root/nnunet_prep/Dataset701_PanoramaPDAC_tight/splits_final.json'
json.dump(out, open(pp, 'w'), indent=2); print('wrote', pp)
PY
  for f in 0 1 2; do
    echo "=== LOMO fold $f START $(date) ===" >> "$LOG"
    nnUNetv2_train 701 3d_fullres $f -tr nnUNetTrainer_250epochs --npz >> "$LOG" 2>&1
    echo "=== LOMO fold $f DONE $(date) ===" >> "$LOG"
  done
  echo "TIGHT_LOMO_COMPLETE $(date)" >> "$LOG"
fi
for t in 1 2 3 4 5; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: Mozilla/5.0")
  case "$code" in 200|201|204) echo "pod $POD_ID stopped ($code)" >> "$LOG"; break;; esac; sleep 10
done
