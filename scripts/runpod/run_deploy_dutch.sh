#!/bin/bash
# Deployment-ROI on the Dutch cohort (1964 cases), pod-side, self-stopping.
# docs/deployment_and_mitigation_design.md. Raw scans come from the official Zenodo batches (MD5-checked);
# labels are the uploaded local copy (hash manifest); code = the repo's tools/ (same as local runs).
# Identity gate first: rebuilt oracle crops must equal the Dataset701 training crops voxel-for-voxel and
# reproduce the Dutch CV scores. Any failure -> save logs, stop the pod. 30 h failsafe.
# Expects in /root: work/ (bundle), labels_md5.txt, raw_md5_sample.txt, gate_cases.txt, .runpod_api, and either
# labels.tar (uploaded local copy) or nothing (labels are then cloned from GitHub; same hash manifest check).
set -u
source /root/.runpod_api
W=/root/work; V=/workspace/deploy_dutch; LOG=$V/run.log; mkdir -p $V
RAW=$W/data/raw/ct/panorama/images; LAB=$W/data/raw/ct/panorama_labels; mkdir -p $RAW $LAB
export TOTALSEG_HOME_DIR=$W/data/envs/totalseg PYTHONUNBUFFERED=1
mkdir -p $TOTALSEG_HOME_DIR   # TotalSegmentator creates only the last path component
cd $W
log() { echo "$* $(date -u +%H:%M:%S)" >> $LOG; }
save() {
  mkdir -p $V/results; cp -r $W/reports/deployment_roi/. $V/results/ 2>/dev/null
  for a in baseline_oof totalseg; do cp $W/data/processed/ct/stage1_masks/$a/_log*.csv $V/results/ 2>/dev/null
    cp $W/data/processed/ct/deploy_crops/${a}_dutch/skipped.txt $V/results/skipped_$a.txt 2>/dev/null; done
}
stop_pod() {
  for t in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: Mozilla/5.0")
    case "$code" in 200|201|204) log "pod $POD_ID stopped ($code)"; return;; esac; sleep 10
  done
}
fail() { log "FAIL: $*"; save; stop_pod; exit 1; }
( sleep 108000; log "FAILSAFE"; save; stop_pod ) &

log "=== [0] environment"
TV=$(python3 -c "import torch; print(torch.__version__)")
echo "torch==$TV" > /root/constraints.txt
pip install -q --break-system-packages --root-user-action=ignore -c /root/constraints.txt \
  nnunetv2==2.8.1 totalsegmentator==2.18.0 pandas psutil >> $V/pip.log 2>&1 || fail "pip install"
python3 -c "import torch,nnunetv2,totalsegmentator,importlib.metadata as m; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'nnunetv2',m.version('nnunetv2'),'totalseg',m.version('totalsegmentator'))" >> $LOG 2>&1 || fail "imports"

log "=== [1] labels (uploaded local copy) + models"
if [ -f /root/labels.tar ]; then tar xf /root/labels.tar -C $LAB --no-same-owner || fail "labels untar"
else
  git clone -q --depth 1 https://github.com/DIAGNijmegen/panorama_labels.git /root/pl || fail "labels clone"
  mv /root/pl/manual_labels /root/pl/automatic_labels $LAB/ || fail "labels move"
fi
( cd $LAB && md5sum -c --quiet /root/labels_md5.txt ) >> $LOG 2>&1 || fail "labels md5"
log "  labels md5 OK ($(wc -l < /root/labels_md5.txt) files)"
mkdir -p $W/models/nnunet/tight_cv
T=/workspace/nnunet_results_tight/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
D=$W/models/nnunet/tight_cv/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
for f in 0 1 2 3 4; do mkdir -p $D/fold_$f; cp $T/fold_$f/checkpoint_final.pth $D/fold_$f/ || fail "copy fold $f"; done
cp $T/plans.json $T/dataset.json $T/dataset_fingerprint.json $D/ || fail "copy model json"
curl -sL --retry 10 -o /root/ds103.zip "https://zenodo.org/records/11160381/files/Dataset103_PANORAMA_baseline_Pancreas_Segmentation.zip?download=1"
[ "$(md5sum /root/ds103.zip | cut -d' ' -f1)" = "d4c7e9666157e712f90649086ed395a5" ] || fail "baseline weights md5"
( cd $W/models/panorama_baseline && unzip -q /root/ds103.zip ) && rm /root/ds103.zip
log "  models OK"

log "=== [2] raw scans from Zenodo (4 batches in parallel, MD5, keep Dutch only)"
fetch() {  # name record md5
  local z=/root/$1.zip
  for attempt in 1 2 3; do
    curl -sL --retry 20 --retry-delay 5 -C - -o $z "https://zenodo.org/api/records/$2/files/$1.zip/content"
    [ "$(md5sum $z | cut -d' ' -f1)" = "$3" ] && break
    log "  $1 md5 mismatch (attempt $attempt)"; rm -f $z
  done
  [ -f $z ] || { log "  $1 FAILED"; return 1; }
  flock /root/extract.lock python3 - "$z" "$RAW" <<'PY' || return 1  # one extraction at a time (disk)
import os, sys, zipfile, csv
z, raw = sys.argv[1:3]
ext = {r["case"] for r in csv.DictReader(open("/root/work/reports/nnunet_summaries/external/external_cases.csv"))}
n = 0
with zipfile.ZipFile(z) as zf:
    for info in zf.infolist():
        base = os.path.basename(info.filename)
        if not base.endswith("_0000.nii.gz") or base[:12] in ext:
            continue
        with zf.open(info) as src, open(os.path.join(raw, base), "wb") as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
        n += 1
print(f"{os.path.basename(z)}: extracted {n} Dutch scans", flush=True)
PY
  rm -f $z
}
if [ -f /workspace/panorama_raw_dutch.tar ]; then   # saved by an earlier run: copy, don't re-download
  log "  using /workspace/panorama_raw_dutch.tar"
  cp /workspace/panorama_raw_dutch.tar /root/ && tar xf /root/panorama_raw_dutch.tar -C $W/data/raw/ct/panorama --no-same-owner \
    && rm /root/panorama_raw_dutch.tar || fail "raw tar from volume"
else
fetch batch_1 13715870 b3b3669a82696b954b449c27a9d85074 >> $LOG 2>&1 & F1=$!
fetch batch_2 13742336 9668a43c24d5eb3473fbaa979b1dbaf8 >> $LOG 2>&1 & F2=$!
fetch batch_3 11034011 9d852d09d750fd2e2a2e32a371d3bdd8 >> $LOG 2>&1 & F3=$!
fetch batch_4 10999754 f2820a214aa24fa90daeedbaf99d0609 >> $LOG 2>&1 & F4=$!
wait $F1 $F2 $F3 $F4   # never a bare `wait`: it would also wait for the failsafe timer
fi
N=$(ls $RAW | wc -l); log "  raw Dutch scans: $N"
[ "$N" -eq 1964 ] || fail "expected 1964 Dutch scans, got $N"
( cd $RAW && md5sum -c --quiet /root/raw_md5_sample.txt ) >> $LOG 2>&1 || fail "raw sample md5 vs local copy"
log "  raw sample md5 matches the local copy (30 files)"

log "=== [3] identity gate (24 cases): rebuilt oracle crops vs Dataset701, scores vs Dutch CV"
cp /workspace/Dataset701_tight.tar /root/ || fail "copy Dataset701 tar"
sed 's#.*#Dataset701_PanoramaPDAC_tight/imagesTr/&_0000.nii.gz#' /root/gate_cases.txt > /root/gate_list.txt
tar xf /root/Dataset701_tight.tar -C /root --no-same-owner -T /root/gate_list.txt || fail "gate untar"
rm /root/Dataset701_tight.tar
python3 tools/deploy_identity_gate.py masks /root/gate_cases.txt >> $LOG 2>&1 || fail "gate masks"
DEPLOY_CASES=/root/gate_cases.txt python3 tools/deploy_infer_oof.py oracle_rebuild > $V/gate_infer.log 2>&1 || fail "gate inference"
python3 tools/deploy_identity_gate.py check /root/gate_cases.txt /root/Dataset701_PanoramaPDAC_tight/imagesTr >> $LOG 2>&1 \
  || fail "IDENTITY GATE"
log "  identity gate PASSED"
C1=$(head -1 /root/gate_cases.txt)   # fetch TotalSegmentator weights once, before the parallel workers
python3 -c "from totalsegmentator.python_api import totalsegmentator as t; t('$RAW/${C1}_0000.nii.gz', '/root/ts_smoke', roi_subset=['pancreas'], quiet=True)" \
  >> $V/ts_smoke.log 2>&1 && [ -f /root/ts_smoke/pancreas.nii.gz ] || fail "TotalSegmentator smoke"
log "  TotalSegmentator weights + smoke OK"

log "=== [4] stage 1 (baseline OOF + TotalSegmentator x2) and stage 2 baseline as soon as its masks exist"
( STAGE1_THREADS=4 python3 tools/stage1_segment.py baseline dutch > $V/s1_baseline.log 2>&1
  python3 tools/stage1_quality.py baseline_oof dutch > $V/q_baseline.log 2>&1 & Q=$!
  DEPLOY_THREADS=6 python3 tools/deploy_infer_oof.py baseline_oof > $V/s2_baseline.log 2>&1
  wait $Q ) &
CHAIN=$!
STAGE1_SHARD=0/2 STAGE1_THREADS=6 python3 tools/stage1_segment.py totalseg dutch > $V/s1_ts0.log 2>&1 &
T0=$!
STAGE1_SHARD=1/2 STAGE1_THREADS=6 python3 tools/stage1_segment.py totalseg dutch > $V/s1_ts1.log 2>&1 &
T1=$!
wait $T0 $T1
log "  totalseg masks: $(ls $W/data/processed/ct/stage1_masks/totalseg/*.nii.gz | wc -l)"
python3 tools/stage1_quality.py totalseg dutch > $V/q_totalseg.log 2>&1 & QT=$!
wait $CHAIN
log "  baseline masks: $(ls $W/data/processed/ct/stage1_masks/baseline_oof/*.nii.gz | wc -l); baseline stage 2 rows: $(($(wc -l < $W/reports/deployment_roi/detect_baseline_oof_dutch.csv)-1))"
save

log "=== [5] stage 2 TotalSegmentator"
DEPLOY_THREADS=8 python3 tools/deploy_infer_oof.py totalseg > $V/s2_totalseg.log 2>&1 || log "  totalseg stage 2 exited non-zero"
wait $QT
log "  totalseg stage 2 rows: $(($(wc -l < $W/reports/deployment_roi/detect_totalseg_dutch.csv)-1))"
save
tar czf $V/stage1_masks_dutch.tgz -C $W/data/processed/ct/stage1_masks baseline_oof totalseg
log "DEPLOY_DUTCH_COMPLETE"
stop_pod
