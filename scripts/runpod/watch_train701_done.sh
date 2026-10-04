#!/usr/bin/env bash
# Wait for tight CV 5-fold to complete; grab per-fold pos-case Dice before the pod auto-stops.
KEY="$HOME/.ssh/id_ed25519"
SSH_ARGS=(-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -p 40044 -i "$KEY")
HOST="root@213.192.2.110"
BASE=/workspace/nnunet_results_tight/Dataset701_PanoramaPDAC_tight/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
rc(){ ssh "${SSH_ARGS[@]}" "$HOST" "$@" 2>/dev/null; }
sshfail=0
for i in $(seq 1 1400); do   # ~23h at 60s
  OUT=$(rc "echo OK; grep -c TIGHT_TRAIN_COMPLETE /workspace/train701.log 2>/dev/null; grep -c ABORT /workspace/train701.log 2>/dev/null")
  if ! echo "$OUT" | grep -q OK; then
    sshfail=$((sshfail+1))
    if [ $sshfail -ge 8 ]; then echo "POD UNREACHABLE x8 — tight CV likely finished + auto-stopped; results on volume ($BASE/fold_*/validation/summary.json)."; exit 0; fi
    sleep 60; continue
  fi
  sshfail=0
  if [ "$(echo "$OUT"|sed -n '3p')" -ge 1 ] 2>/dev/null; then echo "=== ABORTED ==="; rc "grep -A2 ABORT /workspace/train701.log"; exit 0; fi
  if [ "$(echo "$OUT"|sed -n '2p')" -ge 1 ] 2>/dev/null; then
    echo "=== TIGHT CV 5-FOLD COMPLETE — per-fold positive-case Dice ==="
    rc "python3 -c \"
import json,glob,os,numpy as np
b='$BASE'
for f in range(5):
    p=f'{b}/fold_{f}/validation/summary.json'
    if not os.path.exists(p): print(f'fold {f}: (no summary)'); continue
    s=json.load(open(p)); pos=[c['metrics']['1']['Dice'] for c in s['metric_per_case'] if c['metrics']['1']['n_ref']>0]
    print(f'fold {f}: pos={len(pos)} meanDice={np.nanmean(pos):.4f} median={np.nanmedian(pos):.4f}')
\""
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT"; rc "tail -15 /workspace/train701.log"; exit 4
