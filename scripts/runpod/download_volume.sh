#!/usr/bin/env bash
# Mirror everything worth keeping from the RunPod volume to data/nnunet_checkpoints/ (offline copy).
# Skips *.npz softmax (regenerable from checkpoints; per-case scores already saved) and the
# Dataset701 tar (already local). One tar stream per folder, retried, so a drop only redoes that folder.
# Run in the USER's terminal (agent background tasks get killed):  bash download_volume.sh <ip> <port>
H="root@$1"; P="$2"; K="$HOME/.ssh/id_ed25519"
DEST="/d/Documents/Projects/Multimodal_Cancer_Detection/data/nnunet_checkpoints"
SSH=(ssh -p "$P" -i "$K" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
mkdir -p "$DEST/volume_misc"
for d in nnunet_results nnunet_results_lomo nnunet_results_tight nnunet_results_tight_lomo; do
  for try in 1 2 3 4 5; do
    echo "$(date +%H:%M:%S) == $d (try $try)"
    "${SSH[@]}" "$H" "cd /workspace && tar cf - --exclude='*.npz' $d" | tar xf - -C "$DEST" && break
    echo "   interrupted, retrying"; sleep 5
  done
done
echo "$(date +%H:%M:%S) == misc files"
"${SSH[@]}" "$H" "cd /workspace && tar cf - --exclude='*.tar' --exclude='nnunet_results*' --exclude='lost+found' ." | tar xf - -C "$DEST/volume_misc"
echo "== verify: remote vs local file counts and bytes (excluding npz)"
for d in nnunet_results nnunet_results_lomo nnunet_results_tight nnunet_results_tight_lomo; do
  r=$("${SSH[@]}" "$H" "cd /workspace && find $d -type f ! -name '*.npz' -printf '%s\n' | awk '{n++; s+=\$1} END {print n, s}'")
  l=$(cd "$DEST" && find $d -type f ! -name '*.npz' -printf '%s\n' | awk '{n++; s+=$1} END {print n, s}')
  [ "$r" = "$l" ] && echo "  OK   $d  ($l)" || echo "  MISMATCH $d remote=($r) local=($l)"
done
echo "=== DOWNLOAD_COMPLETE ==="
