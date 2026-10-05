#!/usr/bin/env bash
# Resumable upload of Dataset701 tar -> pod /root. Re-appends from the remote size each
# iteration, so a dropped connection just resumes. Runs in the user's terminal (not an
# agent background task), so it survives to completion. Touches /root/upload_done when done.
L="/d/Documents/Projects/Multimodal_Cancer_Detection/data/processed/ct/nnunet_raw/Dataset701_tight.tar"
R="/root/Dataset701_tight.tar"
P=20440; H="root@213.173.99.24"; K="$HOME/.ssh/id_ed25519"
SSH=(ssh -p "$P" -i "$K" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
s=$(stat -c %s "$L"); echo "local tar = $s bytes"
tries=0
while :; do
  r=$("${SSH[@]}" "$H" "stat -c %s $R 2>/dev/null || echo 0" 2>/dev/null)
  r=${r:-0}
  echo "$(date +%H:%M:%S)  remote=$r / $s  ($(( r*100/s ))%)"
  if [ "$r" -ge "$s" ]; then
    "${SSH[@]}" "$H" "touch /root/upload_done" && echo "=== UPLOAD_COMPLETE ==="
    break
  fi
  tail -c +$((r+1)) "$L" | "${SSH[@]}" "$H" "cat >> $R" || echo "(append interrupted; resuming)"
  tries=$((tries+1))
  sleep 3
done
