#!/usr/bin/env bash
# Resumable upload of one local file to a pod (re-appends from the remote size after a drop).
# Run in the USER's terminal (agent background tasks get killed):
#   bash upload_file.sh <local_file> <remote_path> <ip> <port>
# Touches <remote_path>.done on the pod when complete.
L="$1"; R="$2"; H="root@$3"; P="$4"; K="$HOME/.ssh/id_ed25519"
SSH=(ssh -p "$P" -i "$K" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
s=$(stat -c %s "$L"); echo "local = $s bytes"
while :; do
  r=$("${SSH[@]}" "$H" "stat -c %s $R 2>/dev/null || echo 0" 2>/dev/null); r=${r:-0}
  echo "$(date +%H:%M:%S)  remote=$r / $s  ($(( r*100/s ))%)"
  if [ "$r" -ge "$s" ]; then "${SSH[@]}" "$H" "touch $R.done" && echo "=== UPLOAD_COMPLETE ==="; break; fi
  tail -c +$((r+1)) "$L" | "${SSH[@]}" "$H" "cat >> $R" || echo "(interrupted; resuming)"
  sleep 3
done
