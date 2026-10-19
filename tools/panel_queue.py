"""Mitigation panel: run a queue of tools/panel_train.py jobs one after another (one GPU job at a time).

The queue file has one job per line (the panel_train.py arguments; '#' starts a comment). Finished jobs are
skipped by panel_train.py itself (their summary JSON exists) and unfinished ones resume from their last epoch,
so after a power cut just start the queue again.

    python tools/panel_queue.py configs/panel/<queue>.txt
"""
import os
import shlex
import subprocess
import sys

import psutil


def main():
    qfile = sys.argv[1]
    lock = qfile + ".running.pid"
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read() or 0)):
        sys.exit(f"queue already running (pid {open(lock).read()})")
    open(lock, "w").write(str(os.getpid()))
    try:
        jobs = [l.split("#")[0].strip() for l in open(qfile)]
        jobs = [j for j in jobs if j]
        for i, j in enumerate(jobs, 1):
            print(f"=== [{i}/{len(jobs)}] {j}", flush=True)
            rc = subprocess.call([sys.executable, "tools/panel_train.py", *shlex.split(j)])
            if rc:
                print(f"=== FAILED rc={rc}: {j}", flush=True)
        print("=== QUEUE DONE", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
