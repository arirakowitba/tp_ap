#!/usr/bin/env bash
# Runs experiments sequentially. Each line of the queue file: NAME|ENV VARS|algo|stacks|extras
# usage: bash run_queue.sh queue.txt LOGDIR
set -u
Q=$1; LOG=$2; mkdir -p "$LOG"
cd "$(dirname "$0")"
while IFS='|' read -r name envs algo stacks extras; do
  [ -z "$name" ] && continue
  [[ "$name" == \#* ]] && continue
  if [ -f "$AP2026Q2_DATA_DIR/pred_$name.npz" ]; then echo "skip $name"; continue; fi
  echo "start $name $(date +%H:%M)"
  env $envs python3 -u train_gbm.py "$name" "$algo" "$stacks" "$extras" > "$LOG/$name.log" 2>&1
  grep -h "OOF acc\|Traceback\|Error" "$LOG/$name.log" | tail -2
done < "$Q"
echo "queue finished $(date +%H:%M)"
