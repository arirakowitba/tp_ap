#!/usr/bin/env bash
# Cola secuencial del agente A (una sola corrida por vez, 1 hilo). Línea: NAME|ENV VARS|algo|stacks|extras
# usage: bash run_queue_a.sh queue.txt   (espera a que no haya otro train_a.py corriendo)
set -u
Q=$1; cd "$(dirname "$0")"; mkdir -p logs
export AP2026Q2_DATA_DIR=${AP2026Q2_DATA_DIR:-/home/claude/data}
while pgrep -f "^python3 -u train_a.py" > /dev/null; do sleep 20; done
while IFS='|' read -r name envs algo stacks extras; do
  [ -z "$name" ] && continue; [[ "$name" == \#* ]] && continue
  if grep -q "OOF acc" "logs/$name.log" 2>/dev/null; then echo "skip $name"; continue; fi
  echo "start $name $(date +%H:%M)"
  env $envs python3 -u train_a.py "$name" "$algo" "$stacks" "$extras" > "logs/$name.log" 2>&1
  grep -h "OOF acc\|Traceback\|Error" "logs/$name.log" | tail -2
done < "$Q"
echo "queue finished $(date +%H:%M)"
