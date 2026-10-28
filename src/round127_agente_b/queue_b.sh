#!/bin/bash
# Cola secuencial (1 proceso a la vez). Cada línea: NAME|ENV|CMD
cd /home/claude/tp_ap/src/round127_agente_b
export AP2026Q2_DATA_DIR=/home/claude/data THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while IFS='|' read -r name envs cmd; do
  [ -z "$name" ] && continue; [[ "$name" == \#* ]] && continue
  echo "start $name $(date +%H:%M)"; env $envs $cmd > logs/$name.log 2>&1; echo "end $name $(date +%H:%M) $(tail -1 logs/$name.log)"
done < "$1"
