#!/bin/bash
# CatBoost: (4) texto crudo como text_features, (5) segunda semilla para promedio.
set -u
export AP2026Q2_DATA_DIR=${DD:-/home/claude/work/r145_ho} THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DD=$AP2026Q2_DATA_DIR; LOG=${LOG:-/home/claude/tp_ap/results/round145_claude_holdout/logs}; R=/home/claude/tp_ap/src
step() { local n=$1; shift; echo "start $n $(date +%s) $(free -m | awk '/Mem/{print "avail_MB="$7}')"; ( "$@" ) > "$LOG/$n.log" 2>&1; echo "end $n $(date +%s) $(tail -1 $LOG/$n.log)"; }
cd $R/round145_claude_holdout
while pgrep -f "run_chain4.sh" > /dev/null; do sleep 10; done
for spec in "c_fix_text|TEXT=1" "c_fix_s2|SEED=5151"; do
  n=${spec%%|*}; e=${spec#*|}
  while pgrep -f "cat_b.py|cat_r145.py|diag_lgb.py|sparse_" > /dev/null; do sleep 10; done
  [ -f $DD/pred_$n.npz ] || step $n env $e python3 -u cat_r145.py $n
done
echo "chain5 done $(date +%s)"
