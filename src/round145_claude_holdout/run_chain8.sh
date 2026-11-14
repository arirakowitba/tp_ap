#!/bin/bash
# Sobre el holdout externo: sknn anidado, TE anidado (cats nativas), luego 2.ª semilla de la config corregida.
set -u
export THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=/home/claude/tp_ap/results/round145_claude_holdout/logs; R=/home/claude/tp_ap/src
step() { local n=$1; shift; echo "start $n $(date +%s) $(free -m | awk '/Mem/{print "avail_MB="$7}')"; ( "$@" ) > "$LOG/$n.log" 2>&1; echo "end $n $(date +%s) $(tail -1 $LOG/$n.log)"; }
until grep -q "end c_in_sknn" $LOG/chain6.out; do sleep 10; done
pkill -f run_chain6.sh; sleep 2; pkill -f "cat_r145.py c_fix_s2"; sleep 2; rm -f /home/claude/work/r145_ho/pred_c_fix_s2.npz
export AP2026Q2_DATA_DIR=/home/claude/work/r145_ho; DD=$AP2026Q2_DATA_DIR
cd $R/round145_claude_holdout
[ -f $DD/sknn_nested.npy ] || step sknn_nested python3 -u sknn_nested.py
for spec in "c_fix_sknn_nest|EXTRAS=knn_title.pkl,sknn.pkl SKNN_NESTED=1" "c_fix_tenest|TE_NESTED=1" "c_fix_s2|SEED=5151"; do
  n=${spec%%|*}; e=${spec#*|}
  [ -f $DD/pred_$n.npz ] || step $n env $e python3 -u cat_r145.py $n
done
echo "chain8 done $(date +%s)"
