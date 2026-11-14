#!/bin/bash
# Mejora candidata: vecinos de título del mismo vendedor (sknn). Screening con proxy LGB (CV interna) y CatBoost.
set -u
export AP2026Q2_DATA_DIR=${DD:-/home/claude/work/r145_ho} THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DD=$AP2026Q2_DATA_DIR; LOG=${LOG:-/home/claude/tp_ap/results/round145_claude_holdout/logs}; R=/home/claude/tp_ap/src
step() { local n=$1; shift; echo "start $n $(date +%s)"; ( "$@" ) > "$LOG/$n.log" 2>&1; echo "end $n $(date +%s) $(tail -1 $LOG/$n.log)"; }
while pgrep -f "run_chain2.sh|cat_b.py|sparse_fix.py|diag_lgb.py" > /dev/null; do sleep 10; done
cd $R/round145_claude_holdout
[ -f $DD/sknn.pkl ] || step sknn python3 -u seller_knn.py
step d_lgb_fix_sknn env STACKS=sparse_fix,nbsvm FRAME_EXTRA=sknn.pkl python3 -u diag_lgb.py d_lgb_fix_sknn
step d_lgb_fix_nostk env STACKS=sparse_fix,nbsvm DROP=stk_ python3 -u diag_lgb.py d_lgb_fix_nostk
echo "chain3 done $(date +%s)"
