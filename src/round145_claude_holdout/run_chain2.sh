#!/bin/bash
# Tubería corregida: sparse_fix (binning simétrico) -> proxy LGB -> CatBoost b_cat_noso_b idéntico salvo el stack corregido.
set -u
export AP2026Q2_DATA_DIR=${DD:-/home/claude/work/r145_ho} THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DD=$AP2026Q2_DATA_DIR; LOG=${LOG:-/home/claude/tp_ap/results/round145_claude_holdout/logs}; R=/home/claude/tp_ap/src; mkdir -p $LOG
step() { local n=$1; shift; echo "start $n $(date +%s)"; ( "$@" ) > "$LOG/$n.log" 2>&1; echo "end $n $(date +%s) $(tail -1 $LOG/$n.log)"; }
while pgrep -f "cat_b.py|sparse_all.py|diag_lgb.py" > /dev/null; do sleep 10; done
cd $R/round145_claude_holdout
[ -f $DD/pred_sparse_fix.npz ] || step sparse_fix python3 -u sparse_fix.py 1.0
step d_lgb_fix env STACKS=sparse_fix,nbsvm python3 -u diag_lgb.py d_lgb_fix
cd $R/round127_agente_b
[ -f $DD/pred_b_fix_noso_b.npz ] || step b_fix_noso_b env STACKS=sparse_fix,nbsvm SELLEROTHER=none SEED=4242 CAT_LR=0.06 python3 -u cat_b.py b_fix_noso_b
echo "chain2 done $(date +%s)"
