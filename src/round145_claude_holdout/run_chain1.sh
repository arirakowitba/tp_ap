#!/bin/bash
# Continuación serial: nbsvm, knn_title, ablaciones LightGBM (proxy rápido), luego b_cat_noso_b.
set -u
export AP2026Q2_DATA_DIR=/home/claude/work/r145_ho THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DD=$AP2026Q2_DATA_DIR; LOG=/home/claude/tp_ap/results/round145_claude_holdout/logs; R=/home/claude/tp_ap/src
step() { local n=$1; shift; echo "start $n $(date +%s)"; ( "$@" ) > "$LOG/$n.log" 2>&1; echo "end $n $(date +%s) $(tail -1 $LOG/$n.log)"; }
while pgrep -f "sparse_all.py" > /dev/null; do sleep 10; done
echo "sparse_all done $(date +%s) $(tail -1 $LOG/sparse_all.log)"
cd $R/round126_modelos
[ -f $DD/pred_nbsvm.npz ] || step nbsvm python3 -u nbsvm.py
[ -f $DD/knn_title.pkl ] || step knn_title python3 -u knn_title.py
cd $R/round145_claude_holdout
step d_lgb_all env python3 -u diag_lgb.py d_lgb_all
step d_lgb_fixed env ES=fixed FIXED_IT=700 python3 -u diag_lgb.py d_lgb_fixed
step d_lgb_nostk env DROP=stk_ python3 -u diag_lgb.py d_lgb_nostk
step d_lgb_noknn env DROP=knn_ python3 -u diag_lgb.py d_lgb_noknn
step d_lgb_note env DROP=te_,tecnt_ python3 -u diag_lgb.py d_lgb_note
step d_lgb_nolabel env DROP=stk_,knn_,te_,tecnt_ python3 -u diag_lgb.py d_lgb_nolabel
cd $R/round127_agente_b
[ -f $DD/pred_b_cat_noso_b.npz ] || step b_cat_noso_b env SELLEROTHER=none SEED=4242 CAT_LR=0.06 python3 -u cat_b.py b_cat_noso_b
echo "chain1 done $(date +%s)"
