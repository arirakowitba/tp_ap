#!/bin/bash
# Validación interna tipo test (44,8k/11,2k dentro de las 56k): tubería corregida completa + CatBoost base vs +sknn.
# Luego, sobre el holdout externo, una 2.ª semilla de la configuración corregida.
set -u
export THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=/home/claude/tp_ap/results/round145_claude_holdout/logs/inner; mkdir -p $LOG; R=/home/claude/tp_ap/src
step() { local n=$1; shift; echo "start $n $(date +%s) $(free -m | awk '/Mem/{print "avail_MB="$7}')"; ( "$@" ) > "$LOG/$n.log" 2>&1; echo "end $n $(date +%s) $(tail -1 $LOG/$n.log)"; }
while pgrep -f "run_chain4.sh|cat_r145.py|cat_b.py" > /dev/null; do sleep 10; done
export AP2026Q2_DATA_DIR=/home/claude/work/r145_inner; DD=$AP2026Q2_DATA_DIR
cd $R/round126_modelos
[ -f $DD/feats.pkl ] || step in_feats python3 -u feats.py
cd $R/round145_claude_holdout
[ -f $DD/pred_sparse_fix.npz ] || step in_sparse_fix python3 -u sparse_fix.py 1.0
cd $R/round126_modelos
[ -f $DD/pred_nbsvm.npz ] || step in_nbsvm python3 -u nbsvm.py
[ -f $DD/knn_title.pkl ] || step in_knn_title python3 -u knn_title.py
cd $R/round145_claude_holdout
[ -f $DD/sknn.pkl ] || step in_sknn python3 -u seller_knn.py
[ -f $DD/pred_c_in_base.npz ] || step c_in_base python3 -u cat_r145.py c_in_base
[ -f $DD/pred_c_in_sknn.npz ] || step c_in_sknn env EXTRAS=knn_title.pkl,sknn.pkl python3 -u cat_r145.py c_in_sknn
export AP2026Q2_DATA_DIR=/home/claude/work/r145_ho; DD=$AP2026Q2_DATA_DIR; LOG=/home/claude/tp_ap/results/round145_claude_holdout/logs
[ -f $DD/pred_c_fix_s2.npz ] || step c_fix_s2 env SEED=5151 python3 -u cat_r145.py c_fix_s2
echo "chain6 done $(date +%s)"
