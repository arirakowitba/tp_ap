#!/bin/bash
# Tubería base (round126 + b_cat_noso_b de round127) sobre un holdout. Serial, 1 hilo.
# usage: bash run_base.sh DATA_DIR LOGDIR
set -eu
export AP2026Q2_DATA_DIR=$1 THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=$2; mkdir -p "$LOG"
R=/home/claude/tp_ap/src
step() { local n=$1; shift; echo "start $n $(date +%s)"; ( "$@" ) > "$LOG/$n.log" 2>&1; echo "end $n $(date +%s) $(tail -1 $LOG/$n.log)"; }
cd $R/round126_modelos
[ -f $1/feats.pkl ] || step feats python3 -u feats.py
[ -f $1/pred_sparse_all.npz ] || step sparse_all python3 -u sparse_all.py 1.0
[ -f $1/pred_nbsvm.npz ] || step nbsvm python3 -u nbsvm.py
[ -f $1/knn_title.pkl ] || step knn_title python3 -u knn_title.py
cd $R/round127_agente_b
[ -f $1/pred_b_cat_noso_b.npz ] || step b_cat_noso_b env SELLEROTHER=none SEED=4242 CAT_LR=0.06 python3 -u cat_b.py b_cat_noso_b
