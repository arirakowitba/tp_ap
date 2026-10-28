export OMP_NUM_THREADS=1 NOSELLER=1
nice -n 15 python c_mlp.py c_mlp6_nosel --seeds 3 --epochs 20 --lr 1e-3 --wd 1e-3 --drop 0.45 --mincnt 5 --emb 8 --hidden 256 > logs/c_mlp6_nosel.log 2>&1
nice -n 15 python c_sk.py c_et2_nosel et 300 2 0.3 > logs/c_et2_nosel.log 2>&1
