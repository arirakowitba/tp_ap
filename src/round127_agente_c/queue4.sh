export OMP_NUM_THREADS=1 SELLEROTHER=perfold
nice -n 15 python c_mlp.py c_mlp10_ple_pf_s3 --ple 12 --seeds 3 --epochs 20 --lr 1e-3 --wd 1e-3 --drop 0.45 --mincnt 5 --emb 8 --hidden 256 > logs/c_mlp10_ple_pf_s3.log 2>&1
