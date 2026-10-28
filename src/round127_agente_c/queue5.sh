export OMP_NUM_THREADS=1 SELLEROTHER=perfold
nice -n 15 python c_sk.py c_et3_pf et 300 2 0.3 > logs/c_et3_pf.log 2>&1
NOSELLER=1 nice -n 15 python c_mlp.py c_mlp11_ple_nosel_pf_s3 --ple 12 --seeds 3 --epochs 20 --lr 1e-3 --wd 1e-3 --drop 0.45 --mincnt 5 --emb 8 --hidden 256 > logs/c_mlp11_ple_nosel_pf_s3.log 2>&1
nice -n 15 python c_mlp.py c_mlp12_ple_pf_h512_s3 --ple 16 --seeds 3 --epochs 20 --lr 7e-4 --wd 1e-2 --drop 0.5 --mincnt 3 --emb 12 --hidden 512 > logs/c_mlp12_ple_pf_h512_s3.log 2>&1
