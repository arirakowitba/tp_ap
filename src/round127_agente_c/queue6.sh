export OMP_NUM_THREADS=1 SELLEROTHER=perfold
nice -n 15 python c_mlp.py c_mlp13_pseudo_s3 --pseudo 0.97 --ple 16 --seeds 3 --epochs 20 --lr 7e-4 --wd 1e-2 --drop 0.5 --mincnt 3 --emb 12 --hidden 512 > logs/c_mlp13_pseudo_s3.log 2>&1
nice -n 15 python c_mlp.py c_res1_s3 --arch resnet --ple 16 --seeds 3 --epochs 20 --lr 7e-4 --wd 1e-2 --drop 0.4 --mincnt 3 --emb 12 --hidden 192 > logs/c_res1_s3.log 2>&1
