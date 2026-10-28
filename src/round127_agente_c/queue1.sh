export OMP_NUM_THREADS=1
nice -n 15 python c_sk.py c_knn50 knn 50 > logs/c_knn50.log 2>&1
nice -n 15 python c_mlp.py c_mlp4_nostk --nostk --epochs 20 --lr 1e-3 --wd 1e-3 --drop 0.45 --mincnt 5 --emb 8 --hidden 256 > logs/c_mlp4_nostk.log 2>&1
nice -n 15 python c_mlp.py c_mlp5_s3 --seeds 3 --epochs 20 --lr 1e-3 --wd 1e-3 --drop 0.45 --mincnt 5 --emb 8 --hidden 256 > logs/c_mlp5_s3.log 2>&1
