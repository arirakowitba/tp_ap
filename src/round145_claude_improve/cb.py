"""CatBoost (native categoricals incl. seller_id) on B10 stack features + round145 new features.
usage: python cb.py STACK_DIR MODE VARIANT [iters] [depth] [lr]
  VARIANT base (B10 lgb stack) | plus (stack + all new features)"""
import os, sys, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
D, MODE, V = sys.argv[1], sys.argv[2], sys.argv[3]
it = int(sys.argv[4]) if len(sys.argv) > 4 else 600; dep = int(sys.argv[5]) if len(sys.argv) > 5 else 6; lr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.07
t0 = time.time(); s = pd.read_pickle(f"{D}/stack_{MODE}.pkl")
a, b = s["st"], s["sv"]
if V == "plus": a = pd.concat([a, s["new_tr"]], axis=1); b = pd.concat([b, s["new_ev"]], axis=1)
m = CatBoostClassifier(iterations=it, depth=dep, learning_rate=lr, l2_leaf_reg=6, random_seed=145, loss_function="Logloss", thread_count=1,
                       allow_writing_files=False, one_hot_max_size=10, random_strength=.5, verbose=100)
m.fit(a, s["ytr"], cat_features=s["cats"]); p = m.predict_proba(b)[:, 1]
out = f"{D}/cb_{V}_i{it}_d{dep}.npy"; np.save(out, p); print("saved", out, round(time.time() - t0), "s", flush=True)
