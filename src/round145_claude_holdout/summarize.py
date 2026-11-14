"""Junta métricas (metrics.json de eval_holdout.py), tiempos de los logs de las cadenas y comparaciones pareadas en el holdout.
usage: python summarize.py DATA_DIR LABELS_CSV RESULTS_DIR"""
import sys, os, re, json, glob, numpy as np, pandas as pd
D, LAB, RES = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3].rstrip("/") + "/"
M = json.load(open(RES + "metrics.json"))
L = pd.read_csv(LAB).sort_values("ID"); yh = (L.condition.values == "used").astype(int); idx = L.ID.values - 1
times = {}
for f in sorted(glob.glob(RES + "logs/chain*.out")) + [RES + "logs/run_base.out"]:
    st = {}
    for line in open(f):
        m = re.match(r"start (\S+) (\d+)", line)
        if m: st[m.group(1)] = int(m.group(2))
        m = re.match(r"end (\S+) (\d+)", line)
        if m and m.group(1) in st: times[m.group(1)] = int(m.group(2)) - st[m.group(1)]
def pair(a, b):
    pa = (np.load(D + f"pred_{a}.npz")["te"][idx] > .5) == yh; pb = (np.load(D + f"pred_{b}.npz")["te"][idx] > .5) == yh
    n01, n10 = int((~pa & pb).sum()), int((pa & ~pb).sum())
    z = (n01 - n10) / np.sqrt(max(n01 + n10, 1))
    return {"a": a, "b": b, "b_right_a_wrong": n01, "a_right_b_wrong": n10, "diff_pp": 100 * (pb.mean() - pa.mean()), "mcnemar_z": float(z)}
pairs = [p for p in [("b_cat_noso_b", "b_fix_noso_b"), ("sparse_all", "sparse_fix"), ("d_lgb_all", "d_lgb_fix"), ("b_fix_noso_b", "c_fix_inner"),
                     ("b_fix_noso_b", "c_fix_sknn"), ("b_fix_noso_b", "c_fix_nocat"), ("b_fix_noso_b", "c_fix_tenest"),
                     ("c_fix_sknn", "c_fix_sknn_nest"), ("b_fix_noso_b", "c_fix_sknn_nest"), ("b_fix_noso_b", "avg_fix2")] if all(os.path.exists(D + f"pred_{n}.npz") for n in p)]
MI = json.load(open(RES + "metrics_inner.json")) if os.path.exists(RES + "metrics_inner.json") else {}
out = {"data_dir": D, "metrics": M, "metrics_inner_split_44k8_11k2": MI, "seconds_per_step": times, "paired_holdout": [pair(*p) for p in pairs]}
json.dump(out, open(RES + "summary.json", "w"), indent=1)
for p in out["paired_holdout"]: print(p)
print(times)
