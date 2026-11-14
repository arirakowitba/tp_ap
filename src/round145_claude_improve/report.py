"""Collect metrics for all round145 experiments (inner split and holdout) into results JSON. Blend weights are fixed a priori (0.5)
or taken from the inner split (never tuned on the holdout)."""
import json, hashlib, os, numpy as np
from evalh import *
OUT = "/home/claude/tp_ap/results/round145_claude_improve"; os.makedirs(OUT, exist_ok=True)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
res = {"inner": {}, "holdout": {}, "files": {}}
def ld(p):
    res["files"][p] = sha(p); return np.load(p)
# inner (reference = my lgb_base = B10's lgb_tpe_2 recipe on inner train)
y, seen = inner_ctx(); D = H2 + "/inner/"
b = ld(D + "lgb_base.npy"); res["inner"]["lgb_base (lgb_tpe_2 recipe)"] = metrics(b, y, seen)
P = {k: ld(D + f) for k, f in [("lgb_plus", "lgb_plus.npy"), ("lgb_plus_noagg", "lgb_plus_noagg.npy"), ("lgb_ns_only", "lgb_ns_only.npy"),
                                 ("cb_plus", "cb_plus_i600_d6.npy"), ("cb_base", "cb_base_i600_d6.npy"), ("newspec_lgb", "newspec_lgb.npy")]}
for k, p in P.items(): res["inner"][k] = metrics(p, y, seen, b)
for k in ["lgb_plus", "lgb_plus_noagg", "cb_plus", "cb_base"]: res["inner"][f"0.5*base+0.5*{k}"] = metrics(.5 * b + .5 * P[k], y, seen, b)
p = b.copy(); p[~seen] = .5 * b[~seen] + .5 * P["newspec_lgb"][~seen]; res["inner"]["base; new rows 0.5*base+0.5*newspec"] = metrics(p, y, seen, b)
# holdout (reference = B10)
y, seen = holdout_ctx(); D = H2 + "/ho/"
B = ld(H + "/b10/full/test_p.npy"); res["holdout"]["B10"] = metrics(B, y, seen)
Q = {k: ld(D + f) for k, f in [("lgb_base (repro lgb_tpe_2)", "lgb_base.npy"), ("lgb_plus", "lgb_plus.npy"), ("lgb_plus_noagg", "lgb_plus_noagg.npy"),
                                 ("cb_plus", "cb_plus_i600_d6.npy"), ("cb_base", "cb_base_i600_d6.npy"), ("newspec_lgb", "newspec_lgb.npy")] if os.path.exists(D + f)}
for k, p in Q.items(): res["holdout"][k] = metrics(p, y, seen, B)
for k in ["lgb_plus", "lgb_plus_noagg", "cb_plus", "cb_base"]:
    if k in Q: res["holdout"][f"0.5*B10+0.5*{k}"] = metrics(.5 * B + .5 * Q[k], y, seen, B)
if "newspec_lgb" in Q:
    p = B.copy(); p[~seen] = .5 * B[~seen] + .5 * Q["newspec_lgb"][~seen]; res["holdout"]["B10; new rows 0.5*B10+0.5*newspec"] = metrics(p, y, seen, B)
if "cb_base" in Q and "newspec_lgb" in Q:
    e = .5 * B + .5 * Q["cb_base"]; p = e.copy(); p[~seen] = .5 * e[~seen] + .5 * Q["newspec_lgb"][~seen]
    res["holdout"]["0.5*B10+0.5*cb_base; new rows 0.5*that+0.5*newspec"] = metrics(p, y, seen, B)
for sec in ["inner", "holdout"]:
    print("==", sec)
    for k, r in res[sec].items(): print(fmt(k, r))
json.dump(res, open(OUT + "/metrics.json", "w"), indent=1)
