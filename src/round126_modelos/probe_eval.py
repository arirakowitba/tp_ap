"""Evaluate a test prediction against the public-probe evidence.
For each model: disagreements with 48 inside / outside the 1,929 probed rows and the expected
public net vs 48 on probed rows according to the exact posterior (E[c_i] = P(pub & 48 wrong) - P(pub & 48 right)).
Outside the probed rows we use the historical rate q (48 wrong in ~1/3 of model disagreements) only as a reference."""
import numpy as np, json, csv, sys
import os
PF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "revision_submissions", "posterior_flips_20261027")
POST = os.path.join(PF, "posterior_por_id.csv")
S48 = os.path.join(PF, "s48.json")
D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
post = {int(r["ID"]): r for r in csv.DictReader(open(POST))}
s48 = {int(k): v for k, v in json.load(open(S48)).items()}
p48 = np.array([s48[i] == "used" for i in range(1, 30001)])

def evaluate(te_prob, thr=0.5):
    pred = te_prob > thr
    dis = np.where(pred != p48)[0] + 1
    inside = [i for i in dis if i in post]
    e_in = sum(float(post[i]["ganancia_esperada_publica"]) for i in inside)
    return dict(disagree=len(dis), probed=len(inside), exp_public_net_probed=round(e_in, 2), unprobed=len(dis) - len(inside))

if __name__ == "__main__":
    for nm in sys.argv[1:]:
        d = np.load(D + f"pred_{nm}.npz")
        print(nm, evaluate(d["te"]))
