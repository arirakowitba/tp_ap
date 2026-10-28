"""Builds disjoint public probes on top of 48 from a model's disagreements, split by model confidence.
Each probe = 48 + flips of the IDs where the model contradicts 48 with P(48 wrong) in a band.
Scores of these probes calibrate how often 48 is wrong as a function of model confidence
(the posterior of round125 cannot resolve that: it shrinks to the common prior q≈1/3).
usage: python build_band_probes.py MODEL OUTDIR START_NUMBER "0.85,1.01" "0.7,0.85" ..."""
import os, sys, json, hashlib, numpy as np, csv

D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "..", "results", "revision_submissions", "posterior_flips_20261027")
model, outdir, start = sys.argv[1], sys.argv[2], int(sys.argv[3])
bands = [tuple(map(float, b.split(","))) for b in sys.argv[4:]]
EXCLUDE_PROBED = os.environ.get("EXCLUDE_PROBED", "1") == "1"   # rows already probed carry their evidence in the posterior
os.makedirs(outdir, exist_ok=True)
s48 = {int(k): v for k, v in json.load(open(os.path.join(PF, "s48.json"))).items()}
post = {int(r["ID"]): r for r in csv.DictReader(open(os.path.join(PF, "posterior_por_id.csv")))}
te = np.load(D + f"pred_{model}.npz")["te"]
p48 = np.array([s48[i] == "used" for i in range(1, 30001)])
pw = np.where(p48, 1 - te, te)          # model probability that 48 is wrong
manifest = {"exclude_probed": EXCLUDE_PROBED, "base": "48", "base_sha256": "3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722", "model": model, "probes": []}
for k, (lo, hi) in enumerate(bands):
    ids = [i for i in range(1, 30001) if lo <= pw[i - 1] < hi and not (EXCLUDE_PROBED and i in post)]
    sub = dict(s48)
    for i in ids: sub[i] = "new" if sub[i] == "used" else "used"
    txt = "ID,condition\n" + "".join(f"{i},{sub[i]}\n" for i in range(1, 30001))
    n = start + k
    fn = f"submission_{n}_{model}_banda_{lo:.2f}_{min(hi,1):.2f}_sobre48.csv"
    open(os.path.join(outdir, fn), "w", newline="").write(txt)
    probed = [i for i in ids if i in post]
    manifest["probes"].append({"number": n, "file": fn, "sha256": hashlib.sha256(txt.encode()).hexdigest(), "band_P48_wrong": [lo, hi],
                               "n_flips": len(ids), "to_used": int(sum(s48[i] == "new" for i in ids)), "to_new": int(sum(s48[i] == "used" for i in ids)),
                               "already_probed": len(probed), "posterior_exp_public_net_probed": round(sum(float(post[i]["ganancia_esperada_publica"]) for i in probed), 2),
                               "model_exp_net_total": round(float(sum(2 * pw[i - 1] - 1 for i in ids)), 1), "ids": ids})
    print(n, (lo, hi), "flips", len(ids), "probed", len(probed), "model-expected net (all 30k)", manifest["probes"][-1]["model_exp_net_total"])
json.dump(manifest, open(os.path.join(outdir, f"manifest_bandas_{model}.json"), "w"), indent=1)
