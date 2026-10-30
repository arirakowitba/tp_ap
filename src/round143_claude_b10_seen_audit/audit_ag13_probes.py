"""Audit Antigravity AG-0013 probes (CSV paths as args) against: submission 48, B10 full70k test probs, variant_c test probs,
round-125 posterior over public probes 04-58, the five 28-10 blocks, and seller train majority. No test labels.
usage: python audit_ag13_probes.py REPO DATA_DIR B10_TEST_P VARIANT_C_NPY OUT_JSON probe1.csv [probe2.csv ...]"""
import sys, json, numpy as np, pandas as pd
repo, data, b10p, vcp, outj = sys.argv[1:6]; files = sys.argv[6:]
s48 = {int(k): v for k, v in json.load(open(f"{repo}/results/revision_submissions/posterior_flips_20261027/s48.json")).items()}
v48 = np.array([s48[i] == "used" for i in range(1, 30001)]); pb = np.load(b10p); vc = np.load(vcp)
post = pd.read_csv(f"{repo}/results/revision_submissions/posterior_flips_20261027/posterior_por_id.csv").set_index("ID")
p28 = set(i for v in json.load(open(f"{repo}/src/round140_diagnostico/bloques_28_10.json")).values() for i in v)
tr = pd.read_json(f"{data}/train_data.jsonlines", lines=True, dtype={"seller_id": str}); te = pd.read_json(f"{data}/test_data.jsonlines", lines=True, dtype={"seller_id": str})
cnt = tr.seller_id.astype(str).value_counts(); sel = te.seller_id.astype(str).values
urate = tr.assign(u=(tr.condition == "used").astype(int)).groupby(tr.seller_id.astype(str)).u.mean()
out = {}
for f in files:
    d = pd.read_csv(f).sort_values("ID"); v = (d.condition.astype(str) == "used").values
    ids = np.where(v != v48)[0] + 1; tu = v[ids - 1]; seen = np.isin(sel[ids - 1], cnt.index); maj = pd.Series(sel[ids - 1]).map(urate).values
    am = np.where(seen, np.where(tu, maj > .5, maj < .5), np.nan); pr = [i for i in ids if i in post.index]
    out[f.split("/")[-1]] = dict(flips=len(ids), to_used=int(tu.sum()), seen=int(seen.sum()), b10_agrees_48=float(((pb[ids - 1] >= .5) == v48[ids - 1]).mean()),
        vc_supports_flip=int(((vc[ids - 1] >= .5) == tu).sum()), probed_04_58=len(pr),
        posterior_expected_public_net=float(post.loc[pr, "ganancia_esperada_publica"].sum()) if pr else 0.0, probed_28_10=int(sum(i in p28 for i in ids)),
        seen_flip_agrees_seller_train_majority=float(np.nanmean(am)) if seen.any() else None, ids=[int(i) for i in ids])
json.dump(out, open(outj, "w"), indent=1); print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in out.items()}, indent=1))
