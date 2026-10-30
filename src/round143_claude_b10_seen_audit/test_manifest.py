"""Test-side manifest (no labels): rows where B10 full-70k refit disagrees with submission 48, and with Antigravity's
Blend075_Bag3 inductive 70k, split by seen/new seller, train seller frequency, direction and historical public probing.
usage: python test_manifest.py REPO DATA_DIR B10_FULL_DIR AG_TEST_BLEND_NPY OUT_CSV"""
import sys, json, hashlib, numpy as np, pandas as pd
repo, data, b10dir, blend_npy, out = sys.argv[1:6]
tr = pd.read_json(f"{data}/train_data.jsonlines", lines=True, dtype={"seller_id": str}); te = pd.read_json(f"{data}/test_data.jsonlines", lines=True, dtype={"seller_id": str})
cnt = tr.seller_id.astype(str).value_counts(); used = tr.assign(u=(tr.condition == "used").astype(int)).groupby(tr.seller_id.astype(str)).u.mean()
s48 = {int(k): v for k, v in json.load(open(f"{repo}/results/revision_submissions/posterior_flips_20261027/s48.json")).items()}
v48 = np.array([s48[i] == "used" for i in range(1, 30001)])
pb = np.load(f"{b10dir}/test_p.npy"); pblend = np.load(blend_npy)
post = pd.read_csv(f"{repo}/results/revision_submissions/posterior_flips_20261027/posterior_por_id.csv").set_index("ID")
blocks = json.load(open(f"{repo}/src/round140_diagnostico/bloques_28_10.json"))
probed28 = set(i for k, v in blocks.items() for i in v)
sel = te.seller_id.astype(str).values
df = pd.DataFrame({"ID": np.arange(1, 30001), "seller_id": sel, "seen": np.isin(sel, cnt.index), "seller_train_n": pd.Series(sel).map(cnt).fillna(0).astype(int).values,
                   "seller_train_used_rate": pd.Series(sel).map(used).values, "pred_48": np.where(v48, "used", "new"),
                   "p_b10": pb.round(5), "p_blend075_bag3": pblend.round(5)})
df["b10_vs_48"] = (pb >= .5) != v48; df["blend_vs_48"] = (pblend >= .5) != v48
df["probed_04_58"] = df.ID.isin(post.index); df["probed_28_10"] = df.ID.isin(probed28)
df["posterior_P48_wrong"] = df.ID.map(post["P_48_mal"]) if "P_48_mal" in post.columns else np.nan
df["freq_band"] = pd.cut(df.seller_train_n, [-1, 0, 1, 4, 20, 10**9], labels=["0(new)", "1", "2-4", "5-20", ">20"])
m = df[df.b10_vs_48 | df.blend_vs_48].copy()
m["direction_b10"] = np.where(m.b10_vs_48, np.where(m.p_b10 >= .5, "->used", "->new"), "")
m["direction_blend"] = np.where(m.blend_vs_48, np.where(m.p_blend075_bag3 >= .5, "->used", "->new"), "")
m.to_csv(out, index=False)
summ = {"n_b10_vs_48": int(df.b10_vs_48.sum()), "n_blend_vs_48": int(df.blend_vs_48.sum()),
        "b10_vs_48_by_seen": df[df.b10_vs_48].seen.value_counts().to_dict(), "blend_vs_48_by_seen": df[df.blend_vs_48].seen.value_counts().to_dict(),
        "b10_vs_48_by_freq": df[df.b10_vs_48].freq_band.value_counts().sort_index().astype(int).to_dict(),
        "blend_vs_48_by_freq": df[df.blend_vs_48].freq_band.value_counts().sort_index().astype(int).to_dict(),
        "blend_vs_48_seen_where_b10_agrees_with_48": int((df.blend_vs_48 & df.seen & ~df.b10_vs_48).sum()),
        "b10_vs_48_probed_04_58": int((df.b10_vs_48 & df.probed_04_58).sum()), "blend_vs_48_probed_04_58": int((df.blend_vs_48 & df.probed_04_58).sum()),
        "blend_vs_48_probed_28_10": int((df.blend_vs_48 & df.probed_28_10).sum()),
        "csv_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
print(json.dumps(summ, indent=1, default=str))
