"""Descriptive audit on the 3 reused confirm sets: in SEEN-seller rows where B10 (fit-only) and Antigravity's
Blend075_Bag3 disagree, who is right, by seller frequency in fit and direction. No selection is made with this.
usage: python confirm_seen_audit.py B10_DIR AG_DIR DATA_DIR"""
import sys, numpy as np, pandas as pd
b10, ag, data = sys.argv[1:4]
tr = pd.read_json(f"{data}/train_data.jsonlines", lines=True, dtype={"seller_id": str}); sel = tr.seller_id.astype(str).values
rows = []
for s in [2026, 314159, 20261030]:
    b = pd.read_csv(f"{b10}/seed{s}_fit/paired_b10_vs_A_hist_seed{s}.csv.gz"); a = pd.read_csv(f"{ag}/paired_all_{s}.csv.gz")
    d = b.merge(a[["row_idx", "p_Blend075_Bag3_hist"]], on="row_idx")
    tu = pd.read_csv(f"{ag}/preds_A_hist_seed{s}_tune.csv.gz"); fit = np.setdiff1d(np.arange(len(tr)), np.r_[tu.row_idx, d.row_idx])
    d["n_fit"] = pd.Series(sel[d.row_idx]).map(pd.Series(sel[fit]).value_counts()).fillna(0).astype(int).values; d["seed"] = s; rows.append(d)
D = pd.concat(rows); y = D.y_used.values; pb = D.p_b10.values >= .5; pl = D.p_Blend075_Bag3_hist.values >= .5
dis = (pb != pl) & (D.is_seen.values == 1)
X = D[dis].copy(); X["b10_right"] = (pb == y)[dis]
X["band"] = pd.cut(X.n_fit, [0, 1, 4, 20, 10**9], labels=["1", "2-4", "5-20", ">20"]); X["dir"] = np.where(X.p_b10 >= .5, "b10_used", "b10_new")
print("pooled (overlapping cuts) n", len(X), "B10 right", round(X.b10_right.mean(), 4))
print(X.groupby("seed").b10_right.agg(["count", "mean"]).round(4)); print(X.groupby(["band", "dir"], observed=True).b10_right.agg(["count", "mean"]).round(3))
