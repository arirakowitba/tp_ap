"""Reconstruye exactamente la tabla de train_gbm.py (agregados de vendedor, TE OOF, stacks y extras)
para usarla con otros learners (LightGBM afinado, XGBoost, MLP). Cachea en D/b_frame_<tag>.pkl."""
import os, numpy as np, pandas as pd
from common import D, folds

def build(stacks=("sparse_all", "nbsvm"), extras=("knn_title.pkl",), tag=None):
    tag = tag or "_".join(list(stacks) + [e.replace(".pkl", "") for e in extras])
    path = D + f"b_frame_{tag}.pkl"
    if os.path.exists(path): return pd.read_pickle(path)
    P = pd.read_pickle(D + "feats.pkl")
    F, y, is_test, CAT = P["F"].copy(), P["y"], P["is_test"], list(P["CAT"])
    ntr = int((is_test == 0).sum()); fo = folds(y)
    g = F.groupby("seller_id", observed=True)
    for c in ["warranty_null", "t_usado", "t_nuevo", "w_usado", "w_nuevo", "ship_free", "automatic_relist", "n_pics", "pic_mean_area",
              "sold_quantity", "dur_days", "accepts_mp", "n_pm", "title_upper_frac", "has_video", "price_round100"]:
        F["sell_mean_" + c] = g[c].transform("mean")
    for lt in ["free", "bronze", "silver", "gold_special", "gold", "gold_premium", "gold_pro"]:
        F["sell_frac_lt_" + lt] = (F.listing_type_id.astype(str) == lt).groupby(F.seller_id, observed=True).transform("mean")
    def te_oof(keys, alpha=5):
        keys = np.asarray(keys).astype(str); prior = y.mean(); out = np.zeros(len(F)); cnt = np.zeros(len(F)); ktr = keys[:ntr]
        for a, b in fo:
            s = pd.DataFrame({"k": ktr[a], "y": y[a]}).groupby("k").y.agg(["sum", "count"]).reindex(ktr[b])
            out[b] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values; cnt[b] = s["count"].fillna(0).values
        s = pd.DataFrame({"k": ktr, "y": y}).groupby("k").y.agg(["sum", "count"]).reindex(keys[ntr:])
        out[ntr:] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values; cnt[ntr:] = s["count"].fillna(0).values * 0.8
        return out, cnt
    F["te_seller"], F["tecnt_seller"] = te_oof(F.seller_id)
    F["te_cat"], F["tecnt_cat"] = te_oof(F.category_id)
    F["te_city"], _ = te_oof(F.city)
    F["te_seller_cat"], F["tecnt_seller_cat"] = te_oof(F.seller_id.astype(str) + "|" + F.category_id.astype(str))
    F["te_seller_lt"], _ = te_oof(F.seller_id.astype(str) + "|" + F.listing_type_id.astype(str))
    for s in stacks:
        d = np.load(D + f"pred_{s}.npz"); v = np.r_[d["oof"], d["te"]]
        F["stk_" + s] = v
        df = pd.DataFrame({"s": F.seller_id.astype(str).values, "v": v})
        sm = df.groupby("s").v.transform("sum"); sc = df.groupby("s").v.transform("count")
        F["stk_sellerother_" + s] = ((sm - v) / (sc - 1)).where(sc > 1, np.nan).values
    for e in extras:
        E = pd.read_pickle(D + e)
        for c in E.columns: F[c] = E[c].values
    F = F.copy(); out = {"F": F, "y": y, "ntr": ntr, "CAT": CAT}
    pd.to_pickle(out, path); return out
