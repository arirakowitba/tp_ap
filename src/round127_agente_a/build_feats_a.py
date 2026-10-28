"""Features nuevas del agente A. Genera tablas de 100k filas (orden de feats.pkl: 70k train + 30k test):
  a_feats_sel.pkl  agregados de vendedor no supervisados (train+test)
  a_feats_raw.pkl  campos crudos del JSON no usados por feats.py (atributos, variaciones, fotos, fechas)
  a_feats_cat.pkl  contexto de categoría: posición del precio, TE OOF de categoría × tipo de aviso / cantidad / modo
Todas las codificaciones con etiqueta son out-of-fold con los 5 folds compartidos (StratifiedKFold seed 2026).
usage: AP2026Q2_DATA_DIR=/home/claude/data python build_feats_a.py"""
import os, re, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
D = os.environ.get("AP2026Q2_DATA_DIR", "/home/claude/data").rstrip("/") + "/"
tr = pd.read_pickle(D + "tr.pkl"); te = pd.read_pickle(D + "te.pkl")
df = pd.concat([tr, te], ignore_index=True); ntr = len(tr)
P = pd.read_pickle(D + "feats.pkl"); F = P["F"]; y = P["y"]
assert (df.seller_id.astype(str).values == F.seller_id.astype(str).values).all()
folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y))
lst = lambda x: x if isinstance(x, list) else []
is_test = np.r_[np.zeros(ntr), np.ones(len(te))]

# ------------------------------------------------------------------ raw
R = pd.DataFrame(index=df.index)
attrs = df.attributes.map(lst)
def aval(l, suf):
    for a in l:
        if str(a.get("id", "")).endswith(suf): return a.get("value_name")
    return None
R["attr_km"] = pd.to_numeric(attrs.map(lambda l: aval(l, "-KMTS")), errors="coerce")
R["attr_year"] = pd.to_numeric(attrs.map(lambda l: aval(l, "-YEAR")), errors="coerce")
R["attr_has_part_number"] = attrs.map(lambda l: int(any("PART_NUMBER" in str(a.get("id")) for a in l)))
R["attr_n_no"] = attrs.map(lambda l: sum(a.get("value_name") == "No" for a in l))
R["attr_n_empty"] = attrs.map(lambda l: sum((a.get("value_name") or "") == "" for a in l))
R["attr_season"] = attrs.map(lambda l: aval(l, "Season") or "na")
R["attr_gender"] = attrs.map(lambda l: aval(l, "GENDER") or "na")
R["attr_group"] = attrs.map(lambda l: "|".join(sorted({str(a.get("attribute_group_id")) for a in l})) or "na")
var = df.variations.map(lst)
R["var_avail"] = var.map(lambda l: sum(v.get("available_quantity", 0) or 0 for v in l))
R["var_n_pics"] = var.map(lambda l: len({p for v in l for p in lst(v.get("picture_ids"))}))
R["var_n_price"] = var.map(lambda l: len({v.get("price") for v in l}))
R["var_n_attr"] = var.map(lambda l: max([len(lst(v.get("attribute_combinations"))) for v in l], default=0))
R["var_has_custom"] = var.map(lambda l: int(any(v.get("seller_custom_field") for v in l)))
R["var_avail_minus_avail"] = R.var_avail - pd.to_numeric(df.available_quantity, errors="coerce")
pics = df.pictures.map(lst)
def wh(s):
    try: w, h = str(s).split("x"); return int(w), int(h)
    except Exception: return 0, 0
R["pic_n_sizes"] = pics.map(lambda l: len({p.get("size") for p in l}))
R["pic_frac_500sq"] = pics.map(lambda l: np.mean([p.get("size") == "500x500" for p in l]) if l else np.nan)
R["pic_frac_landscape"] = pics.map(lambda l: np.mean([wh(p.get("size"))[0] > wh(p.get("size"))[1] for p in l]) if l else np.nan)
R["pic_frac_portrait"] = pics.map(lambda l: np.mean([wh(p.get("size"))[0] < wh(p.get("size"))[1] for p in l]) if l else np.nan)
R["pic_min_area"] = pics.map(lambda l: min([wh(p.get("size"))[0] * wh(p.get("size"))[1] for p in l], default=-1))
R["pic_size_ne_max"] = pics.map(lambda l: np.mean([p.get("size") != p.get("max_size") for p in l]) if l else np.nan)
R["pic0_aspect"] = pics.map(lambda l: (wh(l[0].get("size"))[0] / max(wh(l[0].get("size"))[1], 1)) if l else np.nan)
def pmonth(pid):
    m = re.search(r"_(\d{2})(\d{4})$", pid or "")
    if m and 1 <= int(m.group(1)) <= 12: return int(m.group(2)) * 12 + int(m.group(1)) - 1
    return np.nan
pm = pics.map(lambda l: [pmonth(p.get("id")) for p in l])
cr = pd.to_datetime(df.date_created, utc=True)
crm = (cr.dt.year * 12 + cr.dt.month - 1).values
R["pic_month_lag_min"] = crm - pm.map(lambda l: np.nanmin(l) if l and not all(np.isnan(l)) else np.nan).values
R["pic_month_lag_max"] = crm - pm.map(lambda l: np.nanmax(l) if l and not all(np.isnan(l)) else np.nan).values
R["pic_n_months"] = pm.map(lambda l: len({v for v in l if not np.isnan(v)}))
R["pic_oldfmt"] = pics.map(lambda l: np.mean([bool(re.search(r"_\d{3,4}$", p.get("id", ""))) for p in l]) if l else np.nan)
R["pic_prefix_len"] = pics.map(lambda l: len((l[0].get("id") or "").split("-")[0]) if l else -1)
lu = pd.to_datetime(df.last_updated, utc=True)
R["upd_hour"] = lu.dt.hour; R["created_month"] = cr.dt.month; R["created_year_month"] = crm
R["upd_same_sec_created"] = ((lu - cr).dt.total_seconds().abs() < 5).astype(int)
R["upd_minus_created_log"] = np.log1p((lu - cr).dt.total_seconds().clip(lower=0))
R["created_minute"] = cr.dt.minute
import unicodedata
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower(); return re.sub(r"[^a-z0-9]+", "", s)
slug = df.permalink.map(lambda u: re.sub(r"^.*?/MLA\d+-", "", u or "").replace("-_JM", "")).map(norm)
R["title_edited"] = (slug != df.title.map(norm)).astype(int)
R["price_minus_base"] = pd.to_numeric(df.price, errors="coerce") - pd.to_numeric(df.base_price, errors="coerce")
R["orig_price_ratio"] = pd.to_numeric(df.original_price, errors="coerce") / pd.to_numeric(df.price, errors="coerce")
R["n_desc"] = df.descriptions.map(lambda l: len(lst(l)))
t = df.title.fillna("").astype(str)
R["title_has_x_dims"] = t.str.contains(r"\d+\s*x\s*\d+", case=False).astype(int)
R["title_n_upper_words"] = t.map(lambda s: sum(w.isupper() and len(w) > 1 for w in s.split()))
R["title_has_model_code"] = t.str.contains(r"\b[A-Za-z]{1,4}-?\d{2,}[A-Za-z0-9-]*\b").astype(int)
R["title_has_year"] = t.str.contains(r"\b(19[0-9]{2}|20[01][0-9])\b").astype(int)
R["title_has_qty"] = t.str.contains(r"\bx\s?\d+\b|\bpack\b|\bkit\b|\bset\b|unidades|\bpar\b", case=False).astype(int)
R["title_punct"] = t.str.count(r"[!*+/,.-]")
for c in ["attr_season", "attr_gender", "attr_group"]: R[c] = R[c].astype("category")
R.to_pickle(D + "a_feats_raw.pkl"); print("raw", R.shape, flush=True)

# ------------------------------------------------------------------ seller (no supervisado, train+test)
S = pd.DataFrame(index=df.index)
sid = F.seller_id.astype(str).values
G = pd.DataFrame({"s": sid, "test": is_test, "lp": F.log_price.values, "iq": np.log1p(F.initial_quantity.values),
                  "iq1": (F.initial_quantity.values == 1).astype(float), "sold0": (F.sold_quantity.values == 0).astype(float),
                  "cat": F.category_id.astype(str).values, "lt": F.listing_type_id.astype(str).values, "bm": F.buying_mode.astype(str).values,
                  "cr": F.created_days.values, "tl": F.title_len.values, "wl": F.warranty_len.values, "nvar": (F.n_variations.values > 0).astype(float),
                  "paused": (F.status.astype(str).values == "paused").astype(float), "nattr": (F.n_attributes.values > 0).astype(float),
                  "w": df.warranty.fillna("").astype(str).str.lower().str.strip().values, "tedit": R.title_edited.values,
                  "pic500": R.pic_frac_500sq.values, "piclag": R.pic_month_lag_min.values, "upsame": R.upd_same_sec_created.values,
                  "usd": F.currency_usd.values, "ship_me2": (F.ship_mode.astype(str).values == "me2").astype(float)})
gg = G.groupby("s")
S["sel_n_test"] = gg.test.transform("sum"); S["sel_n_train"] = gg.test.transform("count") - S.sel_n_test
S["sel_std_lp"] = gg.lp.transform("std"); S["sel_min_lp"] = gg.lp.transform("min"); S["sel_max_lp"] = gg.lp.transform("max")
S["lp_minus_sel_mean"] = G.lp - gg.lp.transform("mean")
for c in ["iq", "iq1", "sold0", "tl", "wl", "nvar", "paused", "nattr", "tedit", "pic500", "piclag", "upsame", "usd", "ship_me2"]:
    S["sel_mean_" + c] = gg[c].transform("mean")
S["sel_std_iq"] = gg.iq.transform("std")
for c in ["lt", "bm", "w"]:
    S["sel_nuniq_" + c] = gg[c].transform("nunique")
S["sel_frac_bm_auction"] = (G.bm == "auction").groupby(G.s).transform("mean")
S["sel_frac_bm_classified"] = (G.bm == "classified").groupby(G.s).transform("mean")
S["sel_frac_same_warranty"] = G.groupby(["s", "w"]).w.transform("count") / gg.w.transform("count")
S["sel_frac_same_cat"] = G.groupby(["s", "cat"]).cat.transform("count") / gg.cat.transform("count")
S["sel_cr_span"] = gg.cr.transform("max") - gg.cr.transform("min")
S["sel_n_same_day"] = G.assign(d=np.floor(G.cr)).groupby(["s", "d"]).d.transform("count")
S["cr_rank_in_seller"] = gg.cr.rank(pct=True)
# títulos casi duplicados dentro del vendedor (mismo título normalizado)
tn = df.title.map(norm).values
S["sel_dup_title"] = pd.Series(tn).groupby([sid, tn]).transform("count").values
S["global_dup_title"] = pd.Series(tn).map(pd.Series(tn).value_counts()).values
S["global_dup_title_nsellers"] = pd.DataFrame({"t": tn, "s": sid}).groupby("t").s.transform("nunique").values
S.to_pickle(D + "a_feats_sel.pkl"); print("sel", S.shape, flush=True)

# ------------------------------------------------------------------ categoría
C = pd.DataFrame(index=df.index)
cat = F.category_id.astype(str).values
lp = F.log_price.values
C["cat_price_pct"] = pd.Series(lp).groupby(cat).rank(pct=True).values
C["cat_lp_std"] = pd.Series(lp).groupby(cat).transform("std").values
C["cat_lp_q10"] = pd.Series(lp).groupby(cat).transform(lambda s: s.quantile(.1)).values
C["cat_lp_q90"] = pd.Series(lp).groupby(cat).transform(lambda s: s.quantile(.9)).values
C["cat_frac_iq1"] = pd.Series((F.initial_quantity.values == 1).astype(float)).groupby(cat).transform("mean").values
C["cat_nsellers"] = pd.DataFrame({"c": cat, "s": sid}).groupby("c").s.transform("nunique").values
C["cat_frac_free"] = pd.Series((F.listing_type_id.astype(str).values == "free").astype(float)).groupby(cat).transform("mean").values
prior = y.mean()
def te_oof(keys, alpha=10):
    keys = np.asarray(keys).astype(str); out = np.zeros(len(keys)); ktr = keys[:ntr]
    for a, b in folds:
        s = pd.DataFrame({"k": ktr[a], "y": y[a]}).groupby("k").y.agg(["sum", "count"]).reindex(ktr[b])
        out[b] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values
    s = pd.DataFrame({"k": ktr, "y": y}).groupby("k").y.agg(["sum", "count"]).reindex(keys[ntr:])
    out[ntr:] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values
    return out
iq1 = (F.initial_quantity.values == 1).astype(int).astype(str)
lt = F.listing_type_id.astype(str).values
bm = F.buying_mode.astype(str).values
pq = pd.Series(lp).groupby(cat).transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) if len(s) >= 8 else s * 0).values.astype(int).astype(str)
C["te_cat_lt"] = te_oof(np.char.add(np.char.add(cat, "|"), lt))
C["te_cat_iq1"] = te_oof(np.char.add(np.char.add(cat, "|"), iq1))
C["te_cat_bm"] = te_oof(np.char.add(np.char.add(cat, "|"), bm))
C["te_cat_pq"] = te_oof(np.char.add(np.char.add(cat, "|"), pq))
C["te_lt_iq1_sold0"] = te_oof(np.char.add(np.char.add(lt, iq1), (F.sold_quantity.values == 0).astype(int).astype(str)))
st = F.state.astype(str).values
C["te_cat_state"] = te_oof(np.char.add(np.char.add(cat, "|"), st))
# vendedores con un solo aviso: tasa OOF de 'used' en esa categoría entre vendedores con un solo aviso
one = (pd.Series(sid).map(pd.Series(sid).value_counts()).values == 1).astype(int).astype(str)
C["te_cat_single"] = te_oof(np.char.add(np.char.add(cat, "|"), one))
C.to_pickle(D + "a_feats_cat.pkl"); print("cat", C.shape, flush=True)
