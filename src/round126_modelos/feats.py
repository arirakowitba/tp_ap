"""Feature engineering for AP 2026 Q2 (new vs used). Builds one frame for train+test."""
import pandas as pd, numpy as np, re, ast

import os
D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
if os.path.exists(D + "tr.pkl"):
    tr = pd.read_pickle(D + "tr.pkl"); te = pd.read_pickle(D + "te.pkl")
else:
    tr = pd.read_json(D + "train_data.jsonlines", lines=True); te = pd.read_json(D + "test_data.jsonlines", lines=True)
    tr.to_pickle(D + "tr.pkl"); te.to_pickle(D + "te.pkl")
tr["is_test"] = 0; te["is_test"] = 1; te["condition"] = np.nan
df = pd.concat([tr, te], ignore_index=True)
y = (tr.condition == "used").astype(int).values

def lst(x):
    return x if isinstance(x, list) else []

F = pd.DataFrame(index=df.index)
for c in ["price", "base_price", "initial_quantity", "available_quantity", "sold_quantity"]:
    F[c] = pd.to_numeric(df[c], errors="coerce")
F["log_price"] = np.log1p(F.price)
F["price_eq_base"] = (F.price == F.base_price).astype(int)
F["sold_ratio"] = F.sold_quantity / (F.initial_quantity + 1)
F["avail_ratio"] = F.available_quantity / (F.initial_quantity + 1)
F["init_minus_avail"] = F.initial_quantity - F.available_quantity
F["price_cents"] = ((F.price * 100) % 100).round()
F["price_round100"] = (F.price % 100 == 0).astype(int)
F["price_round10"] = (F.price % 10 == 0).astype(int)
F["price_end9"] = ((F.price.round() % 10) == 9).astype(int)
F["has_original_price"] = df.original_price.notna().astype(int)
F["official_store"] = df.official_store_id.notna().astype(int)
F["has_video"] = df.video_id.notna().astype(int)
F["has_catalog"] = df.catalog_product_id.notna().astype(int)
F["has_parent"] = df.parent_item_id.notna().astype(int)
F["automatic_relist"] = df.automatic_relist.astype(int)
F["accepts_mp"] = df.accepts_mercadopago.astype(int)
F["n_deals"] = df.deal_ids.map(lambda x: len(lst(x)))
F["currency_usd"] = (df.currency_id == "USD").astype(int)

for c in ["listing_type_id", "buying_mode", "status", "category_id", "seller_id"]:
    F[c] = df[c].astype(str)
F["sub_status"] = df.sub_status.map(lambda x: "|".join(lst(x)))
for t in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist", "poor_quality_thumbnail"]:
    F["tag_" + t] = df.tags.map(lambda x: int(t in lst(x)))
F["n_tags"] = df.tags.map(lambda x: len(lst(x)))

# payment methods
pm = df.non_mercado_pago_payment_methods.map(lambda x: [m.get("id") for m in lst(x)])
for code in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLACD", "MLABC", "MLAVE", "MLAMC", "MLAAM", "MLADC", "MLAVS", "MLAMS", "MLAWT", "MLAMP", "MLAAB"]:
    F["pm_" + code] = pm.map(lambda l: int(code in l))
F["n_pm"] = pm.map(len)

# shipping
sh = df.shipping.map(lambda x: x if isinstance(x, dict) else {})
F["ship_local_pick_up"] = sh.map(lambda s: int(bool(s.get("local_pick_up"))))
F["ship_free"] = sh.map(lambda s: int(bool(s.get("free_shipping"))))
F["ship_mode"] = sh.map(lambda s: str(s.get("mode")))
F["ship_n_methods"] = sh.map(lambda s: len(lst(s.get("methods"))))
F["ship_n_tags"] = sh.map(lambda s: len(lst(s.get("tags"))))
F["ship_tags"] = sh.map(lambda s: "|".join(sorted(map(str, lst(s.get("tags"))))))
F["ship_dims"] = sh.map(lambda s: int(s.get("dimensions") is not None))

# address
ad = df.seller_address.map(lambda x: x if isinstance(x, dict) else {})
F["state"] = ad.map(lambda a: str((a.get("state") or {}).get("id")))
F["city"] = ad.map(lambda a: str((a.get("city") or {}).get("name")).lower().strip())

# pictures
pics = df.pictures.map(lst)
F["n_pics"] = pics.map(len)
def dims(p):
    try:
        w, h = p.get("max_size", p.get("size", "0x0")).split("x"); return int(w), int(h)
    except Exception:
        return 0, 0
F["pic0_w"] = pics.map(lambda l: dims(l[0])[0] if l else -1)
F["pic0_h"] = pics.map(lambda l: dims(l[0])[1] if l else -1)
F["pic_max_area"] = pics.map(lambda l: max([dims(p)[0] * dims(p)[1] for p in l], default=-1))
F["pic_mean_area"] = pics.map(lambda l: np.mean([dims(p)[0] * dims(p)[1] for p in l]) if l else -1)
F["pic_square"] = pics.map(lambda l: int(bool(l) and dims(l[0])[0] == dims(l[0])[1]))
def picnum(p):
    m = re.search(r"/(\d+)-MLA\d+_(\d+)-", p.get("url", "") or "")
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
F["pic_prefix"] = pics.map(lambda l: picnum(l[0])[0] if l else -1)
F["pic_code"] = pics.map(lambda l: picnum(l[0])[1] if l else -1)
F["pic_code_min"] = pics.map(lambda l: min([picnum(p)[1] for p in l], default=-1))
F["pic_code_max"] = pics.map(lambda l: max([picnum(p)[1] for p in l], default=-1))
# thumbnail code
F["thumb_code"] = df.thumbnail.map(lambda u: int(m.group(1)) if isinstance(u, str) and (m := re.search(r"_(\d+)-", u)) else -1)

# variations / attributes
F["n_variations"] = df.variations.map(lambda x: len(lst(x)))
F["n_attributes"] = df.attributes.map(lambda x: len(lst(x)))
F["var_sold"] = df.variations.map(lambda x: sum(v.get("sold_quantity", 0) or 0 for v in lst(x)))
attr_names = df.attributes.map(lambda x: " ".join(str(a.get("id", "")) for a in lst(x)))

# dates
for c in ["date_created", "last_updated"]:
    df[c + "_dt"] = pd.to_datetime(df[c], utc=True, errors="coerce")
st = pd.to_datetime(df.start_time, unit="ms", utc=True, errors="coerce")
sp = pd.to_datetime(df.stop_time, unit="ms", utc=True, errors="coerce")
ref = pd.Timestamp("2013-01-01", tz="UTC")
F["created_days"] = (df.date_created_dt - ref).dt.total_seconds() / 86400
F["updated_days"] = (df.last_updated_dt - ref).dt.total_seconds() / 86400
F["start_days"] = (st - ref).dt.total_seconds() / 86400
F["stop_days"] = (sp - ref).dt.total_seconds() / 86400
F["dur_days"] = F.stop_days - F.start_days
F["upd_minus_created"] = F.updated_days - F.created_days
F["start_minus_created"] = (F.start_days - F.created_days) * 86400
F["created_hour"] = df.date_created_dt.dt.hour
F["created_dow"] = df.date_created_dt.dt.dayofweek
F["created_sec"] = df.date_created_dt.dt.second

# warranty & title text stats
w = df.warranty.fillna("").astype(str)
t = df.title.fillna("").astype(str)
F["warranty_null"] = df.warranty.isna().astype(int)
F["warranty_len"] = w.str.len()
for kw, pat in {"w_sin": r"\bsin\b", "w_no": r"\bno\b", "w_meses": r"mes", "w_anio": r"a[ñn]o", "w_fabrica": r"f[aá]brica", "w_oficial": r"oficial",
                "w_usado": r"usad", "w_nuevo": r"nuev", "w_dias": r"d[ií]as", "w_garantia": r"garant", "w_reputacion": r"reputaci|calificaci",
                "w_prueba": r"prueba|probad|funciona"}.items():
    F[kw] = w.str.lower().str.contains(pat).astype(int)
F["title_len"] = t.str.len()
F["title_words"] = t.str.split().map(len)
F["title_upper_frac"] = t.map(lambda s: sum(ch.isupper() for ch in s) / max(len(s), 1))
F["title_digits"] = t.str.count(r"\d")
for kw, pat in {"t_usado": r"usad", "t_nuevo": r"nuev", "t_original": r"original", "t_antiguo": r"antig|vintage|retro|colecci", "t_libro": r"libro",
                "t_lote": r"lote", "t_impecable": r"impecable|excelente estado|muy buen estado"}.items():
    F[kw] = t.str.lower().str.contains(pat).astype(int)

CAT = ["listing_type_id", "buying_mode", "status", "category_id", "seller_id", "sub_status", "ship_mode", "ship_tags", "state", "city"]
for c in CAT:
    F[c] = F[c].astype("category")

# seller / category frequency (train+test, unsupervised)
for c in ["seller_id", "category_id", "city"]:
    F["freq_" + c] = F[c].map(F[c].value_counts()).astype(float)
F["seller_n_cats"] = F.groupby("seller_id", observed=True)["category_id"].transform("nunique")
F["seller_mean_logprice"] = F.groupby("seller_id", observed=True)["log_price"].transform("mean")
F["seller_mean_init_q"] = F.groupby("seller_id", observed=True)["initial_quantity"].transform("mean")
F["cat_med_logprice"] = F.groupby("category_id", observed=True)["log_price"].transform("median")
F["rel_price_cat"] = F.log_price - F.cat_med_logprice

text = (t + " ").values
is_test = df.is_test.values
pd.to_pickle({"F": F, "y": y, "text": t.values, "warranty": w.values, "attr": attr_names.values, "is_test": is_test, "CAT": CAT}, D + "feats.pkl")
print(F.shape)
