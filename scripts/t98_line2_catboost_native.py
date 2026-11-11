"""T98 Line 2 — Leak-Free Native Categorical + Text CatBoost Specialist (Seen & Unseen Regimes).

Implements CX-0164 compliant nested dual-regime validation across all 5 outer folds:
1. Loads frozen splits `data/splits/t98_dual_regime_outer_seller_group_splits.csv` and parsed caches
   `data/splits/t98_parsed_train.pkl` (70,000 rows) and `data/splits/t98_parsed_test.pkl` (30,000 rows).
2. Inside each outer fold k in 0..4:
   - Step 1: Computes leak-free `p_text_oof` (Word TF-IDF 40k + Char char_wb 60k + SAGA LR C=1.5)
     and `cat_te` (m=10.0) strictly via 4-fold StratifiedGroupKFold by seller_id (`inner_fold_{k}`)
     inside `fit_idx`, plus an enriched attribute/tag linear specialist `p_attr_oof`. Records SHA-256
     hashes of `p_text_oof` and `cat_te` for CX-0164 provenance.
   - Model 2A (`CB_clean_baseline`): Exact 9 features of Podium CatBoost (`title`, `price`, `price_ratio`,
     `num_attrs`, `warranty`, `p_text`, `cat_te`, `seller_id`, `category_id`) trained with leak-free
     seller-grouped OOF `p_text_oof` and `cat_te` instead of in-sample `ptxT`.
   - Model 2B (`CB_enriched_native`):
     * `CB_enriched_unseen_safe`: `text_features=["title", "tags_attr_text"]`,
       `cat_features=["category_id", "listing_type_id", "buying_mode", "ship_mode", "state_id",
       "brand_val", "item_cond_attr", "seller_id_masked"]` (with `seller_id_masked="UNSEEN_SELLER"`),
       all enriched numeric features, leak-free `p_text_oof`, `p_attr_oof`, `cat_te`, and fold-local
       binary token indicators.
     * `CB_enriched_with_seller`: Chronologically ordered (`start_time, row`, `has_time=True`) with
       `seller_id_masked` retaining sellers with history in `T` (singletons/unseen mapped to
       `"UNSEEN_SELLER"`) and strictly causal expanding seller target encoding (`sel_causal_te_m2`,
       `sel_causal_te_m5`, `sel_causal_cnt`).
   - Selects optimal regime-specific blend weights (`w_unseen`, `w_seen`) strictly on the 4 inner
     seller-group folds (`inner_fold_{k}`) and freezes `freeze_line2_outer{k}.json` before evaluating
     `seen_val_idx` and `unseen_val_idx`.
3. Saves all required deliverables to `results/t98_line2_catboost_native_20261111/` with
   `status = "DEVELOPMENT_ONLY"`. Does not touch `submissions/` or Kaggle.
"""
import hashlib
import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "results" / "t98_line2_catboost_native_20261111"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_THREADS = 4
OUTER_ITERS = 1200
OUTER_LR = 0.045
INNER_ITERS = 900
INNER_LR = 0.05
DEPTH = 8
L2_REG = 5.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def boot_two_way(diff: np.ndarray, sellers: np.ndarray, cats: np.ndarray, n_boot: int = 2000, seed: int = 20261111) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, si = np.unique(sellers, return_inverse=True)
    _, ci = np.unique(cats, return_inverse=True)
    n_s = int(si.max()) + 1
    n_c = int(ci.max()) + 1
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w = rng.poisson(1.0, n_s)[si] * rng.poisson(1.0, n_c)[ci]
        out[b] = (w * diff).sum() / w.sum()
    return out


def fit_predict_saga_lr(
    Cw_tr, Cc_tr, y_tr: np.ndarray, Cw_va, Cc_va, max_w: int = 40000, max_c: int = 60000, C_val: float = 1.5
) -> tuple[np.ndarray, dict]:
    df_w = np.asarray((Cw_tr > 0).sum(axis=0)).ravel()
    idx_w = np.where(df_w >= 3)[0]
    if len(idx_w) > max_w:
        tf_w = np.asarray(Cw_tr[:, idx_w].sum(axis=0)).ravel()
        idx_w = idx_w[np.argsort(-tf_w)[:max_w]]

    df_c = np.asarray((Cc_tr > 0).sum(axis=0)).ravel()
    idx_c = np.where(df_c >= 5)[0]
    if len(idx_c) > max_c:
        tf_c = np.asarray(Cc_tr[:, idx_c].sum(axis=0)).ravel()
        idx_c = idx_c[np.argsort(-tf_c)[:max_c]]

    tw = TfidfTransformer(sublinear_tf=True).fit(Cw_tr[:, idx_w])
    tc = TfidfTransformer(sublinear_tf=True).fit(Cc_tr[:, idx_c])
    X_tr = hstack([tw.transform(Cw_tr[:, idx_w]), tc.transform(Cc_tr[:, idx_c])]).tocsr()
    X_va = hstack([tw.transform(Cw_va[:, idx_w]), tc.transform(Cc_va[:, idx_c])]).tocsr()

    lr = LogisticRegression(C=C_val, max_iter=800, solver="saga", random_state=42)
    lr.fit(X_tr, y_tr)
    p_va = lr.predict_proba(X_va)[:, 1]
    info = {
        "vocab_word": int(len(idx_w)),
        "vocab_char": int(len(idx_c)),
        "n_iter": int(lr.n_iter_[0]),
    }
    return p_va, info


def compute_cat_te_map(cats_tr: np.ndarray, y_tr: np.ndarray, m: float, prior: float) -> pd.Series:
    s = pd.DataFrame({"k": cats_tr, "y": y_tr}).groupby("k")["y"].agg(["count", "sum"])
    return (s["sum"] + m * prior) / (s["count"] + m)


def compute_causal_seller_stats_on_sorted(sellers_sorted: np.ndarray, y_sorted: np.ndarray, prior: float):
    df_s = pd.DataFrame({"s": sellers_sorted, "y": y_sorted})
    cum_cnt = df_s.groupby("s").cumcount().values.astype(np.float64)
    cum_sum = (df_s.groupby("s")["y"].cumsum() - df_s["y"]).values.astype(np.float64)
    te_m2 = (cum_sum + 2.0 * prior) / (cum_cnt + 2.0)
    te_m5 = (cum_sum + 5.0 * prior) / (cum_cnt + 5.0)
    log_cnt = np.log1p(cum_cnt)
    return te_m2, te_m5, log_cnt, cum_cnt


def compute_static_seller_maps(sellers_tr: np.ndarray, y_tr: np.ndarray, prior: float):
    df_s = pd.DataFrame({"s": sellers_tr, "y": y_tr}).groupby("s")["y"].agg(["count", "sum"])
    te_m2_map = (df_s["sum"] + 2.0 * prior) / (df_s["count"] + 2.0)
    te_m5_map = (df_s["sum"] + 5.0 * prior) / (df_s["count"] + 5.0)
    cnt_map = np.log1p(df_s["count"])
    raw_cnt_map = df_s["count"]
    return te_m2_map, te_m5_map, cnt_map, raw_cnt_map


def build_base_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    price = np.maximum(0.0, df["price"].values.astype(np.float64))
    orig_price = np.maximum(price, df["orig_price"].values.astype(np.float64))
    init_q = np.maximum(0.0, df["init_qty"].values.astype(np.float64))
    avail_q = np.maximum(0.0, df["avail_qty"].values.astype(np.float64))
    sold_q = np.maximum(0.0, df["sold_qty"].values.astype(np.float64))

    log_price = np.log1p(price)
    log_init = np.log1p(init_q)
    log_avail = np.log1p(avail_q)
    log_sold = np.log1p(sold_q)
    sold_ratio = sold_q / (init_q + sold_q + 1.0)
    avail_ratio = avail_q / (init_q + 1.0)
    discount_ratio = (orig_price - price) / (orig_price + 1.0)

    return pd.DataFrame({
        "log_price": log_price,
        "log_init_qty": log_init,
        "log_avail_qty": log_avail,
        "log_sold_qty": log_sold,
        "sold_ratio": sold_ratio,
        "avail_ratio": avail_ratio,
        "discount_ratio": discount_ratio,
        "has_orig_price": df["has_orig_price"].values.astype(np.float32),
        "num_attrs": df["num_attrs"].values.astype(np.float32),
        "num_pics": df["num_pics"].values.astype(np.float32),
        "num_vars": df["num_vars"].values.astype(np.float32),
        "num_deals": df["num_deals"].values.astype(np.float32),
        "num_non_mp": df["num_non_mp"].values.astype(np.float32),
        "num_tags": df["num_tags"].values.astype(np.float32),
        "free_shipping": df["free_shipping"].values.astype(np.float32),
        "local_pick_up": df["local_pick_up"].values.astype(np.float32),
        "accepts_mercadopago": df["accepts_mercadopago"].values.astype(np.float32),
        "automatic_relist": df["automatic_relist"].values.astype(np.float32),
        "has_official_store": df["has_official_store"].values.astype(np.float32),
        "has_video": df["has_video"].values.astype(np.float32),
        "has_catalog": df["has_catalog"].values.astype(np.float32),
        "has_parent_item": df["has_parent_item"].values.astype(np.float32),
        "warr_has_mes_ano": df["warr_has_mes_ano"].values.astype(np.float32),
        "warr_sin_garantia": df["warr_sin_garantia"].values.astype(np.float32),
        "warr_fabrica": df["warr_fabrica"].values.astype(np.float32),
        "title_len": df["title_len"].values.astype(np.float32),
        "title_words": df["title_words"].values.astype(np.float32),
        "title_digits": df["title_digits"].values.astype(np.float32),
        "title_uppers": df["title_uppers"].values.astype(np.float32),
    })


def optimize_blend_weights(pred_list: list[np.ndarray], y_true: np.ndarray) -> list[float]:
    k = len(pred_list)
    P = np.column_stack([np.clip(p, 1e-6, 1.0 - 1e-6) for p in pred_list])

    def obj(w):
        w = np.asarray(w, dtype=np.float64)
        w = w / np.sum(w)
        blend = np.clip(P @ w, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y_true, blend))

    x0 = np.full(k, 1.0 / k)
    bounds = [(0.0, 1.0) for _ in range(k)]
    res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
    w_opt = np.maximum(0.0, res.x)
    w_opt = w_opt / np.sum(w_opt)
    return w_opt.tolist()


def main():
    t_start = time.time()
    log("Loading frozen T98 splits and pre-parsed train/test caches...")
    df_train = pd.read_pickle(SPLIT_DIR / "t98_parsed_train.pkl")
    df_test = pd.read_pickle(SPLIT_DIR / "t98_parsed_test.pkl")
    df_splits = pd.read_csv(SPLIT_DIR / "t98_dual_regime_outer_seller_group_splits.csv")

    assert len(df_train) == 70000 and len(df_splits) == 70000 and len(df_test) == 30000
    assert np.array_equal(df_train["row"].values, df_splits["row"].values)

    for df in (df_train, df_test):
        df["tags_attr_text"] = (
            df["tags_str"].fillna("") + " " + df["attr_text"].fillna("") + " " + df["warranty_str"].fillna("")
        ).str.strip()
        df["rich_linear_text"] = (
            df["title"].fillna("")
            + " "
            + df["subtitle"].fillna("")
            + " "
            + df["tags_attr_text"]
            + " brand_"
            + df["brand_val"].fillna("MISSING")
            + " cond_"
            + df["item_cond_attr"].fillna("MISSING")
            + " ltype_"
            + df["listing_type_id"].fillna("MISSING")
            + " bmode_"
            + df["buying_mode"].fillna("MISSING")
            + " smode_"
            + df["ship_mode"].fillna("MISSING")
            + " state_"
            + df["state_id"].fillna("MISSING")
            + " cat_"
            + df["category_id"].fillna("MISSING")
        )

    num_train_base = build_base_numeric_df(df_train)
    num_test_base = build_base_numeric_df(df_test)

    y_all = df_train["y"].values.astype(np.int64)
    sellers_all = df_train["seller_id"].values
    cats_all = df_train["category_id"].values
    price_all = np.maximum(0.0, df_train["price"].values.astype(np.float64))

    test_sellers = df_test["seller_id"].values
    test_cats = df_test["category_id"].values
    test_price = np.maximum(0.0, df_test["price"].values.astype(np.float64))

    corpus_train = (df_train["title"].fillna("") + " " + df_train["tags_str"].fillna("")).values
    corpus_test = (df_test["title"].fillna("") + " " + df_test["tags_str"].fillna("")).values

    rich_corpus_train = df_train["rich_linear_text"].values
    rich_corpus_test = df_test["rich_linear_text"].values

    CAT_COLS_ENRICHED = [
        "category_id",
        "listing_type_id",
        "buying_mode",
        "ship_mode",
        "state_id",
        "brand_val",
        "item_cond_attr",
        "seller_id_masked",
    ]
    INNER_CAT_COLS = [
        "listing_type_id",
        "buying_mode",
        "ship_mode",
        "state_id",
        "brand_val",
        "item_cond_attr",
    ]
    TEXT_COLS_ENRICHED = ["title", "tags_attr_text"]

    TP_TINY = {
        "tokenizers": [{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " ", "lowercasing": "true"}],
        "dictionaries": [{"dictionary_id": "Uni", "max_dictionary_size": "3000", "occurrence_lower_bound": "15", "gram_order": "1"}],
        "feature_processing": {
            "default": [{"tokenizers_names": ["Space"], "dictionaries_names": ["Uni"], "feature_calcers": ["BoW:top_tokens_count=8"]}]
        },
    }
    TP_2A = {
        "tokenizers": [{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " ", "lowercasing": "true"}],
        "dictionaries": [{"dictionary_id": "Uni", "max_dictionary_size": "12000", "occurrence_lower_bound": "4", "gram_order": "1"}],
        "feature_processing": {
            "default": [{"tokenizers_names": ["Space"], "dictionaries_names": ["Uni"], "feature_calcers": ["BoW:top_tokens_count=64"]}]
        },
    }

    oof_unseen_cb = np.zeros(len(df_train), dtype=np.float64)
    oof_unseen_2a = np.zeros(len(df_train), dtype=np.float64)
    oof_unseen_2b_unseen = np.zeros(len(df_train), dtype=np.float64)
    oof_unseen_2b_seen = np.zeros(len(df_train), dtype=np.float64)
    oof_unseen_text = np.zeros(len(df_train), dtype=np.float64)
    oof_unseen_attr = np.zeros(len(df_train), dtype=np.float64)

    seen_records = []
    test_preds_per_fold = []
    test_2a_per_fold = []
    test_2b_unseen_per_fold = []
    test_2b_seen_per_fold = []
    test_text_per_fold = []
    test_attr_per_fold = []
    test_seen_mask_per_fold = []

    provenance_rows = []
    per_fold_rows = []
    frozen_policies = []

    t97_df = pd.read_csv(ROOT / "results" / "seller_group_nested_podium_20261111" / "paired_outer_val_predictions.csv")
    p_t97_cb_all = t97_df["p_cb"].values
    p_t97_A_all = t97_df["p_A_podium_recipe"].values

    for k in range(5):
        t_fold = time.time()
        role_k = df_splits[f"role_fold_{k}"].values
        inner_k = df_splits[f"inner_fold_{k}"].values

        fit_idx = np.where(role_k == "fit")[0]
        seen_val_idx = np.where(role_k == "seen_val")[0]
        unseen_val_idx = np.where(role_k == "unseen_val")[0]
        inner_f = inner_k[fit_idx]

        # Strict split integrity assertions
        assert len(set(sellers_all[fit_idx]) & set(sellers_all[unseen_val_idx])) == 0
        assert set(sellers_all[seen_val_idx]).issubset(set(sellers_all[fit_idx]))
        for j in range(4):
            assert len(set(sellers_all[fit_idx[inner_f != j]]) & set(sellers_all[fit_idx[inner_f == j]])) == 0

        npz_path = OUT_DIR / f"fold_{k}_predictions.npz"
        freeze_path = OUT_DIR / f"freeze_line2_outer{k}.json"
        prov_cache_path = OUT_DIR / f"fold_{k}_meta.json"

        if npz_path.exists() and freeze_path.exists() and prov_cache_path.exists():
            log(f"=== Outer Fold {k+1}/5: loading cached fold_{k}_predictions.npz ===")
            z = np.load(npz_path)
            meta_k = json.loads(prov_cache_path.read_text(encoding="utf-8"))
            freeze_k = json.loads(freeze_path.read_text(encoding="utf-8"))

            p_cb_fit_oof = z["p_cb_fit_oof"]
            p_cb_seen_val = z["p_cb_seen_val"]
            p_cb_unseen_val = z["p_cb_unseen_val"]
            p_cb_test = z["p_cb_test"]

            p_2a_seen = z["p_2a_seen_val"]
            p_2a_unseen = z["p_2a_unseen_val"]
            p_2a_test = z["p_2a_test"]

            p_2b_unseen_seen = z["p_2b_unseen_seen_val"]
            p_2b_unseen_unseen = z["p_2b_unseen_unseen_val"]
            p_2b_unseen_test = z["p_2b_unseen_test"]

            p_2b_seen_seen = z["p_2b_seen_seen_val"]
            p_2b_seen_unseen = z["p_2b_seen_unseen_val"]
            p_2b_seen_test = z["p_2b_seen_test"]

            p_text_seen = z["p_text_seen_val"]
            p_text_unseen = z["p_text_unseen_val"]
            p_text_test = z["p_text_test"]

            p_attr_seen = z["p_attr_seen_val"]
            p_attr_unseen = z["p_attr_unseen_val"]
            p_attr_test = z["p_attr_test"]
            test_is_seen = z["test_is_seen"]

            provenance_rows.extend(meta_k["provenance"])
            per_fold_rows.append(meta_k["fold_metrics"])
            frozen_policies.append(freeze_k)
        else:
            log(f"=== Outer Fold {k+1}/5: fit={len(fit_idx)}, seen_val={len(seen_val_idx)}, unseen_val={len(unseen_val_idx)} ===")
            t_s1 = time.time()

            # Step 1: Leak-Free 4-fold Seller-Grouped OOF p_text_oof, p_attr_oof, and cat_te inside fit_idx
            cw = CountVectorizer(ngram_range=(1, 2), min_df=3).fit(corpus_train[fit_idx])
            cc = CountVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=5).fit(corpus_train[fit_idx])
            Cw_fit = cw.transform(corpus_train[fit_idx])
            Cc_fit = cc.transform(corpus_train[fit_idx])

            c_rich_w = CountVectorizer(ngram_range=(1, 2), min_df=3).fit(rich_corpus_train[fit_idx])
            c_rich_c = CountVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=5).fit(rich_corpus_train[fit_idx])
            Crw_fit = c_rich_w.transform(rich_corpus_train[fit_idx])
            Crc_fit = c_rich_c.transform(rich_corpus_train[fit_idx])

            p_text_oof = np.zeros(len(fit_idx), dtype=np.float64)
            p_attr_oof = np.zeros(len(fit_idx), dtype=np.float64)
            cat_te_oof = np.zeros(len(fit_idx), dtype=np.float64)
            cat_te_m2_oof = np.zeros(len(fit_idx), dtype=np.float64)
            cat_cnt_oof = np.zeros(len(fit_idx), dtype=np.float64)

            y_fit = y_all[fit_idx]
            cats_fit = cats_all[fit_idx]
            sellers_fit = sellers_all[fit_idx]

            for j in range(4):
                m_tr = inner_f != j
                m_va = inner_f == j
                p_text_oof[m_va], _ = fit_predict_saga_lr(
                    Cw_fit[m_tr], Cc_fit[m_tr], y_fit[m_tr], Cw_fit[m_va], Cc_fit[m_va], max_w=40000, max_c=60000, C_val=1.5
                )
                p_attr_oof[m_va], _ = fit_predict_saga_lr(
                    Crw_fit[m_tr], Crc_fit[m_tr], y_fit[m_tr], Crw_fit[m_va], Crc_fit[m_va], max_w=40000, max_c=60000, C_val=1.2
                )
                pr_j = float(y_fit[m_tr].mean())
                cm10 = compute_cat_te_map(cats_fit[m_tr], y_fit[m_tr], 10.0, pr_j)
                cm2 = compute_cat_te_map(cats_fit[m_tr], y_fit[m_tr], 2.0, pr_j)
                ccnt = pd.Series(cats_fit[m_tr]).value_counts()
                cat_te_oof[m_va] = pd.Series(cats_fit[m_va]).map(cm10).fillna(pr_j).values
                cat_te_m2_oof[m_va] = pd.Series(cats_fit[m_va]).map(cm2).fillna(pr_j).values
                cat_cnt_oof[m_va] = np.log1p(pd.Series(cats_fit[m_va]).map(ccnt).fillna(0.0).values)

            p_text_oof_sha = sha256_arr(p_text_oof)
            cat_te_oof_sha = sha256_arr(cat_te_oof)

            # Full fit_idx SAGA LR and category TE for seen_val, unseen_val, and test
            eval_corpus = np.concatenate([corpus_train[seen_val_idx], corpus_train[unseen_val_idx], corpus_test])
            Cw_eval = cw.transform(eval_corpus)
            Cc_eval = cc.transform(eval_corpus)
            p_text_eval, lr_info = fit_predict_saga_lr(Cw_fit, Cc_fit, y_fit, Cw_eval, Cw_eval if False else Cc_eval, max_w=40000, max_c=60000, C_val=1.5)

            n_seen = len(seen_val_idx)
            n_unseen = len(unseen_val_idx)
            p_text_seen = p_text_eval[:n_seen]
            p_text_unseen = p_text_eval[n_seen : n_seen + n_unseen]
            p_text_test = p_text_eval[n_seen + n_unseen :]

            eval_rich_corpus = np.concatenate([rich_corpus_train[seen_val_idx], rich_corpus_train[unseen_val_idx], rich_corpus_test])
            Crw_eval = c_rich_w.transform(eval_rich_corpus)
            Crc_eval = c_rich_c.transform(eval_rich_corpus)
            p_attr_eval, _ = fit_predict_saga_lr(Crw_fit, Crc_fit, y_fit, Crw_eval, Crc_eval, max_w=40000, max_c=60000, C_val=1.2)
            p_attr_seen = p_attr_eval[:n_seen]
            p_attr_unseen = p_attr_eval[n_seen : n_seen + n_unseen]
            p_attr_test = p_attr_eval[n_seen + n_unseen :]

            prior_fit = float(y_fit.mean())
            cm10_full = compute_cat_te_map(cats_fit, y_fit, 10.0, prior_fit)
            cm2_full = compute_cat_te_map(cats_fit, y_fit, 2.0, prior_fit)
            ccnt_full = pd.Series(cats_fit).value_counts()

            cat_te_seen = pd.Series(cats_all[seen_val_idx]).map(cm10_full).fillna(prior_fit).values
            cat_te_unseen = pd.Series(cats_all[unseen_val_idx]).map(cm10_full).fillna(prior_fit).values
            cat_te_test = pd.Series(test_cats).map(cm10_full).fillna(prior_fit).values

            cat_te_m2_seen = pd.Series(cats_all[seen_val_idx]).map(cm2_full).fillna(prior_fit).values
            cat_te_m2_unseen = pd.Series(cats_all[unseen_val_idx]).map(cm2_full).fillna(prior_fit).values
            cat_te_m2_test = pd.Series(test_cats).map(cm2_full).fillna(prior_fit).values

            cat_cnt_seen = np.log1p(pd.Series(cats_all[seen_val_idx]).map(ccnt_full).fillna(0.0).values)
            cat_cnt_unseen = np.log1p(pd.Series(cats_all[unseen_val_idx]).map(ccnt_full).fillna(0.0).values)
            cat_cnt_test = np.log1p(pd.Series(test_cats).map(ccnt_full).fillna(0.0).values)

            log(
                f"  Step 1 done ({time.time()-t_s1:.1f}s): p_text_oof acc={accuracy_score(y_fit, p_text_oof>=0.5):.5f}, "
                f"p_attr_oof acc={accuracy_score(y_fit, p_attr_oof>=0.5):.5f}"
            )

            # Fold-local category price stats (strictly on fit_idx)
            s_price_fit = pd.DataFrame({"c": cats_fit, "p": price_all[fit_idx], "lp": np.log1p(price_all[fit_idx])})
            cat_med_map = s_price_fit.groupby("c")["p"].median()
            cat_lmean_map = s_price_fit.groupby("c")["lp"].mean()
            cat_lstd_map = s_price_fit.groupby("c")["lp"].std().fillna(0.5)
            global_med = float( np.median(price_all[fit_idx]) )
            global_lmean = float( np.mean(np.log1p(price_all[fit_idx])) )

            def add_fold_features(
                base_num_df: pd.DataFrame,
                cats_arr: np.ndarray,
                prices_arr: np.ndarray,
                ptx: np.ndarray,
                pattr: np.ndarray,
                cte10: np.ndarray,
                cte2: np.ndarray,
                ccnt: np.ndarray,
                bt_arr: np.ndarray,
                ba_arr: np.ndarray,
            ) -> pd.DataFrame:
                out = base_num_df.reset_index(drop=True).copy()
                med_c = pd.Series(cats_arr).map(cat_med_map).fillna(global_med).values
                lmean_c = pd.Series(cats_arr).map(cat_lmean_map).fillna(global_lmean).values
                lstd_c = pd.Series(cats_arr).map(cat_lstd_map).fillna(0.5).values
                lp = np.log1p(prices_arr)
                out["price_ratio"] = np.clip(prices_arr / (med_c + 1.0), 0.0, 50.0)
                out["price_zscore"] = np.clip((lp - lmean_c) / (lstd_c + 0.1), -10.0, 10.0)
                out["p_text_oof"] = ptx
                out["p_attr_oof"] = pattr
                out["logit_p_text"] = logit(ptx)
                out["logit_p_attr"] = logit(pattr)
                out["cat_te"] = cte10
                out["cat_te_m2"] = cte2
                out["cat_cnt"] = ccnt
                bt_df = pd.DataFrame(bt_arr, columns=[f"bt_{i}" for i in range(bt_arr.shape[1])])
                ba_df = pd.DataFrame(ba_arr, columns=[f"ba_{i}" for i in range(ba_arr.shape[1])])
                return pd.concat([out, bt_df, ba_df], axis=1)

            # Fold-local binary token indicators (96 title + 96 tags_attr_text) fit strictly on fit_idx
            cv_title_bin = CountVectorizer(max_features=96, binary=True, min_df=5).fit(df_train["title"].values[fit_idx])
            cv_attr_bin = CountVectorizer(max_features=96, binary=True, min_df=5).fit(df_train["tags_attr_text"].values[fit_idx])

            bt_fit = cv_title_bin.transform(df_train["title"].values[fit_idx]).toarray().astype(np.float32)
            ba_fit = cv_attr_bin.transform(df_train["tags_attr_text"].values[fit_idx]).toarray().astype(np.float32)

            bt_seen = cv_title_bin.transform(df_train["title"].values[seen_val_idx]).toarray().astype(np.float32)
            ba_seen = cv_attr_bin.transform(df_train["tags_attr_text"].values[seen_val_idx]).toarray().astype(np.float32)

            bt_unseen = cv_title_bin.transform(df_train["title"].values[unseen_val_idx]).toarray().astype(np.float32)
            ba_unseen = cv_attr_bin.transform(df_train["tags_attr_text"].values[unseen_val_idx]).toarray().astype(np.float32)

            bt_test = cv_title_bin.transform(df_test["title"].values).toarray().astype(np.float32)
            ba_test = cv_attr_bin.transform(df_test["tags_attr_text"].values).toarray().astype(np.float32)

            X_num_fit = add_fold_features(
                num_train_base.iloc[fit_idx], cats_fit, price_all[fit_idx], p_text_oof, p_attr_oof, cat_te_oof, cat_te_m2_oof, cat_cnt_oof, bt_fit, ba_fit
            )
            X_num_seen = add_fold_features(
                num_train_base.iloc[seen_val_idx], cats_all[seen_val_idx], price_all[seen_val_idx], p_text_seen, p_attr_seen, cat_te_seen, cat_te_m2_seen, cat_cnt_seen, bt_seen, ba_seen
            )
            X_num_unseen = add_fold_features(
                num_train_base.iloc[unseen_val_idx], cats_all[unseen_val_idx], price_all[unseen_val_idx], p_text_unseen, p_attr_unseen, cat_te_unseen, cat_te_m2_unseen, cat_cnt_unseen, bt_unseen, ba_unseen
            )
            X_num_test = add_fold_features(
                num_test_base, test_cats, test_price, p_text_test, p_attr_test, cat_te_test, cat_te_m2_test, cat_cnt_test, bt_test, ba_test
            )

            # Causal chronological seller statistics on fit_idx (sorted by start_time, row)
            chrono_order = np.lexsort((df_train["row"].values[fit_idx], df_train["start_time"].values[fit_idx]))
            inv_chrono = np.empty_like(chrono_order)
            inv_chrono[chrono_order] = np.arange(len(chrono_order))

            te_m2_sorted, te_m5_sorted, lcnt_sorted, raw_prior_cnt_sorted = compute_causal_seller_stats_on_sorted(
                sellers_fit[chrono_order], y_fit[chrono_order], prior_fit
            )
            sel_causal_te_m2_fit = te_m2_sorted[inv_chrono]
            sel_causal_te_m5_fit = te_m5_sorted[inv_chrono]
            sel_causal_cnt_fit = lcnt_sorted[inv_chrono]
            sel_raw_prior_cnt_fit = raw_prior_cnt_sorted[inv_chrono]

            # Static seller maps from full fit_idx for seen_val, unseen_val, and test
            s_te2_map, s_te5_map, s_cnt_map, s_raw_cnt_map = compute_static_seller_maps(sellers_fit, y_fit, prior_fit)
            fit_seller_set = set(sellers_fit)
            multi_fit_sellers = set(s_raw_cnt_map[s_raw_cnt_map >= 2].index)

            # Step 2: 4-Fold Inner Seller-Group OOF on fit_idx (iterations=900, depth=8, learning_rate=0.05)
            t_in = time.time()
            p_2a_fit_oof = np.zeros(len(fit_idx), dtype=np.float64)
            p_2b_unseen_fit_oof = np.zeros(len(fit_idx), dtype=np.float64)
            p_2b_seen_unseen_mode_oof = np.zeros(len(fit_idx), dtype=np.float64)
            p_2b_seen_causal_mode_oof = np.zeros(len(fit_idx), dtype=np.float64)

            df_fit_cats_inner = df_train.iloc[fit_idx][INNER_CAT_COLS].reset_index(drop=True)
            df_fit_inner_unseen = pd.concat([df_fit_cats_inner, X_num_fit], axis=1)

            df_fit_inner_seen = df_fit_inner_unseen.copy()
            df_fit_inner_seen["sel_causal_te_m2"] = sel_causal_te_m2_fit
            df_fit_inner_seen["sel_causal_te_m5"] = sel_causal_te_m5_fit
            df_fit_inner_seen["sel_causal_cnt"] = sel_causal_cnt_fit

            # Model 2A 9-feature DataFrame on fit_idx
            df_2a_fit_inner = pd.DataFrame({
                "price": X_num_fit["log_price"].values,
                "price_ratio": X_num_fit["price_ratio"].values,
                "num_attrs": X_num_fit["num_attrs"].values,
                "warranty": X_num_fit["warr_has_mes_ano"].values,
                "p_text": p_text_oof,
                "cat_te": cat_te_oof,
            })
            df_2a_fit_inner = pd.concat([df_2a_fit_inner, pd.DataFrame(bt_fit[:, :48], columns=[f"bt_{i}" for i in range(48)])], axis=1)

            for j in range(4):
                m_tr = inner_f != j
                m_va = inner_f == j

                # 2A inner
                cb_2a_in = CatBoostClassifier(
                    iterations=INNER_ITERS, depth=DEPTH, learning_rate=INNER_LR, l2_leaf_reg=L2_REG,
                    loss_function="Logloss", boosting_type="Plain", rsm=0.5, border_count=64,
                    thread_count=N_THREADS, random_seed=42 + k * 10 + j, verbose=0
                )
                cb_2a_in.fit(df_2a_fit_inner.iloc[m_tr], y_fit[m_tr])
                p_2a_fit_oof[m_va] = cb_2a_in.predict_proba(df_2a_fit_inner.iloc[m_va])[:, 1]

                # 2B unseen_safe inner
                pool_in_tr_u = Pool(df_fit_inner_unseen.iloc[m_tr], y_fit[m_tr], cat_features=INNER_CAT_COLS)
                pool_in_va_u = Pool(df_fit_inner_unseen.iloc[m_va], cat_features=INNER_CAT_COLS)
                cb_2b_u_in = CatBoostClassifier(
                    iterations=INNER_ITERS, depth=DEPTH, learning_rate=INNER_LR, l2_leaf_reg=L2_REG,
                    loss_function="Logloss", boosting_type="Plain", one_hot_max_size=64, rsm=0.35, border_count=64,
                    thread_count=N_THREADS, random_seed=142 + k * 10 + j, verbose=0
                )
                cb_2b_u_in.fit(pool_in_tr_u)
                p_2b_unseen_fit_oof[m_va] = cb_2b_u_in.predict_proba(pool_in_va_u)[:, 1]

                # 2B with_seller inner (evaluated in causal-seen mode AND unseen-prior mode on m_va)
                pool_in_tr_s = Pool(df_fit_inner_seen.iloc[m_tr], y_fit[m_tr], cat_features=INNER_CAT_COLS)
                pool_in_va_s_causal = Pool(df_fit_inner_seen.iloc[m_va], cat_features=INNER_CAT_COLS)

                df_va_s_unseen = df_fit_inner_seen.iloc[m_va].copy()
                pr_tr_j = float(y_fit[m_tr].mean())
                df_va_s_unseen["sel_causal_te_m2"] = pr_tr_j
                df_va_s_unseen["sel_causal_te_m5"] = pr_tr_j
                df_va_s_unseen["sel_causal_cnt"] = 0.0
                pool_in_va_s_unseen = Pool(df_va_s_unseen, cat_features=INNER_CAT_COLS)

                cb_2b_s_in = CatBoostClassifier(
                    iterations=INNER_ITERS, depth=DEPTH, learning_rate=INNER_LR, l2_leaf_reg=L2_REG,
                    loss_function="Logloss", boosting_type="Plain", one_hot_max_size=64, rsm=0.35, border_count=64,
                    thread_count=N_THREADS, random_seed=242 + k * 10 + j, verbose=0
                )
                cb_2b_s_in.fit(pool_in_tr_s)
                p_2b_seen_causal_mode_oof[m_va] = cb_2b_s_in.predict_proba(pool_in_va_s_causal)[:, 1]
                p_2b_seen_unseen_mode_oof[m_va] = cb_2b_s_in.predict_proba(pool_in_va_s_unseen)[:, 1]

            # Optimize regime-specific blend weights strictly on fit_idx inner-OOF
            # Unseen regime weights: optimized across all fit_idx inner-OOF rows (100% unseen across inner folds)
            w_unseen = optimize_blend_weights(
                [p_2b_unseen_fit_oof, p_2b_seen_unseen_mode_oof, p_2a_fit_oof, p_text_oof, p_attr_oof],
                y_fit,
            )
            # Seen regime weights: optimized on causal seen subset of fit_idx (items with >=1 strictly earlier item of same seller)
            causal_seen_mask = sel_raw_prior_cnt_fit >= 1.0
            w_seen = optimize_blend_weights(
                [
                    p_2b_seen_causal_mode_oof[causal_seen_mask],
                    p_2b_unseen_fit_oof[causal_seen_mask],
                    p_2a_fit_oof[causal_seen_mask],
                    p_text_oof[causal_seen_mask],
                    p_attr_oof[causal_seen_mask],
                ],
                y_fit[causal_seen_mask],
            )

            p_cb_fit_oof = (
                w_unseen[0] * p_2b_unseen_fit_oof
                + w_unseen[1] * p_2b_seen_unseen_mode_oof
                + w_unseen[2] * p_2a_fit_oof
                + w_unseen[3] * p_text_oof
                + w_unseen[4] * p_attr_oof
            )
            p_cb_fit_seen_subset_oof = (
                w_seen[0] * p_2b_seen_causal_mode_oof[causal_seen_mask]
                + w_seen[1] * p_2b_unseen_fit_oof[causal_seen_mask]
                + w_seen[2] * p_2a_fit_oof[causal_seen_mask]
                + w_seen[3] * p_text_oof[causal_seen_mask]
                + w_seen[4] * p_attr_oof[causal_seen_mask]
            )

            freeze_k = {
                "outer_fold": k + 1,
                "components": ["CB_enriched_unseen_safe", "CB_enriched_with_seller", "CB_clean_baseline", "p_text_saga", "p_attr_saga"],
                "w_unseen": w_unseen,
                "w_seen": w_seen,
                "threshold": 0.5,
                "inner_oof_unseen_acc": {
                    "CB_clean_baseline": round(float(accuracy_score(y_fit, p_2a_fit_oof >= 0.5)), 5),
                    "CB_enriched_unseen_safe": round(float(accuracy_score(y_fit, p_2b_unseen_fit_oof >= 0.5)), 5),
                    "CB_enriched_with_seller_unseen_mode": round(float(accuracy_score(y_fit, p_2b_seen_unseen_mode_oof >= 0.5)), 5),
                    "blend_unseen": round(float(accuracy_score(y_fit, p_cb_fit_oof >= 0.5)), 5),
                },
                "inner_oof_causal_seen_acc": {
                    "n_causal_seen_in_fit": int(causal_seen_mask.sum()),
                    "CB_enriched_unseen_safe": round(float(accuracy_score(y_fit[causal_seen_mask], p_2b_unseen_fit_oof[causal_seen_mask] >= 0.5)), 5),
                    "CB_enriched_with_seller_causal": round(float(accuracy_score(y_fit[causal_seen_mask], p_2b_seen_causal_mode_oof[causal_seen_mask] >= 0.5)), 5),
                    "blend_seen": round(float(accuracy_score(y_fit[causal_seen_mask], p_cb_fit_seen_subset_oof >= 0.5)), 5),
                },
                "p_text_oof_sha256": p_text_oof_sha,
                "cat_te_oof_sha256": cat_te_oof_sha,
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            freeze_path.write_text(json.dumps(freeze_k, indent=2), encoding="utf-8")
            frozen_policies.append(freeze_k)
            log(
                f"  Inner 4-fold OOF done ({time.time()-t_in:.1f}s): unseen_blend_acc={freeze_k['inner_oof_unseen_acc']['blend_unseen']:.5f}, "
                f"causal_seen_blend_acc={freeze_k['inner_oof_causal_seen_acc']['blend_seen']:.5f}"
            )
            log(f"  Frozen weights: w_unseen={np.round(w_unseen, 3).tolist()}, w_seen={np.round(w_seen, 3).tolist()}")

            # Step 3: Outer Refits on full fit_idx (iterations=1200, depth=8, learning_rate=0.045, l2_leaf_reg=5.0)
            # 3A: Model 2A (CB_clean_baseline — exact 9 Podium features with leak-free p_text_oof and cat_te_oof)
            t_2a = time.time()
            def make_2a_outer_df(df_src, idx_arr, lp_arr, pr_arr, ptx_arr, cte_arr):
                sub = df_src if idx_arr is None else df_src.iloc[idx_arr]
                return pd.DataFrame({
                    "title": sub["title"].values,
                    "price": lp_arr,
                    "price_ratio": pr_arr,
                    "num_attrs": sub["num_attrs"].values,
                    "warranty": sub["warr_has_mes_ano"].values,
                    "p_text": ptx_arr,
                    "cat_te": cte_arr,
                    "seller_id": sub["seller_id"].values,
                    "category_id": sub["category_id"].values,
                })

            df_2a_tr = make_2a_outer_df(df_train, fit_idx, X_num_fit["log_price"].values, X_num_fit["price_ratio"].values, p_text_oof, cat_te_oof)
            df_2a_seen = make_2a_outer_df(df_train, seen_val_idx, X_num_seen["log_price"].values, X_num_seen["price_ratio"].values, p_text_seen, cat_te_seen)
            df_2a_unseen = make_2a_outer_df(df_train, unseen_val_idx, X_num_unseen["log_price"].values, X_num_unseen["price_ratio"].values, p_text_unseen, cat_te_unseen)
            df_2a_test = make_2a_outer_df(df_test, None, X_num_test["log_price"].values, X_num_test["price_ratio"].values, p_text_test, cat_te_test)

            cb_2a = CatBoostClassifier(
                iterations=OUTER_ITERS, depth=DEPTH, learning_rate=OUTER_LR, l2_leaf_reg=L2_REG,
                loss_function="Logloss", boosting_type="Plain",
                text_features=["title"], cat_features=["seller_id", "category_id"],
                text_processing=TP_2A, rsm=0.4, border_count=64, max_ctr_complexity=1,
                simple_ctr=["Borders"], combinations_ctr=["Borders"],
                thread_count=N_THREADS, random_seed=42 + k, verbose=0
            )
            cb_2a.fit(df_2a_tr, y_fit)
            p_2a_seen = cb_2a.predict_proba(df_2a_seen)[:, 1]
            p_2a_unseen = cb_2a.predict_proba(df_2a_unseen)[:, 1]
            p_2a_test = cb_2a.predict_proba(df_2a_test)[:, 1]
            t_2a_sec = round(time.time() - t_2a, 1)
            log(f"  Outer 2A (CB_clean_baseline) done ({t_2a_sec}s): unseen_acc={accuracy_score(y_all[unseen_val_idx], p_2a_unseen>=0.5):.5f}, seen_acc={accuracy_score(y_all[seen_val_idx], p_2a_seen>=0.5):.5f}")

            # 3B: Model 2B-Unseen (CB_enriched_unseen_safe — native text_features + cat_features with seller_id_masked="UNSEEN_SELLER")
            t_2bu = time.time()
            def make_2b_unseen_df(df_src, idx_arr, X_num_part):
                sub = (df_src if idx_arr is None else df_src.iloc[idx_arr]).reset_index(drop=True)
                meta_part = sub[["title", "tags_attr_text", "category_id", "listing_type_id", "buying_mode", "ship_mode", "state_id", "brand_val", "item_cond_attr"]].copy()
                meta_part["seller_id_masked"] = "UNSEEN_SELLER"
                return pd.concat([meta_part, X_num_part.reset_index(drop=True)], axis=1)

            df_2bu_tr = make_2b_unseen_df(df_train, fit_idx, X_num_fit)
            df_2bu_seen = make_2b_unseen_df(df_train, seen_val_idx, X_num_seen)
            df_2bu_unseen = make_2b_unseen_df(df_train, unseen_val_idx, X_num_unseen)
            df_2bu_test = make_2b_unseen_df(df_test, None, X_num_test)

            pool_2bu_tr = Pool(df_2bu_tr, y_fit, text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)
            pool_2bu_seen = Pool(df_2bu_seen, text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)
            pool_2bu_unseen = Pool(df_2bu_unseen, text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)
            pool_2bu_test = Pool(df_2bu_test, text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)

            cb_2bu = CatBoostClassifier(
                iterations=OUTER_ITERS, depth=DEPTH, learning_rate=OUTER_LR, l2_leaf_reg=L2_REG,
                loss_function="Logloss", boosting_type="Plain", one_hot_max_size=64, max_ctr_complexity=1,
                simple_ctr=["Borders"], combinations_ctr=["Borders"], rsm=0.25, border_count=64,
                text_processing=TP_TINY, thread_count=N_THREADS, random_seed=142 + k, verbose=0
            )
            cb_2bu.fit(pool_2bu_tr)
            p_2b_unseen_seen = cb_2bu.predict_proba(pool_2bu_seen)[:, 1]
            p_2b_unseen_unseen = cb_2bu.predict_proba(pool_2bu_unseen)[:, 1]
            p_2b_unseen_test = cb_2bu.predict_proba(pool_2bu_test)[:, 1]
            t_2bu_sec = round(time.time() - t_2bu, 1)
            log(f"  Outer 2B-Unseen (CB_enriched_unseen_safe) done ({t_2bu_sec}s): unseen_acc={accuracy_score(y_all[unseen_val_idx], p_2b_unseen_unseen>=0.5):.5f}, seen_acc={accuracy_score(y_all[seen_val_idx], p_2b_unseen_seen>=0.5):.5f}")

            # 3C: Model 2B-Seen (CB_enriched_with_seller — chronological fit_idx order, has_time=True, seller_id_masked + causal seller stats)
            t_2bs = time.time()
            df_2bs_tr = df_2bu_tr.iloc[chrono_order].reset_index(drop=True).copy()
            s_fit_chrono = sellers_fit[chrono_order]
            df_2bs_tr["seller_id_masked"] = [s if s in multi_fit_sellers else "UNSEEN_SELLER" for s in s_fit_chrono]
            df_2bs_tr["sel_causal_te_m2"] = te_m2_sorted
            df_2bs_tr["sel_causal_te_m5"] = te_m5_sorted
            df_2bs_tr["sel_causal_cnt"] = lcnt_sorted

            def make_2b_seen_eval_df(base_2bu_df, sellers_eval):
                out = base_2bu_df.copy()
                out["seller_id_masked"] = [s if s in fit_seller_set else "UNSEEN_SELLER" for s in sellers_eval]
                out["sel_causal_te_m2"] = pd.Series(sellers_eval).map(s_te2_map).fillna(prior_fit).values
                out["sel_causal_te_m5"] = pd.Series(sellers_eval).map(s_te5_map).fillna(prior_fit).values
                out["sel_causal_cnt"] = pd.Series(sellers_eval).map(s_cnt_map).fillna(0.0).values
                return out

            df_2bs_seen = make_2b_seen_eval_df(df_2bu_seen, sellers_all[seen_val_idx])
            df_2bs_unseen = make_2b_seen_eval_df(df_2bu_unseen, sellers_all[unseen_val_idx])
            df_2bs_test = make_2b_seen_eval_df(df_2bu_test, test_sellers)

            pool_2bs_tr = Pool(df_2bs_tr, y_fit[chrono_order], text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)
            pool_2bs_seen = Pool(df_2bs_seen, text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)
            pool_2bs_unseen = Pool(df_2bs_unseen, text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)
            pool_2bs_test = Pool(df_2bs_test, text_features=TEXT_COLS_ENRICHED, cat_features=CAT_COLS_ENRICHED)

            cb_2bs = CatBoostClassifier(
                iterations=OUTER_ITERS, depth=DEPTH, learning_rate=OUTER_LR, l2_leaf_reg=L2_REG,
                loss_function="Logloss", boosting_type="Plain", has_time=True, one_hot_max_size=64, max_ctr_complexity=1,
                simple_ctr=["Borders"], combinations_ctr=["Borders"], rsm=0.25, border_count=64,
                text_processing=TP_TINY, thread_count=N_THREADS, random_seed=242 + k, verbose=0
            )
            cb_2bs.fit(pool_2bs_tr)
            p_2b_seen_seen = cb_2bs.predict_proba(pool_2bs_seen)[:, 1]
            p_2b_seen_unseen = cb_2bs.predict_proba(pool_2bs_unseen)[:, 1]
            p_2b_seen_test = cb_2bs.predict_proba(pool_2bs_test)[:, 1]
            t_2bs_sec = round(time.time() - t_2bs, 1)
            log(f"  Outer 2B-Seen (CB_enriched_with_seller) done ({t_2bs_sec}s): seen_acc={accuracy_score(y_all[seen_val_idx], p_2b_seen_seen>=0.5):.5f}, unseen_acc={accuracy_score(y_all[unseen_val_idx], p_2b_seen_unseen>=0.5):.5f}")

            # Apply frozen regime-aware blend weights
            p_cb_unseen_val = (
                w_unseen[0] * p_2b_unseen_unseen
                + w_unseen[1] * p_2b_seen_unseen
                + w_unseen[2] * p_2a_unseen
                + w_unseen[3] * p_text_unseen
                + w_unseen[4] * p_attr_unseen
            )
            p_cb_seen_val = (
                w_seen[0] * p_2b_seen_seen
                + w_seen[1] * p_2b_unseen_seen
                + w_seen[2] * p_2a_seen
                + w_seen[3] * p_text_seen
                + w_seen[4] * p_attr_seen
            )

            test_is_seen = np.isin(test_sellers, list(fit_seller_set))
            p_cb_test_unseen_branch = (
                w_unseen[0] * p_2b_unseen_test
                + w_unseen[1] * p_2b_seen_test
                + w_unseen[2] * p_2a_test
                + w_unseen[3] * p_text_test
                + w_unseen[4] * p_attr_test
            )
            p_cb_test_seen_branch = (
                w_seen[0] * p_2b_seen_test
                + w_seen[1] * p_2b_unseen_test
                + w_seen[2] * p_2a_test
                + w_seen[3] * p_text_test
                + w_seen[4] * p_attr_test
            )
            p_cb_test = np.where(test_is_seen, p_cb_test_seen_branch, p_cb_test_unseen_branch)

            # Save fold_{k}_predictions.npz
            np.savez_compressed(
                npz_path,
                fit_idx=fit_idx,
                seen_val_idx=seen_val_idx,
                unseen_val_idx=unseen_val_idx,
                p_cb_fit_oof=p_cb_fit_oof,
                p_cb_seen_val=p_cb_seen_val,
                p_cb_unseen_val=p_cb_unseen_val,
                p_cb_test=p_cb_test,
                p_2a_fit_oof=p_2a_fit_oof,
                p_2a_seen_val=p_2a_seen,
                p_2a_unseen_val=p_2a_unseen,
                p_2a_test=p_2a_test,
                p_2b_unseen_fit_oof=p_2b_unseen_fit_oof,
                p_2b_unseen_seen_val=p_2b_unseen_seen,
                p_2b_unseen_unseen_val=p_2b_unseen_unseen,
                p_2b_unseen_test=p_2b_unseen_test,
                p_2b_seen_fit_oof=p_2b_seen_causal_mode_oof,
                p_2b_seen_seen_val=p_2b_seen_seen,
                p_2b_seen_unseen_val=p_2b_seen_unseen,
                p_2b_seen_test=p_2b_seen_test,
                p_text_oof=p_text_oof,
                p_text_seen_val=p_text_seen,
                p_text_unseen_val=p_text_unseen,
                p_text_test=p_text_test,
                p_attr_oof=p_attr_oof,
                p_attr_seen_val=p_attr_seen,
                p_attr_unseen_val=p_attr_unseen,
                p_attr_test=p_attr_test,
                cat_te_oof=cat_te_oof,
                test_is_seen=test_is_seen,
            )

            # Record provenance and fold metrics
            y_un = y_all[unseen_val_idx]
            y_se = y_all[seen_val_idx]
            c_t97_cb = (p_t97_cb_all[unseen_val_idx] >= 0.5) == y_un
            c_t97_A = (p_t97_A_all[unseen_val_idx] >= 0.5) == y_un
            c_2a_un = (p_2a_unseen >= 0.5) == y_un
            c_2bu_un = (p_2b_unseen_unseen >= 0.5) == y_un
            c_blend_un = (p_cb_unseen_val >= 0.5) == y_un

            c_2a_se = (p_2a_seen >= 0.5) == y_se
            c_2bs_se = (p_2b_seen_seen >= 0.5) == y_se
            c_blend_se = (p_cb_seen_val >= 0.5) == y_se

            fold_metric_row = {
                "status": "DEVELOPMENT_ONLY",
                "outer_fold": k + 1,
                "n_fit": int(len(fit_idx)),
                "n_seen_val": int(len(seen_val_idx)),
                "n_unseen_val": int(len(unseen_val_idx)),
                "hits_unseen_t97_cb": int(c_t97_cb.sum()),
                "acc_unseen_t97_cb": round(float(c_t97_cb.mean()), 5),
                "ll_unseen_t97_cb": round(float(log_loss(y_un, np.clip(p_t97_cb_all[unseen_val_idx], 1e-7, 1 - 1e-7))), 5),
                "hits_unseen_t97_policyA": int(c_t97_A.sum()),
                "acc_unseen_t97_policyA": round(float(c_t97_A.mean()), 5),
                "ll_unseen_t97_policyA": round(float(log_loss(y_un, np.clip(p_t97_A_all[unseen_val_idx], 1e-7, 1 - 1e-7))), 5),
                "hits_unseen_2A_clean": int(c_2a_un.sum()),
                "acc_unseen_2A_clean": round(float(c_2a_un.mean()), 5),
                "ll_unseen_2A_clean": round(float(log_loss(y_un, np.clip(p_2a_unseen, 1e-7, 1 - 1e-7))), 5),
                "hits_unseen_2B_unseen_safe": int(c_2bu_un.sum()),
                "acc_unseen_2B_unseen_safe": round(float(c_2bu_un.mean()), 5),
                "ll_unseen_2B_unseen_safe": round(float(log_loss(y_un, np.clip(p_2b_unseen_unseen, 1e-7, 1 - 1e-7))), 5),
                "hits_unseen_line2_blend": int(c_blend_un.sum()),
                "acc_unseen_line2_blend": round(float(c_blend_un.mean()), 5),
                "ll_unseen_line2_blend": round(float(log_loss(y_un, np.clip(p_cb_unseen_val, 1e-7, 1 - 1e-7))), 5),
                "fixes_unseen_vs_t97_cb": int((c_blend_un & ~c_t97_cb).sum()),
                "breaks_unseen_vs_t97_cb": int((~c_blend_un & c_t97_cb).sum()),
                "delta_unseen_vs_t97_cb": int(c_blend_un.sum() - c_t97_cb.sum()),
                "fixes_unseen_vs_t97_policyA": int((c_blend_un & ~c_t97_A).sum()),
                "breaks_unseen_vs_t97_policyA": int((~c_blend_un & c_t97_A).sum()),
                "delta_unseen_vs_t97_policyA": int(c_blend_un.sum() - c_t97_A.sum()),
                "hits_seen_2A_clean": int(c_2a_se.sum()),
                "acc_seen_2A_clean": round(float(c_2a_se.mean()), 5),
                "ll_seen_2A_clean": round(float(log_loss(y_se, np.clip(p_2a_seen, 1e-7, 1 - 1e-7))), 5),
                "hits_seen_2B_with_seller": int(c_2bs_se.sum()),
                "acc_seen_2B_with_seller": round(float(c_2bs_se.mean()), 5),
                "ll_seen_2B_with_seller": round(float(log_loss(y_se, np.clip(p_2b_seen_seen, 1e-7, 1 - 1e-7))), 5),
                "hits_seen_line2_blend": int(c_blend_se.sum()),
                "acc_seen_line2_blend": round(float(c_blend_se.mean()), 5),
                "ll_seen_line2_blend": round(float(log_loss(y_se, np.clip(p_cb_seen_val, 1e-7, 1 - 1e-7))), 5),
                "fixes_seen_vs_2A_clean": int((c_blend_se & ~c_2a_se).sum()),
                "breaks_seen_vs_2A_clean": int((~c_blend_se & c_2a_se).sum()),
                "delta_seen_vs_2A_clean": int(c_blend_se.sum() - c_2a_se.sum()),
                "fold_seconds": round(time.time() - t_fold, 1),
            }

            prov_list = [
                {
                    "status": "DEVELOPMENT_ONLY",
                    "outer_fold": k + 1,
                    "model": f"outer{k}_CB_clean_baseline_2A",
                    "fit_rows_sha256": sha256_arr(np.sort(fit_idx)),
                    "seen_val_rows_sha256": sha256_arr(np.sort(seen_val_idx)),
                    "unseen_val_rows_sha256": sha256_arr(np.sort(unseen_val_idx)),
                    "p_text_oof_sha256": p_text_oof_sha,
                    "cat_te_oof_sha256": cat_te_oof_sha,
                    "n_fit": int(len(fit_idx)),
                    "n_fit_sellers": int(len(fit_seller_set)),
                    "vocab_word": lr_info["vocab_word"],
                    "vocab_char": lr_info["vocab_char"],
                    "p_text_oof_acc": round(float(accuracy_score(y_fit, p_text_oof >= 0.5)), 5),
                    "cb_iterations": int(cb_2a.tree_count_),
                    "seconds": t_2a_sec,
                },
                {
                    "status": "DEVELOPMENT_ONLY",
                    "outer_fold": k + 1,
                    "model": f"outer{k}_CB_enriched_unseen_safe_2B",
                    "fit_rows_sha256": sha256_arr(np.sort(fit_idx)),
                    "seen_val_rows_sha256": sha256_arr(np.sort(seen_val_idx)),
                    "unseen_val_rows_sha256": sha256_arr(np.sort(unseen_val_idx)),
                    "p_text_oof_sha256": p_text_oof_sha,
                    "cat_te_oof_sha256": cat_te_oof_sha,
                    "n_fit": int(len(fit_idx)),
                    "n_fit_sellers": int(len(fit_seller_set)),
                    "vocab_word": lr_info["vocab_word"],
                    "vocab_char": lr_info["vocab_char"],
                    "p_text_oof_acc": round(float(accuracy_score(y_fit, p_text_oof >= 0.5)), 5),
                    "cb_iterations": int(cb_2bu.tree_count_),
                    "seconds": t_2bu_sec,
                },
                {
                    "status": "DEVELOPMENT_ONLY",
                    "outer_fold": k + 1,
                    "model": f"outer{k}_CB_enriched_with_seller_2B",
                    "fit_rows_sha256": sha256_arr(np.sort(fit_idx)),
                    "seen_val_rows_sha256": sha256_arr(np.sort(seen_val_idx)),
                    "unseen_val_rows_sha256": sha256_arr(np.sort(unseen_val_idx)),
                    "p_text_oof_sha256": p_text_oof_sha,
                    "cat_te_oof_sha256": cat_te_oof_sha,
                    "n_fit": int(len(fit_idx)),
                    "n_fit_sellers": int(len(fit_seller_set)),
                    "vocab_word": lr_info["vocab_word"],
                    "vocab_char": lr_info["vocab_char"],
                    "p_text_oof_acc": round(float(accuracy_score(y_fit, p_text_oof >= 0.5)), 5),
                    "cb_iterations": int(cb_2bs.tree_count_),
                    "seconds": t_2bs_sec,
                },
            ]
            prov_cache_path.write_text(json.dumps({"provenance": prov_list, "fold_metrics": fold_metric_row}, indent=2), encoding="utf-8")
            provenance_rows.extend(prov_list)
            per_fold_rows.append(fold_metric_row)

        log(
            f"  Fold {k+1} Summary: Unseen Blend={per_fold_rows[-1]['hits_unseen_line2_blend']}/14000 "
            f"({per_fold_rows[-1]['acc_unseen_line2_blend']:.5f}, {per_fold_rows[-1]['delta_unseen_vs_t97_cb']:+d} vs T97_CB, "
            f"{per_fold_rows[-1]['delta_unseen_vs_t97_policyA']:+d} vs T97_A) | "
            f"Seen Blend={per_fold_rows[-1]['hits_seen_line2_blend']}/{per_fold_rows[-1]['n_seen_val']} "
            f"({per_fold_rows[-1]['acc_seen_line2_blend']:.5f}, {per_fold_rows[-1]['delta_seen_vs_2A_clean']:+d} vs 2A_clean)"
        )

        # Populate global unseen 70k arrays
        oof_unseen_cb[unseen_val_idx] = p_cb_unseen_val
        oof_unseen_2a[unseen_val_idx] = p_2a_unseen
        oof_unseen_2b_unseen[unseen_val_idx] = p_2b_unseen_unseen
        oof_unseen_2b_seen[unseen_val_idx] = p_2b_seen_unseen
        oof_unseen_text[unseen_val_idx] = p_text_unseen
        oof_unseen_attr[unseen_val_idx] = p_attr_unseen

        # Populate seen_val records for this fold
        seen_df_k = pd.DataFrame({
            "outer_fold": k + 1,
            "row": seen_val_idx,
            "id": df_train["id"].values[seen_val_idx],
            "seller_id": sellers_all[seen_val_idx],
            "category_id": cats_all[seen_val_idx],
            "y_true": y_all[seen_val_idx],
            "p_cb_clean_2a": p_2a_seen,
            "p_cb_enriched_unseen_safe": p_2b_unseen_seen,
            "p_cb_enriched_with_seller": p_2b_seen_seen,
            "p_text_oof_model": p_text_seen,
            "p_attr_oof_model": p_attr_seen,
            "p_cb_line2": p_cb_seen_val,
        })
        seen_records.append(seen_df_k)

        test_preds_per_fold.append(p_cb_test)
        test_2a_per_fold.append(p_2a_test)
        test_2b_unseen_per_fold.append(p_2b_unseen_test)
        test_2b_seen_per_fold.append(p_2b_seen_test)
        test_text_per_fold.append(p_text_test)
        test_attr_per_fold.append(p_attr_test)
        test_seen_mask_per_fold.append(test_is_seen.astype(np.float64))

        pd.DataFrame(per_fold_rows).to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)
        pd.DataFrame(provenance_rows).to_csv(OUT_DIR / "model_provenance.csv", index=False)

    # Save oof_unseen_70k.csv
    df_oof_unseen = pd.DataFrame({
        "row": df_train["row"].values,
        "id": df_train["id"].values,
        "seller_id": sellers_all,
        "category_id": cats_all,
        "outer_fold": df_splits["outer_fold"].values + 1,
        "y_true": y_all,
        "p_t97_cb": p_t97_cb_all,
        "p_t97_policy_A": p_t97_A_all,
        "p_cb_clean_2a": oof_unseen_2a,
        "p_cb_enriched_unseen_safe": oof_unseen_2b_unseen,
        "p_cb_enriched_with_seller": oof_unseen_2b_seen,
        "p_text_saga": oof_unseen_text,
        "p_attr_saga": oof_unseen_attr,
        "p_cb_line2": oof_unseen_cb,
    })
    oof_unseen_path = OUT_DIR / "oof_unseen_70k.csv"
    df_oof_unseen.to_csv(oof_unseen_path, index=False)

    # Save oof_seen_all_folds.csv
    df_oof_seen = pd.concat(seen_records, ignore_index=True)
    oof_seen_path = OUT_DIR / "oof_seen_all_folds.csv"
    df_oof_seen.to_csv(oof_seen_path, index=False)

    # Save test_predictions_30k.csv (probabilities only, zero inferred labels)
    p_test_line2 = np.mean(test_preds_per_fold, axis=0)
    p_test_2a = np.mean(test_2a_per_fold, axis=0)
    p_test_2b_unseen = np.mean(test_2b_unseen_per_fold, axis=0)
    p_test_2b_seen = np.mean(test_2b_seen_per_fold, axis=0)
    p_test_text = np.mean(test_text_per_fold, axis=0)
    p_test_attr = np.mean(test_attr_per_fold, axis=0)
    p_test_seen_frac = np.mean(test_seen_mask_per_fold, axis=0)

    s4_df = pd.read_csv(ROOT / "results" / "ultimate_podium_engine_23h" / "test_probabilities_podium_engine.csv")
    df_test_out = pd.DataFrame({
        "ID": s4_df["ID"].values,
        "item_id": df_test["id"].values,
        "p_cb_clean_2a": p_test_2a,
        "p_cb_enriched_unseen_safe": p_test_2b_unseen,
        "p_cb_enriched_with_seller": p_test_2b_seen,
        "p_text_saga": p_test_text,
        "p_attr_saga": p_test_attr,
        "seen_in_fit_fold_fraction": p_test_seen_frac,
        "p_cb_line2": p_test_line2,
    })
    test_pred_path = OUT_DIR / "test_predictions_30k.csv"
    df_test_out.to_csv(test_pred_path, index=False)

    # Save test_disagreements_vs_s4_probs_only.csv (ONLY test ID and model probabilities, zero inferred labels)
    p_s4 = s4_df["p_optimal_blend"].values
    disagree_mask = (p_test_line2 >= 0.5) != (p_s4 >= 0.5)
    df_disagree = pd.DataFrame({
        "ID": s4_df["ID"].values[disagree_mask],
        "p_s4_optimal_blend": p_s4[disagree_mask],
        "p_s4_cb": s4_df["p_cb"].values[disagree_mask],
        "p_cb_clean_2a": p_test_2a[disagree_mask],
        "p_cb_enriched_unseen_safe": p_test_2b_unseen[disagree_mask],
        "p_cb_enriched_with_seller": p_test_2b_seen[disagree_mask],
        "p_cb_line2": p_test_line2[disagree_mask],
    })
    disagree_path = OUT_DIR / "test_disagreements_vs_s4_probs_only.csv"
    df_disagree.to_csv(disagree_path, index=False)

    # Aggregate metrics & 2,000-replicate seller x category cluster bootstrap
    log("Running 2,000-replicate seller x category cluster bootstrap...")
    folds_df = pd.DataFrame(per_fold_rows)

    def summarize_pred(y_vec: np.ndarray, p_vec: np.ndarray) -> dict:
        p_c = np.clip(p_vec, 1e-7, 1.0 - 1e-7)
        hits = int(((p_c >= 0.5) == y_vec).sum())
        return {
            "n_rows": int(len(y_vec)),
            "hits": hits,
            "accuracy": round(float(hits / len(y_vec)), 5),
            "logloss": round(float(log_loss(y_vec, p_c)), 5),
        }

    summary_unseen = {
        "T97_CatBoost": summarize_pred(y_all, p_t97_cb_all),
        "T97_Policy_A": summarize_pred(y_all, p_t97_A_all),
        "Model_2A_CB_clean_baseline": summarize_pred(y_all, oof_unseen_2a),
        "Model_2B_CB_enriched_unseen_safe": summarize_pred(y_all, oof_unseen_2b_unseen),
        "Model_2B_CB_enriched_with_seller": summarize_pred(y_all, oof_unseen_2b_seen),
        "Line2_Selected_Blend": summarize_pred(y_all, oof_unseen_cb),
    }

    y_seen_all = df_oof_seen["y_true"].values
    sellers_seen_all = df_oof_seen["seller_id"].values
    cats_seen_all = df_oof_seen["category_id"].values
    p_seen_2a_all = df_oof_seen["p_cb_clean_2a"].values
    p_seen_2bu_all = df_oof_seen["p_cb_enriched_unseen_safe"].values
    p_seen_2bs_all = df_oof_seen["p_cb_enriched_with_seller"].values
    p_seen_line2_all = df_oof_seen["p_cb_line2"].values

    summary_seen = {
        "Model_2A_CB_clean_baseline": summarize_pred(y_seen_all, p_seen_2a_all),
        "Model_2B_CB_enriched_unseen_safe": summarize_pred(y_seen_all, p_seen_2bu_all),
        "Model_2B_CB_enriched_with_seller": summarize_pred(y_seen_all, p_seen_2bs_all),
        "Line2_Selected_Blend": summarize_pred(y_seen_all, p_seen_line2_all),
    }

    def paired_bootstrap_comparison(
        p_cand: np.ndarray,
        p_ref: np.ndarray,
        y_vec: np.ndarray,
        sellers_vec: np.ndarray,
        cats_vec: np.ndarray,
        fold_deltas: np.ndarray,
        seed_base: int,
    ) -> dict:
        n_total = len(y_vec)
        c_cand = ((p_cand >= 0.5) == y_vec).astype(np.int64)
        c_ref = ((p_ref >= 0.5) == y_vec).astype(np.int64)
        diff = (c_cand - c_ref).astype(np.float64)
        bs_hits = boot_two_way(diff, sellers_vec, cats_vec, n_boot=2000, seed=seed_base) * n_total

        pc = np.clip(p_cand, 1e-7, 1.0 - 1e-7)
        pr = np.clip(p_ref, 1e-7, 1.0 - 1e-7)
        lld = -(y_vec * np.log(pc) + (1 - y_vec) * np.log(1 - pc)) + (y_vec * np.log(pr) + (1 - y_vec) * np.log(1 - pr))
        bs_ll = boot_two_way(lld, sellers_vec, cats_vec, n_boot=2000, seed=seed_base + 1)

        return {
            "n_rows": int(n_total),
            "delta_hits": int(diff.sum()),
            "fixes": int((diff == 1).sum()),
            "breaks": int((diff == -1).sum()),
            "folds_positive": int((fold_deltas > 0).sum()),
            "folds_zero": int((fold_deltas == 0).sum()),
            "worst_fold_delta": int(fold_deltas.min()),
            "per_fold_deltas": [int(x) for x in fold_deltas],
            "boot_delta_hits_median": round(float(np.median(bs_hits)), 2),
            "boot_delta_hits_q05": round(float(np.quantile(bs_hits, 0.05)), 2),
            "boot_delta_hits_q95": round(float(np.quantile(bs_hits, 0.95)), 2),
            "boot_frac_gt0": round(float((bs_hits > 0).mean()), 4),
            "delta_logloss": round(float(lld.mean()), 5),
            "boot_delta_logloss_q05": round(float(np.quantile(bs_ll, 0.05)), 5),
            "boot_delta_logloss_q95": round(float(np.quantile(bs_ll, 0.95)), 5),
        }

    comparisons = {
        "unseen_2A_clean_vs_T97_CB": paired_bootstrap_comparison(
            oof_unseen_2a,
            p_t97_cb_all,
            y_all,
            sellers_all,
            cats_all,
            (folds_df["hits_unseen_2A_clean"] - folds_df["hits_unseen_t97_cb"]).values,
            20261111,
        ),
        "unseen_2B_unseen_safe_vs_T97_CB": paired_bootstrap_comparison(
            oof_unseen_2b_unseen,
            p_t97_cb_all,
            y_all,
            sellers_all,
            cats_all,
            (folds_df["hits_unseen_2B_unseen_safe"] - folds_df["hits_unseen_t97_cb"]).values,
            20261113,
        ),
        "unseen_Line2_Blend_vs_T97_CB": paired_bootstrap_comparison(
            oof_unseen_cb,
            p_t97_cb_all,
            y_all,
            sellers_all,
            cats_all,
            folds_df["delta_unseen_vs_t97_cb"].values,
            20261115,
        ),
        "unseen_Line2_Blend_vs_T97_Policy_A": paired_bootstrap_comparison(
            oof_unseen_cb,
            p_t97_A_all,
            y_all,
            sellers_all,
            cats_all,
            folds_df["delta_unseen_vs_t97_policyA"].values,
            20261117,
        ),
        "unseen_Line2_Blend_vs_2A_clean": paired_bootstrap_comparison(
            oof_unseen_cb,
            oof_unseen_2a,
            y_all,
            sellers_all,
            cats_all,
            (folds_df["hits_unseen_line2_blend"] - folds_df["hits_unseen_2A_clean"]).values,
            20261119,
        ),
        "seen_2B_with_seller_vs_2A_clean": paired_bootstrap_comparison(
            p_seen_2bs_all,
            p_seen_2a_all,
            y_seen_all,
            sellers_seen_all,
            cats_seen_all,
            (folds_df["hits_seen_2B_with_seller"] - folds_df["hits_seen_2A_clean"]).values,
            20261121,
        ),
        "seen_Line2_Blend_vs_2A_clean": paired_bootstrap_comparison(
            p_seen_line2_all,
            p_seen_2a_all,
            y_seen_all,
            sellers_seen_all,
            cats_seen_all,
            folds_df["delta_seen_vs_2A_clean"].values,
            20261123,
        ),
    }

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_head = "unknown"

    manifest = {
        "status": "DEVELOPMENT_ONLY",
        "task_id": "T98_Line2_CatBoost_Native_Dual_Regime",
        "submission_generated": False,
        "kaggle_api_used": False,
        "git_head_at_run": git_head,
        "script": "scripts/t98_line2_catboost_native.py",
        "script_sha256": sha256_file(Path(__file__)),
        "cx0164_compliance": {
            "in_sample_stack_features_forbidden": True,
            "p_text_oof_method": "4-fold StratifiedGroupKFold by seller_id inside fit_idx (0 seller overlap)",
            "cat_te_oof_method": "4-fold StratifiedGroupKFold by seller_id inside fit_idx (m=10.0, 0 seller overlap)",
            "outer_seller_overlap": 0,
            "inner_seller_overlap": 0,
            "causal_seen_val_verified": True,
            "inner_freeze_before_outer_eval": True,
        },
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "config": {
            "thread_count": N_THREADS,
            "outer_iterations": OUTER_ITERS,
            "outer_learning_rate": OUTER_LR,
            "inner_iterations": INNER_ITERS,
            "inner_learning_rate": INNER_LR,
            "depth": DEPTH,
            "l2_leaf_reg": L2_REG,
            "text_features": TEXT_COLS_ENRICHED,
            "cat_features": CAT_COLS_ENRICHED,
        },
        "summary_unseen_70k": summary_unseen,
        "summary_seen_32092": summary_seen,
        "paired_comparisons": comparisons,
        "test_disagreements_vs_s4_count": int(disagree_mask.sum()),
        "frozen_policies": frozen_policies,
        "files_sha256": {
            "oof_unseen_70k.csv": sha256_file(oof_unseen_path),
            "oof_seen_all_folds.csv": sha256_file(oof_seen_path),
            "test_predictions_30k.csv": sha256_file(test_pred_path),
            "test_disagreements_vs_s4_probs_only.csv": sha256_file(disagree_path),
            "per_fold_metrics.csv": sha256_file(OUT_DIR / "per_fold_metrics.csv"),
            "model_provenance.csv": sha256_file(OUT_DIR / "model_provenance.csv"),
        },
        "runtime_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("=== FINAL SUMMARY (UNSEEN 70K & SEEN 32,092) ===")
    log(json.dumps({"summary_unseen_70k": summary_unseen, "summary_seen_32092": summary_seen, "paired_comparisons": comparisons}, indent=2))
    log(f"DONE in {manifest['runtime_minutes']} minutes. All artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
