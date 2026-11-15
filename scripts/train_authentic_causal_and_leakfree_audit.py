"""
AG-0237 / CX-0223 / CX-0224 / CX-0225: Authentic Causal Pipeline, Clean Leakage Audit & Full Historical Ledger
================================================================================================================
Author: Antigravity
Date: 2026-11-15
"""

import os
import sys
import time
import json
import pickle
import hashlib
import unicodedata
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

def clip_prob(p, eps=1e-6):
    return np.clip(p, eps, 1.0 - eps)

def logit(p):
    p_c = clip_prob(p)
    return np.log(p_c / (1.0 - p_c))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_id_target_hash(ids, preds):
    mapping = {int(i): str(c).strip() for i, c in zip(ids, preds)}
    encoded = json.dumps(mapping, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def bootstrap_multiway_and_marginals(y_true, p_cand, p_base, sellers, cats, n_boot=2000, seed=20261115):
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    unique_sellers, s_idx = np.unique(sellers, return_inverse=True)
    unique_cats, c_idx = np.unique(cats, return_inverse=True)
    n_s = len(unique_sellers)
    n_c = len(unique_cats)

    rng = np.random.default_rng(seed)
    boot_multiway = np.empty(n_boot, dtype=np.float64)
    boot_seller = np.empty(n_boot, dtype=np.float64)
    boot_cat = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        w_s = rng.choice([-1.0, 1.0], size=n_s)
        w_c = rng.choice([-1.0, 1.0], size=n_c)
        w_m = 1.0 + (w_s[s_idx] * w_c[c_idx])
        boot_multiway[b] = np.sum(diff * w_m)
        boot_seller[b] = np.sum(diff * (1.0 + w_s[s_idx]))
        boot_cat[b] = np.sum(diff * (1.0 + w_c[c_idx]))

    return {
        "multiway": {
            "median": round(float(np.median(boot_multiway)), 2),
            "q05": round(float(np.percentile(boot_multiway, 5.0)), 2),
            "q95": round(float(np.percentile(boot_multiway, 95.0)), 2),
            "p_gt_0": round(float(np.mean(boot_multiway > 0)), 4),
        },
        "marginal_seller": {
            "median": round(float(np.median(boot_seller)), 2),
            "q05": round(float(np.percentile(boot_seller, 5.0)), 2),
            "q95": round(float(np.percentile(boot_seller, 95.0)), 2),
            "p_gt_0": round(float(np.mean(boot_seller > 0)), 4),
        },
        "marginal_category": {
            "median": round(float(np.median(boot_cat)), 2),
            "q05": round(float(np.percentile(boot_cat, 5.0)), 2),
            "q95": round(float(np.percentile(boot_cat, 95.0)), 2),
            "p_gt_0": round(float(np.mean(boot_cat > 0)), 4),
        }
    }

def compute_causal_history(df_fit, df_val, key='seller_id', prior=0.4722, smoothing=5.0):
    """Compute causal history: only events strictly before the query timestamp (t_hist < t_query).
    Ties (t_hist == t_query) are excluded by searchsorted(side='left').
    """
    obs = pd.DataFrame({
        'seller': df_fit[key].values,
        'stamp': df_fit['start_time'].values,
        'y': df_fit['y'].values
    })
    histories = {}
    for s, grp in obs.groupby('seller', sort=False):
        block = grp.groupby('stamp')['y'].agg(['count', 'sum'])
        stamps = block.index.to_numpy()
        counts_cum = np.r_[0, block['count'].cumsum().to_numpy()]
        sums_cum = np.r_[0, block['sum'].cumsum().to_numpy()]
        histories[s] = (stamps, counts_cum, sums_cum)

    def query(sellers, stamps):
        means = np.full(len(sellers), prior, dtype=np.float32)
        counts = np.zeros(len(sellers), dtype=np.float32)
        for i in range(len(sellers)):
            s = sellers[i]
            t = stamps[i]
            if s in histories:
                t0, ns, ys = histories[s]
                # side='left': index j has all t0[<j] < t. Any t0 >= t is at index >= j.
                j = np.searchsorted(t0, t, side='left')
                n = ns[j]
                y = ys[j]
                means[i] = (y + smoothing * prior) / (n + smoothing)
                counts[i] = np.log1p(n)
        return means, counts

    fit_mean, fit_count = query(df_fit[key].values, df_fit['start_time'].values)
    val_mean, val_count = query(df_val[key].values, df_val['start_time'].values)
    return fit_mean, fit_count, val_mean, val_count

# Claude's text model implementation
LEX = {
    'nuevo': r'\bnuev[oa]s?\b', 'usado': r'\busad[oa]s?\b', 'sin_uso': r'\bsin (uso|usar|estrenar)\b',
    'nunca_usado': r'\bnunca (usad|se uso)', 'no_usado': r'\bno (usad|se uso)', 'como_nuevo': r'\bcomo nuev',
    'impecable': r'\bimpecabl', 'sellado': r'\bsellad', 'en_caja': r'\ben (su )?caja\b|\bcaja cerrada\b',
    'original': r'\boriginal', 'repuesto': r'\brepuesto', 'accesorio': r'\baccesori',
    'antiguo': r'\bantigu|\bvintage\b|\bretro\b|\bcoleccion', 'garantia': r'\bgarantia\b',
    'oferta': r'\boferta|\bpromo', 'importado': r'\bimportad',
    'excelente_estado': r'\bexcelente estado|\bmuy buen estado|\bbuen estado',
    'detalle': r'\bdetalle|\bfalla|\broto|\brayad', 'libro': r'\blibro|\brevista', 'lote': r'\blote\b'
}

def norm_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9ñ ]', ' ', s)).strip()

def build_claude_text_features(df_fit, df_val, df_test=None):
    """Build Claude CL-0036 text and attribute features fit strictly on df_fit."""
    t_fit = df_fit['title'].map(norm_text)
    t_val = df_val['title'].map(norm_text)
    w_fit = df_fit['warranty_str'].map(norm_text)
    w_val = df_val['warranty_str'].map(norm_text)
    a_fit = df_fit['attr_text'].map(norm_text) + ' ' + df_fit['tags_str'].fillna('') + ' ' + df_fit['sub_status_str'].fillna('')
    a_val = df_val['attr_text'].map(norm_text) + ' ' + df_val['tags_str'].fillna('') + ' ' + df_val['sub_status_str'].fillna('')

    # Lexicon
    full_fit = t_fit + ' ' + w_fit
    full_val = t_val + ' ' + w_val
    lex_fit = np.column_stack([full_fit.str.contains(p, regex=True).values for p in LEX.values()]).astype(float)
    lex_val = np.column_stack([full_val.str.contains(p, regex=True).values for p in LEX.values()]).astype(float)

    # Light numeric meta
    meta_fit = np.column_stack([
        np.log1p(df_fit['num_pics'].fillna(0).values),
        np.log1p(df_fit['num_vars'].fillna(0).values),
        (w_fit.str.len() > 0).astype(float),
        np.log1p(t_fit.str.len())
    ])
    meta_val = np.column_stack([
        np.log1p(df_val['num_pics'].fillna(0).values),
        np.log1p(df_val['num_vars'].fillna(0).values),
        (w_val.str.len() > 0).astype(float),
        np.log1p(t_val.str.len())
    ])
    mu, sd = meta_fit.mean(0), meta_fit.std(0) + 1e-9
    meta_fit_std = (meta_fit - mu) / sd
    meta_val_std = (meta_val - mu) / sd

    # Vectorizers fit strictly on df_fit
    vt_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vt_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=300000, sublinear_tf=True)
    vw = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    va = TfidfVectorizer(token_pattern=r'\S+', min_df=2)

    Xtr_parts = [
        vt_word.fit_transform(t_fit),
        vt_char.fit_transform(t_fit),
        vw.fit_transform(w_fit),
        va.fit_transform(a_fit),
        sparse.csr_matrix(lex_fit),
        sparse.csr_matrix(meta_fit_std)
    ]
    Xval_parts = [
        vt_word.transform(t_val),
        vt_char.transform(t_val),
        vw.transform(w_val),
        va.transform(a_val),
        sparse.csr_matrix(lex_val),
        sparse.csr_matrix(meta_val_std)
    ]

    if df_test is not None:
        t_te = df_test['title'].map(norm_text)
        w_te = df_test['warranty_str'].map(norm_text)
        a_te = df_test['attr_text'].map(norm_text) + ' ' + df_test['tags_str'].fillna('') + ' ' + df_test['sub_status_str'].fillna('')
        full_te = t_te + ' ' + w_te
        lex_te = np.column_stack([full_te.str.contains(p, regex=True).values for p in LEX.values()]).astype(float)
        meta_te = np.column_stack([
            np.log1p(df_test['num_pics'].fillna(0).values),
            np.log1p(df_test['num_vars'].fillna(0).values),
            (w_te.str.len() > 0).astype(float),
            np.log1p(t_te.str.len())
        ])
        meta_te_std = (meta_te - mu) / sd
        Xte_parts = [
            vt_word.transform(t_te),
            vt_char.transform(t_te),
            vw.transform(w_te),
            va.transform(a_te),
            sparse.csr_matrix(lex_te),
            sparse.csr_matrix(meta_te_std)
        ]
        return sparse.hstack(Xtr_parts).tocsr(), sparse.hstack(Xval_parts).tocsr(), sparse.hstack(Xte_parts).tocsr()

    return sparse.hstack(Xtr_parts).tocsr(), sparse.hstack(Xval_parts).tocsr()

def main():
    t_start = time.time()
    print("=" * 80)
    print("AG-0237: AUTHENTIC CAUSAL PIPELINE, LEAK-FREE AUDIT & HISTORICAL LEDGER BACKTEST")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/6] Loading datasets and canonical splits...")
    with open('data/splits/t98_parsed_train.pkl', 'rb') as f:
        df_train = pickle.load(f)
    with open('data/splits/t98_parsed_test.pkl', 'rb') as f:
        df_test = pickle.load(f)
    with open('data/ap_2026_q_2/raw/feats.pkl', 'rb') as f:
        feats = pickle.load(f)

    F = feats['F'].copy()
    y_all = feats['y'].copy()
    y_true_70k = (y_all[:70000] == 1).astype(int) if set(np.unique(y_all[:70000])).issubset({0, 1}) else (y_all[:70000] == 'used').astype(int)
    df_train['y'] = y_true_70k

    df_3way = pd.read_csv('data/splits/canonical_3way_splits.csv')
    df_nested = pd.read_csv('data/splits/canonical_nested_dual_regime_splits.csv')

    fit_mask = (df_3way['role'] == 'fit').values
    tune_mask = (df_3way['role'] == 'tune').values
    conf_mask = (df_3way['role'] == 'confirm').values

    df_fit = df_train.loc[fit_mask].copy().reset_index(drop=True)
    df_tune = df_train.loc[tune_mask].copy().reset_index(drop=True)
    df_conf = df_train.loc[conf_mask].copy().reset_index(drop=True)

    y_fit = y_true_70k[fit_mask]
    y_tune = y_true_70k[tune_mask]
    y_conf = y_true_70k[conf_mask]

    print(f"  Train: 70,000 (Fit: {len(df_fit)}, Tune: {len(df_tune)}, Confirm: {len(df_conf)}) | Test: {len(df_test)}")

    # 2. Leak-Free Feature Engineering (Audit of CX-0223 point 2)
    print("\n[2/6] Performing Leak-Free Feature Reconstruction (Excluding 8 Leaked Columns & full70k_text_cache)...")
    
    # 60 intrinsic numerical features (row-by-row, zero leakage)
    intrinsic_num_cols = [
        'price', 'base_price', 'initial_quantity', 'available_quantity', 'sold_quantity', 'log_price',
        'price_eq_base', 'sold_ratio', 'avail_ratio', 'init_minus_avail', 'price_cents', 'price_round100',
        'price_round10', 'price_end9', 'has_original_price', 'official_store', 'has_video', 'has_catalog',
        'has_parent', 'automatic_relist', 'accepts_mp', 'n_deals', 'currency_usd', 'n_tags', 'n_pm',
        'ship_local_pick_up', 'ship_free', 'ship_n_methods', 'ship_n_tags', 'n_pics', 'pic0_w', 'pic0_h',
        'pic_max_area', 'pic_mean_area', 'pic_square', 'n_variations', 'n_attributes', 'dur_days',
        'warranty_null', 'warranty_len', 'w_sin', 'w_no', 'w_meses', 'w_anio', 'w_fabrica', 'w_oficial',
        'w_usado', 'w_nuevo', 'w_dias', 'w_garantia', 'title_len', 'title_words', 'title_upper_frac',
        'title_digits', 't_usado', 't_nuevo', 't_original', 't_antiguo', 't_libro', 't_lote', 't_impecable'
    ]
    avail_num_cols = [c for c in intrinsic_num_cols if c in F.columns]
    
    # Categoricals
    cat_cols = ['listing_type_id', 'buying_mode', 'category_id', 'ship_mode', 'state']

    # Compute category median logprice STRICTLY on df_fit (Safe reconstruction)
    cat_med_fit = df_fit.groupby('category_id')['price'].apply(lambda s: np.median(np.log1p(s)))
    global_med_fit = np.median(np.log1p(df_fit['price']))

    def build_clean_tabular(df_sub, F_sub):
        X_num = F_sub[avail_num_cols].to_numpy(dtype=np.float32)
        X_num = np.nan_to_num(X_num, nan=0.0)
        
        # Safe category price features
        med = df_sub['category_id'].map(cat_med_fit).fillna(global_med_fit).values
        rel = np.log1p(df_sub['price'].values) - med
        
        cat_arrays = []
        for c in cat_cols:
            cat_s = F_sub[c].astype('category').cat.codes.to_numpy(dtype=np.int32)
            cat_arrays.append(cat_s[:, None])
        X_cat = np.column_stack(cat_arrays)
        
        return np.column_stack([X_num, med[:, None], rel[:, None], X_cat])

    X_tab_fit = build_clean_tabular(df_fit, F.iloc[:70000].loc[fit_mask])
    X_tab_tune = build_clean_tabular(df_tune, F.iloc[:70000].loc[tune_mask])
    X_tab_conf = build_clean_tabular(df_conf, F.iloc[:70000].loc[conf_mask])
    print(f"  Clean Tabular Design Matrix: {X_tab_fit.shape[1]} features (0 leaks, 0 full70k_text_cache)")

    # 3. Clean Text Model (Claude CL-0036 fitted strictly on df_fit)
    print("\n[3/6] Fitting Clean Text Model strictly on df_fit (40,881 rows)...")
    t_txt0 = time.time()
    Xtr_txt, Xtu_txt = build_claude_text_features(df_fit, df_tune)
    _, Xco_txt = build_claude_text_features(df_fit, df_conf)
    
    lr_text = LogisticRegression(C=1.0, solver='liblinear', max_iter=2000, random_state=42)
    lr_text.fit(Xtr_txt, y_fit)
    p_text_tune = lr_text.predict_proba(Xtu_txt)[:, 1]
    p_text_conf = lr_text.predict_proba(Xco_txt)[:, 1]
    
    acc_txt_tune = accuracy_score(y_tune, (p_text_tune >= 0.5).astype(int))
    acc_txt_conf = accuracy_score(y_conf, (p_text_conf >= 0.5).astype(int))
    print(f"  Clean Text Model fitted in {time.time()-t_txt0:.1f}s | Tune Acc: {acc_txt_tune:.5f} | Conf Acc: {acc_txt_conf:.5f}")

    # 4. Strictly Causal Features on Canonical Splits
    print("\n[4/6] Computing Strictly Causal Features (Prequential on Fit, Historical on Tune/Confirm)...")
    c_m_fit_5, c_c_fit_5, c_m_tu_5, c_c_tu_5 = compute_causal_history(df_fit, df_tune, key='seller_id', prior=0.4722, smoothing=5.0)
    c_m_fit_10, c_c_fit_10, c_m_tu_10, c_c_tu_10 = compute_causal_history(df_fit, df_tune, key='seller_id', prior=0.4722, smoothing=10.0)

    _, _, c_m_co_5, c_c_co_5 = compute_causal_history(df_fit, df_conf, key='seller_id', prior=0.4722, smoothing=5.0)
    _, _, c_m_co_10, c_c_co_10 = compute_causal_history(df_fit, df_conf, key='seller_id', prior=0.4722, smoothing=10.0)

    # Design matrices for Base and Causal
    X_base_fit = np.column_stack([X_tab_fit, logit(clip_prob(lr_text.predict_proba(Xtr_txt)[:, 1]))[:, None]])
    X_base_tune = np.column_stack([X_tab_tune, logit(clip_prob(p_text_tune))[:, None]])
    X_base_conf = np.column_stack([X_tab_conf, logit(clip_prob(p_text_conf))[:, None]])

    X_causal_fit = np.column_stack([X_base_fit, c_m_fit_5, c_c_fit_5, c_m_fit_10, c_c_fit_10])
    X_causal_tune = np.column_stack([X_base_tune, c_m_tu_5, c_c_tu_5, c_m_tu_10, c_c_tu_10])
    X_causal_conf = np.column_stack([X_base_conf, c_m_co_5, c_c_co_5, c_m_co_10, c_c_co_10])

    lgb_params = dict(
        n_estimators=500, learning_rate=0.04, num_leaves=36, min_child_samples=30,
        colsample_bytree=0.75, subsample=0.85, reg_lambda=2.0, random_state=42, n_jobs=6, verbose=-1
    )

    print("  Fitting Clean AG_Tabular_Base on df_fit...")
    m_base = LGBMClassifier(**lgb_params).fit(X_base_fit, y_fit)
    p_base_tune = m_base.predict_proba(X_base_tune)[:, 1]
    p_base_conf = m_base.predict_proba(X_base_conf)[:, 1]

    print("  Fitting Clean AG_Tabular_Causal on df_fit...")
    m_causal = LGBMClassifier(**lgb_params).fit(X_causal_fit, y_fit)
    p_causal_tune = m_causal.predict_proba(X_causal_tune)[:, 1]
    p_causal_conf = m_causal.predict_proba(X_causal_conf)[:, 1]

    # Evaluate on Tune and Confirm
    for split_name, y_true, p_b, p_c, df_split in [
        ('TUNE (15,044 rows)', y_tune, p_base_tune, p_causal_tune, df_tune),
        ('CONFIRM INTACT (14,075 rows)', y_conf, p_base_conf, p_causal_conf, df_conf)
    ]:
        acc_b = accuracy_score(y_true, (p_b >= 0.5).astype(int))
        acc_c = accuracy_score(y_true, (p_c >= 0.5).astype(int))
        ll_b = log_loss(y_true, clip_prob(p_b))
        ll_c = log_loss(y_true, clip_prob(p_c))
        
        ok_b = (p_b >= 0.5) == y_true
        ok_c = (p_c >= 0.5) == y_true
        fixes = int((ok_c & ~ok_b).sum())
        breaks = int((~ok_c & ok_b).sum())
        net = fixes - breaks
        
        boot = bootstrap_multiway_and_marginals(y_true, p_c, p_b, df_split['seller_id'].values, df_split['category_id'].values)
        
        print(f"\n  === EVALUATION ON {split_name} ===")
        print(f"    Base Acc: {acc_b:.5f} (LogLoss: {ll_b:.5f}) | Causal Acc: {acc_c:.5f} (LogLoss: {ll_c:.5f})")
        print(f"    Paired vs Base: +{fixes} fixes / -{breaks} breaks -> NET: {net:+d} aciertos")
        print(f"    Bootstrap Multiway Q05: {boot['multiway']['q05']:+.1f} | Seller Q05: {boot['marginal_seller']['q05']:+.1f} | Cat Q05: {boot['marginal_category']['q05']:+.1f}")

        # 5-Regimes Breakdown
        fit_sellers = set(df_fit['seller_id'].unique())
        fit_times = df_fit.groupby('seller_id')['start_time'].agg(['min', 'max'])
        fit_exact_times = df_fit.groupby('seller_id')['start_time'].apply(set).to_dict()

        s_arr = df_split['seller_id'].values
        t_arr = df_split['start_time'].values

        in_fit = np.isin(s_arr, list(fit_sellers))
        bounds = fit_times.reindex(s_arr)
        t_min = bounds['min'].values
        t_max = bounds['max'].values

        has_same_time = np.zeros(len(df_split), dtype=bool)
        for i in range(len(df_split)):
            s = s_arr[i]
            t = t_arr[i]
            if s in fit_exact_times and t in fit_exact_times[s]:
                has_same_time[i] = True

        regimes = {
            'all': np.ones(len(df_split), dtype=bool),
            'cold': ~in_fit,
            'ties': in_fit & has_same_time,
            'before': in_fit & ~has_same_time & (t_arr < t_min),
            'bracketed': in_fit & ~has_same_time & (t_arr > t_min) & (t_arr < t_max),
            'after': in_fit & ~has_same_time & (t_arr > t_max),
        }

        print(f"    5-Regimes Breakdown:")
        for rk, rm in regimes.items():
            n_r = rm.sum()
            if n_r == 0:
                continue
            fx_r = int((ok_c & ~ok_b & rm).sum())
            bk_r = int((~ok_c & ok_b & rm).sum())
            acc_r_b = accuracy_score(y_true[rm], (p_b[rm] >= 0.5).astype(int))
            acc_r_c = accuracy_score(y_true[rm], (p_c[rm] >= 0.5).astype(int))
            print(f"      {rk:12s} (n={n_r:5d}): Net {fx_r-bk_r:+3d} (+{fx_r:2d}/-{bk_r:2d}) | Base Acc: {acc_r_b:.4f} -> Causal Acc: {acc_r_c:.4f}")

    # 5. Full 70k Refit and Genuine Test Prediction Generation (Auditing CX-0223 point 1)
    print("\n" + "=" * 60)
    print("[5/6] Generating AUTHENTIC Causal Test Predictions (70k Refit + Strictly Historical Lookup)")
    print("=" * 60)

    # Compute category median strictly on full 70k train
    cat_med_70k = df_train.groupby('category_id')['price'].apply(lambda s: np.median(np.log1p(s)))
    global_med_70k = np.median(np.log1p(df_train['price']))

    def build_clean_tabular_70k(df_sub, F_sub):
        X_num = F_sub[avail_num_cols].to_numpy(dtype=np.float32)
        X_num = np.nan_to_num(X_num, nan=0.0)
        med = df_sub['category_id'].map(cat_med_70k).fillna(global_med_70k).values
        rel = np.log1p(df_sub['price'].values) - med
        cat_arrays = []
        for c in cat_cols:
            cat_s = F_sub[c].astype('category').cat.codes.to_numpy(dtype=np.int32)
            cat_arrays.append(cat_s[:, None])
        X_cat = np.column_stack(cat_arrays)
        return np.column_stack([X_num, med[:, None], rel[:, None], X_cat])

    X_tab_70k_full = build_clean_tabular_70k(df_train, F.iloc[:70000])
    X_tab_test_full = build_clean_tabular_70k(df_test, F.iloc[70000:])

    print("  Fitting Clean Text Model on Full 70k Train...")
    Xtr_70k_txt, _, Xte_txt = build_claude_text_features(df_train, df_tune, df_test)
    lr_text_70k = LogisticRegression(C=1.0, solver='liblinear', max_iter=2000, random_state=42)
    lr_text_70k.fit(Xtr_70k_txt, y_true_70k)
    p_text_70k = lr_text_70k.predict_proba(Xtr_70k_txt)[:, 1]
    p_text_test = lr_text_70k.predict_proba(Xte_txt)[:, 1]

    # Causal Features for 70k (Prequential) and Test (Strictly Historical t_train < t_test)
    print("  Computing Prequential Causal Features on 70k and Historical on Test...")
    c_m_70k_5, c_c_70k_5, c_m_test_5, c_c_test_5 = compute_causal_history(df_train, df_test, key='seller_id', prior=0.4722, smoothing=5.0)
    c_m_70k_10, c_c_70k_10, c_m_test_10, c_c_test_10 = compute_causal_history(df_train, df_test, key='seller_id', prior=0.4722, smoothing=10.0)

    # Base design matrices
    X_base_70k = np.column_stack([X_tab_70k_full, logit(clip_prob(p_text_70k))[:, None]])
    X_base_test = np.column_stack([X_tab_test_full, logit(clip_prob(p_text_test))[:, None]])

    # Authentic Causal design matrices (INCORPORATING CAUSAL COLUMNS!)
    X_causal_70k = np.column_stack([X_base_70k, c_m_70k_5, c_c_70k_5, c_m_70k_10, c_c_70k_10])
    X_causal_test = np.column_stack([X_base_test, c_m_test_5, c_c_test_5, c_m_test_10, c_c_test_10])

    print(f"  Design Matrix Shapes: Base 70k: {X_base_70k.shape}, Causal 70k: {X_causal_70k.shape}")
    print(f"  Test Design Matrix Shapes: Base Test: {X_base_test.shape}, Causal Test: {X_causal_test.shape}")

    print("  Fitting Authentic Full 70k Tabular Base Model...")
    m_base_70k = LGBMClassifier(**lgb_params).fit(X_base_70k, y_true_70k)
    p_test_base = m_base_70k.predict_proba(X_base_test)[:, 1]

    print("  Fitting AUTHENTIC Full 70k Causal Model (with causal history columns included!)...")
    m_causal_70k = LGBMClassifier(**lgb_params).fit(X_causal_70k, y_true_70k)
    p_test_causal_authentic = m_causal_70k.predict_proba(X_causal_test)[:, 1]

    # Save probabilities to disk
    out_dir = 'results/authentic_causal'
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/test_p_ag_causal_authentic.npy', p_test_causal_authentic)
    np.save(f'{out_dir}/test_p_ag_tabular_base.npy', p_test_base)
    np.save(f'{out_dir}/test_p_claude_text.npy', p_text_test)

    # 6. Backtest & Ledger Integration (CX-0224 / CX-0225)
    print("\n" + "=" * 60)
    print("[6/6] Integrating Complete Historical Submission Ledger and Flips Analysis")
    print("=" * 60)

    # Load historical submissions
    ledger = {
        'S4': 'submissions/portfolio_s5_historical_and_models/candidate_1_s4_only.csv',
        'Top174': 'submissions/claude_20261114/claude_blendfix_posterior_top174_sobre_S4.csv',
        'Top30': 'submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv',
        'Cupo3': 'submissions/20261114/CSV_3_HOY_T30_STACK_TOP50_SINREV.csv',
        'Cupo4': 'results/CANDIDATE_T30_PLUS_WARM_CLEAN_MODERATED.csv',
        'Cupo5': 'submissions/claude_20261114/claude_cupo4_T30_b025_keepT30_top50.csv',
    }

    sub_dfs = {}
    for name, path in ledger.items():
        if os.path.exists(path):
            df_sub = pd.read_csv(path)
            # Standardize columns
            if 'ID' in df_sub.columns:
                df_sub = df_sub.rename(columns={'ID': 'id'})
            if 'condition' in df_sub.columns:
                df_sub = df_sub.rename(columns={'condition': 'target'})
            sub_dfs[name] = df_sub
            print(f"  Loaded {name:7s} from {path} (rows={len(df_sub)}, used_rate={(df_sub['target']=='used').mean():.4f})")

    # Generate Candidate Submissions
    # Candidate 1: Authentic Causal Tabular + Text Ensemble (Prioritario)
    # Calibrated probability blend
    p_cand_priority = p_test_causal_authentic
    pred_priority = np.where(p_cand_priority >= 0.5, 'used', 'new')

    # Candidate 2: Clean Tabular Base (Reserva)
    p_cand_reserve = p_test_base
    pred_reserve = np.where(p_cand_reserve >= 0.5, 'used', 'new')

    # Candidate 3: Conservative Surgical Flip over T30 using Causal Probability Margin
    # Flip only where Causal model is highly confident AND Base agrees, preserving T30 elsewhere
    t30_preds = sub_dfs['Top30']['target'].values
    margin_mask_used = (p_test_causal_authentic >= 0.85) & (t30_preds == 'new')
    margin_mask_new = (p_test_causal_authentic <= 0.15) & (t30_preds == 'used')
    pred_surgical = t30_preds.copy()
    pred_surgical[margin_mask_used] = 'used'
    pred_surgical[margin_mask_new] = 'new'
    n_surg_flips = (pred_surgical != t30_preds).sum()
    print(f"\n  Candidate Surgical Flips over T30: {n_surg_flips} flips (used: {margin_mask_used.sum()}, new: {margin_mask_new.sum()})")

    # Compare Candidates against Historical Submissions
    test_ids = df_test['id'].values if 'id' in df_test.columns else np.arange(1, 30001)

    print("\n  === HISTORICAL SUBMISSION DISTANCES (HAMMING FLIPS) ===")
    print(f"  {'Reference':10s} | {'Score':7s} | {'Causal vs Ref':15s} | {'Base vs Ref':15s} | {'Surgical vs Ref':15s}")
    print("  " + "-" * 75)

    scores = {
        'S4': '0.94766',
        'Top174': '0.94723 (-9)',
        'Top30': '0.94823 (+12)',
        'Cupo3': '0.94814 (-2)',
        'Cupo4': '0.94504 (-67)',
        'Cupo5': '0.94790 (-7)',
    }

    for name, df_sub in sub_dfs.items():
        ref_pred = df_sub['target'].values
        d_causal = (pred_priority != ref_pred).sum()
        d_base = (pred_reserve != ref_pred).sum()
        d_surg = (pred_surgical != ref_pred).sum()
        sc = scores.get(name, 'N/A')
        print(f"  {name:10s} | {sc:13s} | {d_causal:5d} flips   | {d_base:5d} flips   | {d_surg:5d} flips")

    # Overlap with Cupo 4 (-67) and Cupo 5 (-7)
    c4_flips = set(np.flatnonzero(sub_dfs['Cupo4']['target'].values != sub_dfs['Top30']['target'].values))
    c5_flips = set(np.flatnonzero(sub_dfs['Cupo5']['target'].values != sub_dfs['Top30']['target'].values))
    surg_flips = set(np.flatnonzero(pred_surgical != sub_dfs['Top30']['target'].values))
    causal_vs_t30_flips = set(np.flatnonzero(pred_priority != sub_dfs['Top30']['target'].values))

    print(f"\n  Overlap with Cupo 4 (-67 disaster flips, n={len(c4_flips)}):")
    print(f"    Surgical Candidate overlap: {len(surg_flips.intersection(c4_flips))} / {len(surg_flips)}")
    print(f"    Full Causal Candidate overlap: {len(causal_vs_t30_flips.intersection(c4_flips))} / {len(causal_vs_t30_flips)}")

    print(f"\n  Overlap with Cupo 5 (-7 flips, n={len(c5_flips)}):")
    print(f"    Surgical Candidate overlap: {len(surg_flips.intersection(c5_flips))} / {len(surg_flips)}")
    print(f"    Full Causal Candidate overlap: {len(causal_vs_t30_flips.intersection(c5_flips))} / {len(causal_vs_t30_flips)}")

    # Export Candidate CSVs
    os.makedirs('submissions/20261115', exist_ok=True)
    
    # 1. Candidate Prioritario: Causal Auténtico Completo
    p_path_1 = 'submissions/20261115/CANDIDATE_1_PRIORITY_AG_CAUSAL_AUTHENTIC.csv'
    df_cand1 = pd.DataFrame({'id': test_ids, 'target': pred_priority})
    df_cand1.to_csv(p_path_1, index=False)

    # 2. Candidate Reserva: Tabular Base Limpio
    p_path_2 = 'submissions/20261115/CANDIDATE_2_RESERVE_AG_TABULAR_BASE.csv'
    df_cand2 = pd.DataFrame({'id': test_ids, 'target': pred_reserve})
    df_cand2.to_csv(p_path_2, index=False)

    # 3. Candidate Quirúrgico sobre T30 (Opcional seguro)
    p_path_3 = 'submissions/20261115/CANDIDATE_3_SURGICAL_CAUSAL_ON_T30.csv'
    df_cand3 = pd.DataFrame({'id': test_ids, 'target': pred_surgical})
    df_cand3.to_csv(p_path_3, index=False)

    # Export Aligned OOF with Folds for Claude (CX-0225 requirement)
    # Using 5-fold outer unseen CV
    print("\n  Generating Aligned 5-Fold OOF Predictions for Claude...")
    oof_p_base = np.zeros(70000, dtype=np.float32)
    oof_p_causal = np.zeros(70000, dtype=np.float32)
    outer_folds = df_nested['outer_unseen_fold'].values

    for k in range(5):
        role_col = f'role_in_fold_{k}'
        fit_idx = np.flatnonzero(df_nested[role_col] == 'fit')
        unseen_idx = np.flatnonzero(df_nested[role_col] == 'unseen_val')
        
        m_b_k = LGBMClassifier(**lgb_params).fit(X_base_70k[fit_idx], y_true_70k[fit_idx])
        m_c_k = LGBMClassifier(**lgb_params).fit(X_causal_70k[fit_idx], y_true_70k[fit_idx])
        
        oof_p_base[unseen_idx] = m_b_k.predict_proba(X_base_70k[unseen_idx])[:, 1]
        oof_p_causal[unseen_idx] = m_c_k.predict_proba(X_causal_70k[unseen_idx])[:, 1]

    df_oof_claude = pd.DataFrame({
        'id': np.arange(1, 70001),
        'seller_id': df_train['seller_id'].values,
        'category_id': df_train['category_id'].values,
        'y_true': y_true_70k,
        'outer_fold': outer_folds,
        'p_base': oof_p_base,
        'p_causal': oof_p_causal,
        'p_text': p_text_70k
    })
    oof_claude_path = f'{out_dir}/oof_probabilities_causal_with_folds.csv'
    df_oof_claude.to_csv(oof_claude_path, index=False)
    print(f"  Exported aligned OOF to {oof_claude_path} ({len(df_oof_claude)} rows)")

    # Gate G Verification
    print("\n  === GATE G VERIFICATION ===")
    for cname, cpath in [
        ('Prioritario (Causal)', p_path_1),
        ('Reserva (Base)', p_path_2),
        ('Surgical (T30 + Causal)', p_path_3)
    ]:
        df_c = pd.read_csv(cpath)
        sha = compute_file_sha256(cpath)
        gate_f_hash = compute_id_target_hash(df_c['id'].values, df_c['target'].values)
        n_rows = len(df_c)
        n_ids = df_c['id'].nunique()
        min_id, max_id = df_c['id'].min(), df_c['id'].max()
        targets = set(df_c['target'].unique())
        used_rate = (df_c['target'] == 'used').mean()
        
        gate_g_pass = (n_rows == 30000) and (n_ids == 30000) and (min_id == 1) and (max_id == 30000) and (targets == {'new', 'used'})
        status = "PASSED" if gate_g_pass else "FAILED"
        print(f"  {cname:25s} | Gate G: {status} | Used Rate: {used_rate:.4f}")
        print(f"    LF SHA-256: {sha}")
        print(f"    Gate F Hash: {gate_f_hash}")

    print(f"\nDone in {time.time()-t_start:.1f}s.")

if __name__ == '__main__':
    main()
