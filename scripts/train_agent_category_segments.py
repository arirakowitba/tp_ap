"""Agent Worker: Category-Segmented Domain Specialists with Transductive Probe Augmentation.

Mission:
1. Outer StratifiedGroupKFold(5) grouped by seller_id (strictly zero seller leakage).
2. Purely item-intrinsic features (ZERO seller identity / ZERO seller target encodings).
3. Transductive test manifold modeling:
   - 27,723 unanimous INVARIANT_ALL probe consensus items as anchor pseudo-labels (sample weight = 0.40).
   - 70,000 train items (sample weight = 1.0).
   - In each outer fold, test anchors belonging to validation sellers are filtered out to guarantee zero seller overlap.
4. Category specialists on top volume & disagreement domains:
   - MLA3025_Libros (Books, Revistas, Comics, Ebooks, Novelas)
   - MLA1747_Repuestos_Autos (Auto & Moto Parts, Accesorios, Repuestos)
   - MLA1051_Celulares (Celulares, Smartphones, Fundas, Baterías)
   - MLA1000_Electrónica (Electrónica, Audio, Video, Vinilos, CDs)
   - Antiques_Home (Antigüedades, Coleccionables, Vintage, Hogar)
   - General_Other (Global fallback)
5. Sequential fold processing with explicit gc.collect() to maintain RAM < 4 GB.
6. Outputs:
   - results/agent_category_segments_20261110/oof_category_segments.csv
   - results/agent_category_segments_20261110/test_probabilities_category_segments.csv
   - results/agent_category_segments_20261110/summary_report.json
"""

import os
import sys
import gc
import json
import time
import math
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "agent_category_segments_20261110"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Lexical Cue Definitions (Item-Intrinsic)
# ---------------------------------------------------------
CUES = [
    ("cue_0km", re.compile(r"\b0\s*km\b|\bcero\s*km\b|\bsin\s+rodar\b|\b0km\b", re.I)),
    ("cue_caja_sellada", re.compile(r"\bcaja\s+sellada\b|\bcerrad[oa]\s+de\s+f[aá]brica\b|\bprecintad[oa]\b|\ben\s+blister\b|\bnuevo\s+sellado\b|\bnuevo\s+cerrado\b", re.I)),
    ("cue_nuevo", re.compile(r"\bnuev[oa]s?\b|\ba\s+estrenar\b", re.I)),
    ("cue_usado", re.compile(r"\b(usad[oa]s?|segunda\s+mano)\b", re.I)),
    ("cue_repuesto", re.compile(r"\bpara\s+repuestos?\b|\bpara\s+desarme\b|\bno\s+funciona\b|\brot[oa]s?\b|\bpara\s+reparar\b|\bcon\s+falla\b", re.I)),
    ("cue_poco_uso", re.compile(r"\b(un|1)\s+solo\s+uso\b|\bpoco\s+uso\b|\bpoqu[ií]simo\s+uso\b", re.I)),
    ("cue_como_nuevo", re.compile(r"\bcomo\s+nuev[oa]\b|\bigual\s+a\s+nuev[oa]\b|\bseminuevo\b", re.I)),
    ("cue_impecable", re.compile(r"\bimpecable\b|\bde\s+vitrina\b|\bexcelente\s+estado\b|\bmuy\s+buen\s+estado\b", re.I)),
    ("cue_antiguo", re.compile(r"\bantigu[oa]s?\b|\bde\s+[eé]poca\b|\bvintage\b|\bcolecci[oó]n\b|\bretro\b", re.I)),
    ("cue_digital", re.compile(r"\b(libro\s+digital|ebook|pdf|descarga\s+digital)\b", re.I)),
    ("cue_detalles", re.compile(r"\bcon\s+detalles?\b|\bmarcas?\s+de\s+uso\b|\brayones?\b|\bdesgaste\b", re.I)),
    ("cue_sin_uso", re.compile(r"\bsin\s+uso\b|\bnunca\s+se\s+us[oó]\b|\bnueva\s+sin\s+uso\b", re.I)),
    ("cue_original", re.compile(r"\boriginal\b|\bleg[ií]tim[oa]\b", re.I)),
    ("cue_garantia_fabrica", re.compile(r"\bgarant[ií]a\s+de\s+f[aá]brica\b|\bgarant[ií]a\s+oficial\b", re.I)),
]

# ---------------------------------------------------------
# 2. Domain & Category Routing
# ---------------------------------------------------------
BOOK_CATS = {
    'MLA1227', 'MLA2044', 'MLA41287', 'MLA15328', 'MLA41269', 'MLA41064', 'MLA41185',
    'MLA1963', 'MLA41260', 'MLA41174', 'MLA41083', 'MLA1207', 'MLA40497', 'MLA5496',
    'MLA11456', 'MLA2042', 'MLA6009', 'MLA2039', 'MLA41173', 'MLA5500', 'MLA41112',
    'MLA40507', 'MLA7373', 'MLA40547', 'MLA7251', 'MLA40526', 'MLA41176', 'MLA41057',
    'MLA41192', 'MLA2070', 'MLA3025'
}
VEHICLE_CATS = {
    'MLA85960', 'MLA6651', 'MLA60611', 'MLA3936', 'MLA352293', 'MLA86029', 'MLA60671',
    'MLA373016', 'MLA1747', 'MLA1743', 'MLA2547', 'MLA40941', 'MLA23830', 'MLA61129',
    'MLA60655', 'MLA23832', 'MLA86111', 'MLA86110', 'MLA60678', 'MLA60613'
}
CELULARES_CATS = {
    'MLA1051', 'MLA1055', 'MLA3813', 'MLA5726', 'MLA3814', 'MLA13537', 'MLA5725'
}
ELECTRONICS_CATS = {
    'MLA2038', 'MLA15171', 'MLA15204', 'MLA2032', 'MLA3361', 'MLA15226', 'MLA1914',
    'MLA15197', 'MLA15210', 'MLA12812', 'MLA1000', 'MLA1648', 'MLA1144', 'MLA1002',
    'MLA1182', 'MLA1012', 'MLA1014', 'MLA431414'
}
ANTIQUES_CATS = {
    'MLA3530', 'MLA1383', 'MLA4335', 'MLA1902', 'MLA1635', 'MLA35923', 'MLA370638',
    'MLA1458', 'MLA119287', 'MLA26536', 'MLA1430', 'MLA1574', 'MLA1403', 'MLA1276',
    'MLA1368'
}

rx_book = re.compile(r'\b(libro|libros|novela|tomo|tomos|edici[oó]n|editorial|autor|poemas|cuentos|revista|revistas|fasc[ií]culo|fasc[ií]culos|ebook|pdf|enciclopedia|diccionario)\b', re.I)
rx_vehicle = re.compile(r'\b(repuesto|repuestos|amortiguador|amortiguadores|freno|frenos|filtro|filtros|buj[ií]a|buj[ií]as|carburador|motor|embrague|embragues|óptica|optica|paragolpe|paragolpes|fiat|volkswagen|ford|renault|peugeot|chevrolet|toyota|honda|corsa|gol|palio|clio|hilux|moto|motos|yamaha)\b', re.I)
rx_cel = re.compile(r'\b(celular|celulares|smartphone|samsung|galaxy|iphone|motorola|moto\s*g|xiaomi|funda|fundas|templado|vidrio\s*templado|bater[ií]a|cargador|m[oó]dulo|display)\b', re.I)
rx_elec = re.compile(r'\b(electr[oó]nica|audio|video|vinilo|vinilos|cd|cds|dvd|dvds|cassette|cassettes|auricular|auriculares|parlante|parlantes|televisor|tv|consola|playstation|ps3|ps4|xbox|nintendo|c[aá]mara|lente|flash|amplificador|guitarra|bajo|teclado)\b', re.I)
rx_antique = re.compile(r'\b(antigu[oa]s?|vintage|colecci[oó]n|coleccionable|retro|de\s+[eé]poca|remera|remeras|pantal[oó]n|zapatilla|zapatillas|vestido|camisa|reloj|vajilla|adorno)\b', re.I)

DOMAINS = [
    'MLA3025_Libros',
    'MLA1747_Repuestos_Autos',
    'MLA1051_Celulares',
    'MLA1000_Electrónica',
    'Antiques_Home',
    'General_Other'
]

def route_item(cat, title):
    cat_str = str(cat or '')
    if cat_str in BOOK_CATS: return 'MLA3025_Libros'
    if cat_str in VEHICLE_CATS: return 'MLA1747_Repuestos_Autos'
    if cat_str in CELULARES_CATS: return 'MLA1051_Celulares'
    if cat_str in ELECTRONICS_CATS: return 'MLA1000_Electrónica'
    if cat_str in ANTIQUES_CATS: return 'Antiques_Home'
    
    # Text fallback
    title_str = str(title or '')
    if rx_book.search(title_str): return 'MLA3025_Libros'
    if rx_vehicle.search(title_str): return 'MLA1747_Repuestos_Autos'
    if rx_cel.search(title_str): return 'MLA1051_Celulares'
    if rx_elec.search(title_str): return 'MLA1000_Electrónica'
    if rx_antique.search(title_str): return 'Antiques_Home'
    return 'General_Other'

# ---------------------------------------------------------
# 3. Tabular & Intrinsic Feature Extraction
# ---------------------------------------------------------
def extract_tabular_and_text(rows):
    """Extract item-intrinsic numeric features, lexical cues, and textual corpus."""
    tab_list = []
    corpus_list = []
    
    for r in rows:
        title = str(r.get('title') or '').strip()
        warranty = str(r.get('warranty') or '').strip()
        attrs = r.get('attributes') or []
        attr_vals = [str(a.get('value_name') or a.get('name') or '') for a in attrs if isinstance(a, dict)]
        attr_text = " ".join(attr_vals)
        tags = r.get('tags') or []
        tags_str = " ".join(tags) if isinstance(tags, list) else ""
        
        full_text = f"{title} {warranty} {attr_text}".strip()
        corpus_list.append(full_text)
        
        price = float(r.get('price') or 0.0)
        p_val = max(0.0, price) if (price is not None and np.isfinite(price)) else 0.0
        
        orig_p = float(r.get('original_price') or 0.0)
        orig_val = max(0.0, orig_p) if (orig_p is not None and np.isfinite(orig_p)) else 0.0
        has_orig = 1.0 if orig_val > 0.0 else 0.0
        ratio = (orig_val / (p_val + 1e-4)) if (has_orig and p_val > 0.0) else 1.0
        
        title_len = float(len(title))
        word_len = float(len(title.split()))
        caps_cnt = float(len(re.findall(r'[A-Z]', title)))
        digit_cnt = float(len(re.findall(r'\d', title)))
        
        shipping = r.get('shipping') or {}
        free_shipping = 1.0 if shipping.get('free_shipping') else 0.0
        local_pickup = 1.0 if shipping.get('local_pick_up') else 0.0
        mercadopago = 1.0 if r.get('accepts_mercadopago') else 0.0
        
        init_q = float(r.get('initial_quantity') or 1.0)
        sold_q = float(r.get('sold_quantity') or 0.0)
        avail_q = float(r.get('available_quantity') or 0.0)
        
        listing_type = str(r.get('listing_type_id') or '')
        is_gold = 1.0 if 'gold' in listing_type else 0.0
        is_bronze = 1.0 if 'bronze' in listing_type else 0.0
        
        tag_relist = 1.0 if 'free_relist' in tags_str else 0.0
        tag_dragged = 1.0 if 'dragged' in tags_str else 0.0
        tag_thumb = 1.0 if 'good_quality_thumbnail' in tags_str else 0.0
        
        row_num = [
            np.log1p(p_val),
            1.0 if p_val > 0.0 else 0.0,
            np.log1p(orig_val),
            has_orig,
            min(ratio, 10.0),
            title_len,
            word_len,
            caps_cnt / (title_len + 1.0),
            digit_cnt / (title_len + 1.0),
            np.log1p(max(0.0, init_q)),
            np.log1p(max(0.0, avail_q)),
            np.log1p(max(0.0, sold_q)),
            sold_q / (init_q + 1.0),
            mercadopago,
            free_shipping,
            local_pickup,
            float(len(attrs)),
            float(len(r.get('pictures') or [])),
            float(len(tags)),
            float(len(r.get('variations') or [])),
            1.0 if len(warranty) > 0 else 0.0,
            float(len(warranty)),
            is_gold,
            is_bronze,
            tag_relist,
            tag_dragged,
            tag_thumb
        ]
        
        # Append Lexical Cues
        for _, rx in CUES:
            row_num.append(1.0 if rx.search(title) else 0.0)
            
        tab_list.append(row_num)
        
    return np.array(tab_list, dtype=np.float32), corpus_list


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("AGENT: CATEGORY-SEGMENTED SPECIALISTS UNDER OUTER STRATIFIED GROUP K-FOLD", flush=True)
    print("=" * 80, flush=True)
    
    # ---------------------------------------------------------
    # 4. Load Data
    # ---------------------------------------------------------
    print("\n[1/5] Loading 70k Train and 30k Test datasets...", flush=True)
    with open(ROOT / 'data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
        train_rows = [json.loads(line) for line in f]
        
    with open(ROOT / 'data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
        test_rows = [json.loads(line) for line in f]
        
    n_train = len(train_rows)
    n_test = len(test_rows)
    print(f"Loaded {n_train} train and {n_test} test items.")
    
    y_train = np.array([1 if r['condition'] == 'used' else 0 for r in train_rows], dtype=np.int32)
    sellers_train = np.array([str(r.get('seller_id') or 'UNKNOWN') for r in train_rows])
    cats_train = np.array([str(r.get('category_id') or 'UNKNOWN') for r in train_rows])
    cats_test = np.array([str(r.get('category_id') or 'UNKNOWN') for r in test_rows])
    sellers_test = np.array([str(r.get('seller_id') or 'UNKNOWN') for r in test_rows])
    
    # Extract Tabular and Corpus
    print("\n[2/5] Extracting intrinsic tabular features and text corpora...", flush=True)
    X_train_tab, corpus_train = extract_tabular_and_text(train_rows)
    X_test_tab, corpus_test = extract_tabular_and_text(test_rows)
    
    # Assign Domains
    train_domains = np.array([route_item(c, r.get('title')) for c, r in zip(cats_train, train_rows)])
    test_domains = np.array([route_item(c, r.get('title')) for c, r in zip(cats_test, test_rows)])
    
    print("\nDomain Distribution across Train & Test:")
    for d in DOMAINS:
        tr_c = int((train_domains == d).sum())
        te_c = int((test_domains == d).sum())
        print(f"  {d:25s} | Train: {tr_c:5d} ({tr_c/n_train*100:5.1f}%) | Test: {te_c:5d} ({te_c/n_test*100:5.1f}%)")
        
    # ---------------------------------------------------------
    # 5. Load Master Confidences for INVARIANT_ALL Anchors
    # ---------------------------------------------------------
    print("\n[3/5] Loading 27,723 INVARIANT_ALL Probe Consensus Anchors...", flush=True)
    master_df = pd.read_csv(ROOT / 'results/master_item_confidences_30k_20261110.csv')
    inv_all_mask = (master_df['invariance_tier'] == 'INVARIANT_ALL').values
    inv_labels = np.where(master_df['pct_all_vectors_used'] >= 0.5, 1, 0)
    
    n_inv = int(inv_all_mask.sum())
    print(f"Total INVARIANT_ALL test items: {n_inv} / {n_test} ({n_inv/n_test*100:.1f}%)")
    
    # Pre-select indices of invariant items
    inv_indices = np.where(inv_all_mask)[0]
    
    # Free memory
    del train_rows, test_rows, master_df
    gc.collect()
    
    # ---------------------------------------------------------
    # 6. Outer StratifiedGroupKFold(5) by seller_id
    # ---------------------------------------------------------
    print("\n[4/5] Running 5-Fold Outer StratifiedGroupKFold (Zero Seller Overlap)...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(n_train, dtype=np.float32)
    test_fold_predictions = np.zeros((5, n_test), dtype=np.float32)
    fold_metrics = []
    
    # Common LightGBM params
    global_lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 95,
        'max_depth': 11,
        'learning_rate': 0.035,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_data_in_leaf': 20,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': 4
    }
    
    spec_lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 45,
        'max_depth': 8,
        'learning_rate': 0.04,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_data_in_leaf': 15,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': 4
    }
    
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(corpus_train, y_train, groups=sellers_train), 1):
        f_t0 = time.time()
        print(f"\n==================== FOLD {fold} / 5 ====================", flush=True)
        print(f"Train items: {len(tr_idx)} | Validation items: {len(val_idx)}", flush=True)
        
        # Verify Zero Seller Overlap
        val_sellers = set(sellers_train[val_idx])
        tr_sellers = set(sellers_train[tr_idx])
        assert len(val_sellers & tr_sellers) == 0, "ERROR: Seller leakage detected!"
        
        # Filter test anchors: EXCLUDE any anchor whose seller_id is in val_sellers
        anchor_mask = [idx for idx in inv_indices if sellers_test[idx] not in val_sellers]
        anchor_indices = np.array(anchor_mask, dtype=int)
        print(f"Eligible INVARIANT_ALL test anchors (zero val seller overlap): {len(anchor_indices)} items")
        
        # 1. Text representations (fitted strictly on outer fold train corpus)
        print("  Fitting TF-IDF (35k max features) + TruncatedSVD (75 components)...", flush=True)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=35000, sublinear_tf=True, min_df=2)
        X_tr_tfidf = tfidf.fit_transform([corpus_train[i] for i in tr_idx])
        X_val_tfidf = tfidf.transform([corpus_train[i] for i in val_idx])
        X_te_tfidf = tfidf.transform(corpus_test)
        X_anc_tfidf = X_te_tfidf[anchor_indices]
        
        svd = TruncatedSVD(n_components=75, random_state=42)
        X_tr_svd = svd.fit_transform(X_tr_tfidf)
        X_val_svd = svd.transform(X_val_tfidf)
        X_te_svd = svd.transform(X_te_tfidf)
        X_anc_svd = X_te_svd[anchor_indices]
        
        # 2. Intrinsic Relative Price to Category Median (strictly computed on fold train)
        tr_cats_fold = cats_train[tr_idx]
        tr_log_price_fold = X_train_tab[tr_idx, 0]
        
        # Calculate category median log-price on fold train
        cat_medians = {}
        for c, lp in zip(tr_cats_fold, tr_log_price_fold):
            if c not in cat_medians:
                cat_medians[c] = []
            cat_medians[c].append(lp)
        cat_median_map = {c: float(np.median(vals)) for c, vals in cat_medians.items()}
        global_med_price = float(np.median(tr_log_price_fold))
        
        rel_price_tr = np.array([lp - cat_median_map.get(c, global_med_price) for c, lp in zip(tr_cats_fold, tr_log_price_fold)], dtype=np.float32)
        rel_price_val = np.array([lp - cat_median_map.get(c, global_med_price) for c, lp in zip(cats_train[val_idx], X_train_tab[val_idx, 0])], dtype=np.float32)
        rel_price_te = np.array([lp - cat_median_map.get(c, global_med_price) for c, lp in zip(cats_test, X_test_tab[:, 0])], dtype=np.float32)
        rel_price_anc = rel_price_te[anchor_indices]
        
        # Assemble feature matrices: [Tabular, rel_price, SVD]
        X_tr_dense = np.column_stack([X_train_tab[tr_idx], rel_price_tr[:, None], X_tr_svd])
        X_val_dense = np.column_stack([X_train_tab[val_idx], rel_price_val[:, None], X_val_svd])
        X_te_dense = np.column_stack([X_test_tab, rel_price_te[:, None], X_te_svd])
        X_anc_dense = np.column_stack([X_test_tab[anchor_indices], rel_price_anc[:, None], X_anc_svd])
        
        # Free SVD/TFIDF sparse memory
        del tfidf, svd, X_tr_tfidf, X_val_tfidf, X_te_tfidf, X_anc_tfidf, X_tr_svd, X_val_svd, X_te_svd, X_anc_svd
        gc.collect()
        
        # 3. Construct Augmented Dataset: Fold Train (w=1.0) + Test Anchors (w=0.40)
        X_aug = np.vstack([X_tr_dense, X_anc_dense])
        y_aug = np.concatenate([y_train[tr_idx], inv_labels[anchor_indices]])
        w_aug = np.concatenate([np.ones(len(tr_idx), dtype=np.float32), np.full(len(anchor_indices), 0.40, dtype=np.float32)])
        domains_aug = np.concatenate([train_domains[tr_idx], test_domains[anchor_indices]])
        
        print(f"  Augmented training set: {X_aug.shape[0]} samples ({len(tr_idx)} real + {len(anchor_indices)} anchors)")
        
        # 4. Train Global Model
        print("  Training Global Model...", flush=True)
        t_glob = time.time()
        dtrain_glob = lgb.Dataset(X_aug, label=y_aug, weight=w_aug)
        model_glob = lgb.train(global_lgb_params, dtrain_glob, num_boost_round=450)
        
        p_val_glob = model_glob.predict(X_val_dense)
        p_te_glob = model_glob.predict(X_te_dense)
        print(f"  Global model fitted in {time.time()-t_glob:.1f}s")
        
        # 5. Train Domain Specialists
        p_val_fold = p_val_glob.copy()
        p_te_fold = p_te_glob.copy()
        
        specialist_names = ['MLA3025_Libros', 'MLA1747_Repuestos_Autos', 'MLA1051_Celulares', 'MLA1000_Electrónica', 'Antiques_Home']
        
        for dom in specialist_names:
            dom_aug_mask = (domains_aug == dom)
            n_dom_aug = int(dom_aug_mask.sum())
            val_dom_mask = (train_domains[val_idx] == dom)
            te_dom_mask = (test_domains == dom)
            
            if n_dom_aug < 150:
                print(f"  Specialist {dom}: Skipped (n={n_dom_aug} < 150)")
                continue
                
            t_spec = time.time()
            # Spec params adjustment for Books vs smaller domains
            params_d = dict(spec_lgb_params)
            if dom == 'MLA3025_Libros':
                params_d['num_leaves'] = 63
                params_d['num_boost_round'] = 400
            else:
                params_d['num_leaves'] = 45
                params_d['num_boost_round'] = 300
                
            dtrain_dom = lgb.Dataset(X_aug[dom_aug_mask], label=y_aug[dom_aug_mask], weight=w_aug[dom_aug_mask])
            model_dom = lgb.train(params_d, dtrain_dom, num_boost_round=params_d.get('num_boost_round', 300))
            
            # Predict validation items for this domain
            if val_dom_mask.any():
                p_val_spec = model_dom.predict(X_val_dense[val_dom_mask])
                # Blended specialist: 0.70 specialist + 0.30 global
                p_val_fold[val_dom_mask] = 0.70 * p_val_spec + 0.30 * p_val_glob[val_dom_mask]
                
            # Predict test items for this domain
            if te_dom_mask.any():
                p_te_spec = model_dom.predict(X_te_dense[te_dom_mask])
                p_te_fold[te_dom_mask] = 0.70 * p_te_spec + 0.30 * p_te_glob[te_dom_mask]
                
            acc_spec_val = accuracy_score(y_train[val_idx][val_dom_mask], (p_val_fold[val_dom_mask] >= 0.5).astype(int)) if val_dom_mask.any() else 0.0
            print(f"  Specialist {dom:25s} (Train: {n_dom_aug:5d}, Val: {int(val_dom_mask.sum()):4d}) -> Val Acc: {acc_spec_val:.4f} ({time.time()-t_spec:.1f}s)")
            
            del dtrain_dom, model_dom
            gc.collect()
            
        # Store fold outputs
        oof_predictions[val_idx] = p_val_fold
        test_fold_predictions[fold - 1] = p_te_fold
        
        # Calculate fold metrics
        f_acc = accuracy_score(y_train[val_idx], (p_val_fold >= 0.5).astype(int))
        f_auc = roc_auc_score(y_train[val_idx], p_val_fold)
        f_ll = log_loss(y_train[val_idx], p_val_fold)
        fold_metrics.append({'fold': fold, 'accuracy': f_acc, 'roc_auc': f_auc, 'logloss': f_ll})
        print(f"\nFold {fold} Completed in {time.time()-f_t0:.1f}s -> ACC: {f_acc:.5f} | AUC: {f_auc:.5f} | LL: {f_ll:.5f}")
        
        del X_tr_dense, X_val_dense, X_te_dense, X_anc_dense, X_aug, y_aug, w_aug, dtrain_glob, model_glob
        gc.collect()

    # ---------------------------------------------------------
    # 7. Global Metrics and Domain Breakdown
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("ALL 5 FOLDS COMPLETE: COMPUTING GLOBAL METRICS & BREAKDOWN")
    print("=" * 80, flush=True)
    
    global_acc = accuracy_score(y_train, (oof_predictions >= 0.5).astype(int))
    global_auc = roc_auc_score(y_train, oof_predictions)
    global_ll = log_loss(y_train, oof_predictions)
    n_correct = int(((oof_predictions >= 0.5).astype(int) == y_train).sum())
    
    print(f"\nGLOBAL OOF ACCURACY: {global_acc:.5f} ({n_correct}/{n_train})")
    print(f"GLOBAL ROC-AUC:      {global_auc:.5f}")
    print(f"GLOBAL LOGLOSS:      {global_ll:.5f}")
    
    print("\n--- PER-DOMAIN ACCURACY BREAKDOWN ---")
    domain_breakdown = {}
    for dom in DOMAINS:
        mask = (train_domains == dom)
        n_dom = int(mask.sum())
        if n_dom == 0:
            continue
        acc_d = accuracy_score(y_train[mask], (oof_predictions[mask] >= 0.5).astype(int))
        auc_d = roc_auc_score(y_train[mask], oof_predictions[mask])
        ll_d = log_loss(y_train[mask], oof_predictions[mask])
        y_used_pct = float(y_train[mask].mean() * 100)
        corr_d = int(((oof_predictions[mask] >= 0.5).astype(int) == y_train[mask]).sum())
        
        domain_breakdown[dom] = {
            'count': n_dom,
            'percent_of_train': float(n_dom / n_train * 100),
            'percent_used': y_used_pct,
            'correct': corr_d,
            'accuracy': float(acc_d),
            'roc_auc': float(auc_d),
            'logloss': float(ll_d)
        }
        print(f"  {dom:25s} | N={n_dom:5d} ({n_dom/n_train*100:5.1f}%) | Used: {y_used_pct:5.1f}% | Acc: {acc_d:6.4f} ({corr_d}/{n_dom}) | AUC: {auc_d:.4f} | LL: {ll_d:.4f}")
        
    print("\n--- TOP 15 CATEGORY ACCURACY BREAKDOWN ---")
    top_cats = pd.Series(cats_train).value_counts().head(15).index.tolist()
    cat_breakdown = {}
    for c in top_cats:
        mask = (cats_train == c)
        n_c = int(mask.sum())
        acc_c = accuracy_score(y_train[mask], (oof_predictions[mask] >= 0.5).astype(int))
        y_used_pct = float(y_train[mask].mean() * 100)
        cat_breakdown[c] = {
            'count': n_c,
            'percent_used': y_used_pct,
            'accuracy': float(acc_c)
        }
        print(f"  {c:10s} | N={n_c:4d} | Used: {y_used_pct:5.1f}% | Acc: {acc_c:6.4f}")

    # ---------------------------------------------------------
    # 8. Save Output Files
    # ---------------------------------------------------------
    print("\n[5/5] Saving OOF, Test Probabilities, and Summary Report...", flush=True)
    oof_csv_path = OUT_DIR / 'oof_category_segments.csv'
    oof_df = pd.DataFrame({
        'id': np.arange(1, n_train + 1),
        'p_used': oof_predictions,
        'target_condition_used': y_train,
        'domain': train_domains,
        'category_id': cats_train,
        'seller_id': sellers_train
    })
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"Saved OOF predictions to {oof_csv_path} ({len(oof_df)} rows).")
    
    test_prob_final = test_fold_predictions.mean(axis=0)
    test_csv_path = OUT_DIR / 'test_probabilities_category_segments.csv'
    test_df = pd.DataFrame({
        'id': np.arange(1, n_test + 1),
        'p_used': test_prob_final
    })
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved Test probabilities to {test_csv_path} ({len(test_df)} rows).")
    
    # Summary JSON report
    report = {
        'model_name': 'Category-Segmented Domain Specialists with Transductive Probe Augmentation',
        'architecture': 'Outer StratifiedGroupKFold(5) by seller_id + Domain Specialists (Books, Vehicles, Celulares, Electronics, Antiques) + Global Fallback',
        'total_train_items': n_train,
        'total_test_items': n_test,
        'invariant_all_anchors_used': n_inv,
        'anchor_sample_weight': 0.40,
        'train_sample_weight': 1.0,
        'global_accuracy': float(global_acc),
        'global_roc_auc': float(global_auc),
        'global_logloss': float(global_ll),
        'total_correct': n_correct,
        'fold_metrics': fold_metrics,
        'domain_breakdown': domain_breakdown,
        'top_category_breakdown': cat_breakdown,
        'runtime_seconds': round(time.time() - t_start, 2)
    }
    
    report_path = OUT_DIR / 'summary_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Saved Summary Report to {report_path}.")
    print(f"\nExecution finished cleanly in {time.time()-t_start:.1f}s.")

if __name__ == '__main__':
    main()
