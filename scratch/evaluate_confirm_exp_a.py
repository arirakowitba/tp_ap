import os
import sys
import time
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

sys.path.append('.')
from scripts.train_ag0235_tabular_causal_and_comparison import (
    compute_causal_history,
    build_claude_text_features,
    bootstrap_multiway_and_marginals,
    clip_prob,
    logit, sigmoid
)

t0 = time.time()
print("=== SINGLE EVALUATION ON INTACT CONFIRM SET (14,075 ROWS) ===")

with open('data/splits/t98_parsed_train.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open('data/ap_2026_q_2/raw/feats.pkl', 'rb') as f:
    feats = pickle.load(f)

F = feats['F'].copy()
y_all = feats['y'].copy()
y_true_70k = (y_all == 1).astype(int) if set(np.unique(y_all)).issubset({0, 1}) else (y_all == 'used').astype(int)
df_train['y_num'] = y_true_70k

df_3way = pd.read_csv('data/splits/canonical_3way_splits.csv')
text_cache = np.load('data/splits/full70k_text_cache.npz')

ptx1_train = text_cache['ptx1_oof']
ptx2_train = text_cache['ptx2_oof']
svd_train = text_cache['svd_train'][:, :16]

fit_mask = (df_3way['role'] == 'fit').values
conf_mask = (df_3way['role'] == 'confirm').values

df_fit = df_train.loc[fit_mask].copy().reset_index(drop=True)
df_conf = df_train.loc[conf_mask].copy().reset_index(drop=True)
y_fit = y_true_70k[fit_mask]
y_conf = y_true_70k[conf_mask]

# Numeric and categorical features exactly as in EXP-A
num_cols = [
    'price', 'base_price', 'initial_quantity', 'available_quantity', 'sold_quantity', 'log_price',
    'price_eq_base', 'sold_ratio', 'avail_ratio', 'init_minus_avail', 'price_cents', 'price_round100',
    'price_round10', 'price_end9', 'has_original_price', 'official_store', 'has_video', 'has_catalog',
    'has_parent', 'automatic_relist', 'accepts_mp', 'n_deals', 'currency_usd', 'n_tags', 'n_pm',
    'ship_local_pick_up', 'ship_free', 'ship_n_methods', 'ship_n_tags', 'n_pics', 'pic0_w', 'pic0_h',
    'pic_max_area', 'pic_mean_area', 'pic_square', 'n_variations', 'n_attributes', 'dur_days',
    'warranty_null', 'warranty_len', 'w_sin', 'w_no', 'w_meses', 'w_anio', 'w_fabrica', 'w_oficial',
    'w_usado', 'w_nuevo', 'w_dias', 'w_garantia', 'title_len', 'title_words', 'title_upper_frac',
    'title_digits', 't_usado', 't_nuevo', 't_original', 't_antiguo', 't_libro', 't_lote', 't_impecable',
    'cat_med_logprice', 'rel_price_cat'
]
avail_num_cols = [c for c in num_cols if c in F.columns]
X_tab_70k = F.iloc[:70000][avail_num_cols].to_numpy(dtype=np.float32)
X_tab_70k = np.nan_to_num(X_tab_70k, nan=0.0)

cat_cols = ['listing_type_id', 'buying_mode', 'category_id', 'ship_mode', 'state']
cat_arrays = []
for c in cat_cols:
    cat_s = F.iloc[:70000][c].astype('category').cat.codes.to_numpy(dtype=np.int32)
    cat_arrays.append(cat_s[:, None])
X_cat_70k = np.column_stack(cat_arrays)

X_tab_full = np.column_stack([X_tab_70k, X_cat_70k, ptx1_train[:, None], ptx2_train[:, None], svd_train])

X_base_fit = X_tab_full[fit_mask]
X_base_conf = X_tab_full[conf_mask]

# Causal features on fit -> confirm
c_m_fit, c_c_fit, c_m_conf, c_c_conf = compute_causal_history(df_fit, df_conf, key='seller_id', prior=0.4722, smoothing=5.0)
c_m_fit_b10, c_c_fit_b10, c_m_conf_b10, c_c_conf_b10 = compute_causal_history(df_fit, df_conf, key='seller_id', prior=0.4722, smoothing=10.0)

X_causal_fit = np.column_stack([X_base_fit, c_m_fit, c_c_fit, c_m_fit_b10, c_c_fit_b10])
X_causal_conf = np.column_stack([X_base_conf, c_m_conf, c_c_conf, c_m_conf_b10, c_c_conf_b10])

lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.04,
    'num_leaves': 36,
    'min_child_samples': 30,
    'colsample_bytree': 0.75,
    'subsample': 0.85,
    'reg_lambda': 2.0,
    'random_state': 42,
    'n_jobs': 6,
    'verbose': -1
}

print("Fitting AG_Tabular_Base on fit (40,881)...")
m_base = LGBMClassifier(**lgb_params).fit(X_base_fit, y_fit)
p_base_conf = m_base.predict_proba(X_base_conf)[:, 1]

print("Fitting AG_Tabular_Causal on fit (40,881)...")
m_causal = LGBMClassifier(**lgb_params).fit(X_causal_fit, y_fit)
p_causal_conf = m_causal.predict_proba(X_causal_conf)[:, 1]

print("Fitting Claude text model on fit (40,881)...")
Xtr_text, Xconf_text = build_claude_text_features(df_fit, df_conf)
lr_claude = LogisticRegression(C=1.0, solver='liblinear', max_iter=2000, random_state=42).fit(Xtr_text, y_fit)
p_claude_conf = lr_claude.predict_proba(Xconf_text)[:, 1]

# Blend w=0.2 (frozen from tune selection)
z_blend = 0.8 * logit(p_causal_conf) + 0.2 * logit(p_claude_conf)
p_blend_conf = sigmoid(z_blend)

acc_base = accuracy_score(y_conf, p_base_conf >= 0.5)
acc_causal = accuracy_score(y_conf, p_causal_conf >= 0.5)
acc_blend = accuracy_score(y_conf, p_blend_conf >= 0.5)
acc_claude = accuracy_score(y_conf, p_claude_conf >= 0.5)

ll_base = log_loss(y_conf, clip_prob(p_base_conf))
ll_causal = log_loss(y_conf, clip_prob(p_causal_conf))
ll_blend = log_loss(y_conf, clip_prob(p_blend_conf))

# Fixes and breaks vs base
c_base = ((p_base_conf >= 0.5) == y_conf)
c_causal = ((p_causal_conf >= 0.5) == y_conf)
c_blend = ((p_blend_conf >= 0.5) == y_conf)

fixes_causal = int((c_causal & ~c_base).sum())
breaks_causal = int((~c_causal & c_base).sum())
net_causal = fixes_causal - breaks_causal

fixes_blend = int((c_blend & ~c_base).sum())
breaks_blend = int((~c_blend & c_base).sum())
net_blend = fixes_blend - breaks_blend

# Bootstrap Q05 on Confirm
sellers_conf = df_conf['seller_id'].values
cats_conf = df_conf['category_id'].values

boot_causal = bootstrap_multiway_and_marginals(y_conf, p_causal_conf, p_base_conf, sellers_conf, cats_conf)
boot_blend = bootstrap_multiway_and_marginals(y_conf, p_blend_conf, p_base_conf, sellers_conf, cats_conf)

# Cold vs warm breakdown
is_cold_conf = df_3way.loc[conf_mask, 'is_cold_confirm'].values.astype(bool)
is_warm_conf = df_3way.loc[conf_mask, 'is_warm_confirm'].values.astype(bool)

print("\n" + "=" * 60)
print("CONFIRM SET RESULTS (14,075 rows: 7,875 cold, 6,200 warm)")
print("=" * 60)
print(f"AG_Tabular_Base   : Acc={acc_base:.5f} (Cold={accuracy_score(y_conf[is_cold_conf], p_base_conf[is_cold_conf]>=0.5):.5f}, Warm={accuracy_score(y_conf[is_warm_conf], p_base_conf[is_warm_conf]>=0.5):.5f}) LL={ll_base:.5f}")
print(f"AG_Tabular_Causal : Acc={acc_causal:.5f} (Cold={accuracy_score(y_conf[is_cold_conf], p_causal_conf[is_cold_conf]>=0.5):.5f}, Warm={accuracy_score(y_conf[is_warm_conf], p_causal_conf[is_warm_conf]>=0.5):.5f}) LL={ll_causal:.5f}")
print(f"Claude_Text_Alone : Acc={acc_claude:.5f} (Cold={accuracy_score(y_conf[is_cold_conf], p_claude_conf[is_cold_conf]>=0.5):.5f}, Warm={accuracy_score(y_conf[is_warm_conf], p_claude_conf[is_warm_conf]>=0.5):.5f})")
print(f"Blend (w=0.2)     : Acc={acc_blend:.5f} (Cold={accuracy_score(y_conf[is_cold_conf], p_blend_conf[is_cold_conf]>=0.5):.5f}, Warm={accuracy_score(y_conf[is_warm_conf], p_blend_conf[is_warm_conf]>=0.5):.5f}) LL={ll_blend:.5f}")
print("-" * 60)
print(f"Causal vs Base on Confirm: Fixes={fixes_causal}, Breaks={breaks_causal}, Net={net_causal:+d} aciertos")
print(f"  Bootstrap Q05 Seller: {boot_causal['marginal_seller']['q05']:+.1f}, Q05 Category: {boot_causal['marginal_category']['q05']:+.1f}, Multiway Q05: {boot_causal['multiway']['q05']:+.1f}")
print(f"Blend vs Base on Confirm : Fixes={fixes_blend}, Breaks={breaks_blend}, Net={net_blend:+d} aciertos")
print(f"  Bootstrap Q05 Seller: {boot_blend['marginal_seller']['q05']:+.1f}, Q05 Category: {boot_blend['marginal_category']['q05']:+.1f}, Multiway Q05: {boot_blend['multiway']['q05']:+.1f}")
print(f"Elapsed: {time.time()-t0:.1f}s")
