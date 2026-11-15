"""CC-0041: corrected re-implementation of AG's causal family on AG's canonical 3-way split (fit 40,881 / tune 15,044 / confirm 14,075).
Fixes vs AG-0237: (a) text probability for FIT rows is cross-fitted (5-fold inside fit), never in-sample; eval rows use the fit-only text model;
(b) causal seller history for every row is built ONLY from fit labels, strictly earlier start_time (AG rule, ties excluded); fit rows never see
their own label; (c) the 9 label-free global aggregates (freq_*, seller_*, cat_med_logprice, rel_price_cat) are recomputed on fit rows only.
Models: LightGBM base (no history) and causal (+4 history cols), AG's hyperparameters. Outputs per-row p for tune+confirm.
usage: python causal_corrected.py SPLITS_CSV FEATS_PKL TRAIN_JSONL OUT_CSV"""
import sys, re, unicodedata, json, numpy as np, pandas as pd, scipy.sparse as sp
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
spl, fpk, trj, out = sys.argv[1:5]
s = pd.read_csv(spl); d = pd.read_pickle(fpk); F = d['F'].iloc[:70000].reset_index(drop=True); y = s.y.values.astype(int)
raw = pd.read_json(trj, lines=True); assert (raw.id.values == s.id.values).all()
fit = np.where(s.role == 'fit')[0]; ev = np.where(s.role.isin(['tune', 'confirm']))[0]
AGG = ['freq_seller_id', 'freq_category_id', 'freq_city', 'seller_n_cats', 'seller_mean_logprice', 'seller_mean_init_q', 'cat_med_logprice', 'rel_price_cat']
X = F.drop(columns=AGG + ['seller_id']).copy()
for c in X.columns:
    if X[c].dtype == object or str(X[c].dtype) == 'category': X[c] = pd.Categorical(X[c].astype(str), categories=pd.Index(X[c].astype(str).iloc[fit].unique()))
ff = F.iloc[fit]
for k in ['seller_id', 'category_id', 'city']:
    X['freq_' + k] = F[k].map(ff[k].value_counts()).fillna(0).values
X['seller_n_cats'] = F.seller_id.map(ff.groupby('seller_id').category_id.nunique()).fillna(0).values
X['seller_mean_logprice'] = F.seller_id.map(ff.groupby('seller_id').log_price.mean()).values
X['seller_mean_init_q'] = F.seller_id.map(ff.groupby('seller_id').initial_quantity.mean()).values
med = ff.groupby('category_id').log_price.median(); X['cat_med_logprice'] = F.category_id.map(med).fillna(ff.log_price.median()).values
X['rel_price_cat'] = F.log_price.values - X['cat_med_logprice'].values
def norm(t):
    t = '' if t is None or (isinstance(t, float) and np.isnan(t)) else str(t); t = unicodedata.normalize('NFKD', t.lower())
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9ñ ]', ' ', ''.join(ch for ch in t if not unicodedata.combining(ch)))).strip()
txt = (raw.title.map(norm) + ' ' + raw.warranty.map(norm)).values
def text_fit_predict(tr, te_):
    v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); v2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=300000, sublinear_tf=True)
    A = sp.hstack([v1.fit_transform(txt[tr]), v2.fit_transform(txt[tr])]).tocsr(); B = sp.hstack([v1.transform(txt[te_]), v2.transform(txt[te_])]).tocsr()
    return LogisticRegression(C=1.0, solver='liblinear', max_iter=2000).fit(A, y[tr]).predict_proba(B)[:, 1]
ptext = np.zeros(70000)
for a, b in StratifiedKFold(5, shuffle=True, random_state=2026).split(fit, y[fit]): ptext[fit[b]] = text_fit_predict(fit[a], fit[b])
ptext[ev] = text_fit_predict(fit, ev); X['p_text_crossfit'] = np.log(np.clip(ptext, 1e-6, 1 - 1e-6) / (1 - np.clip(ptext, 1e-6, 1 - 1e-6)))
# causal history from FIT labels only, strictly earlier time
t = s.start_time.values; sel = s.seller_id.astype(str).values; hist = {}
for k, g in pd.DataFrame(dict(s=sel[fit], t=t[fit], y=y[fit])).groupby('s'):
    b = g.groupby('t').y.agg(['count', 'sum']); hist[k] = (b.index.values, np.r_[0, b['count'].cumsum().values], np.r_[0, b['sum'].cumsum().values])
cnt = np.zeros(70000); sm = np.zeros(70000)
for i in range(70000):
    h = hist.get(sel[i])
    if h is None: continue
    j = np.searchsorted(h[0], t[i], side='left'); cnt[i] = h[1][j]; sm[i] = h[2][j]
for beta in (5, 10): X[f'c_m_{beta}'] = (sm + beta * 0.4722) / (cnt + beta); X[f'c_c_{beta}'] = np.log1p(cnt)
prm = dict(n_estimators=500, learning_rate=0.04, num_leaves=36, colsample_bytree=0.75, subsample=0.85, subsample_freq=1, reg_lambda=2.0, random_state=42, n_jobs=1, verbose=-1)
cz = [c for c in X.columns if c.startswith('c_')]; Xb = X.drop(columns=cz)
pb = LGBMClassifier(**prm).fit(Xb.iloc[fit], y[fit]).predict_proba(Xb.iloc[ev])[:, 1]
pc = LGBMClassifier(**prm).fit(X.iloc[fit], y[fit]).predict_proba(X.iloc[ev])[:, 1]
o = pd.DataFrame(dict(row=ev, id=s.id.values[ev], role=s.role.values[ev], regime=s.regime.values[ev], y=y[ev], p_base=pb, p_causal=pc, p_text=ptext[ev]))
o.to_csv(out, index=False)
for r in ['tune', 'confirm']:
    m = o.role == r; print(r, 'base', round(float(((o.p_base[m] >= .5) == o.y[m]).mean()), 5), 'causal', round(float(((o.p_causal[m] >= .5) == o.y[m]).mean()), 5), 'text', round(float(((o.p_text[m] >= .5) == o.y[m]).mean()), 5), flush=True)
