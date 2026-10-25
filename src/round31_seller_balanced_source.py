"""Text + categorical HistGB challenger with leakage-safe stage-one predictions.

Fit/tune/confirm are pre-existing 49k/7k/14k partitions. No Kaggle labels.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

threadpool_limits(2)
ROOT = Path(__file__).resolve().parent
DEEP = os.getenv('AP_ROUND5_DEEP', '0') == '1'
FRESH = os.getenv('AP_ROUND5_FRESH', '0') == '1'
FULL = os.getenv('AP_ROUND5_FULL', '0') == '1'
WIDE = os.getenv('AP_ROUND5_WIDE', '0') == '1'
REG = os.getenv('AP_ROUND5_REG', '0') == '1'
GROUP = os.getenv('AP_ROUND5_GROUP', '0') == '1'
EXTRA = os.getenv('AP_ROUND5_EXTRA', '0') == '1'
IMAGE = os.getenv('AP_ROUND5_IMAGE', '0') == '1'
BOOST = os.getenv('AP_ROUND5_BOOST', 'hist')
OUT = ROOT / 'tp_ap/results/training_round5' / (('text_deep' if DEEP else 'text_tabular') + ('_wide' if WIDE else '') + ('_regularized' if REG else '') + ('_group' if GROUP else '') + ('_extra' if EXTRA else '') + ('_image' if IMAGE else '') + ('_'+BOOST if BOOST!='hist' else '') + '_oof' + ('_fresh' if FRESH else ''))
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def encode_keys(df):
    keys = {}
    names = ['seller_id', 'category_id', 'listing_type_id', 'seller_city',
             'seller_state', 'shipping_mode', 'buying_mode', 'currency_id']
    for name in names:
        keys[name] = df[name].astype('string').fillna('__missing__').astype(str)
    for a, b in [('seller_id', 'category_id'), ('seller_id', 'listing_type_id'),
                 ('category_id', 'listing_type_id')]:
        keys[a + '_' + b] = keys[a] + '|' + keys[b]
    return keys


def tabular_features(df, y, fit_rows, eval_rows, folds=3, seed=31001):
    excluded = {'row_id', 'split', 'docs', 'y', 'seller_id'}
    nums = [c for c in df if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    # Explicitly avoid interpreting seller_id as a numerical magnitude.
    a = df.iloc[fit_rows][nums].replace([np.inf, -np.inf], np.nan)
    med = a.median().fillna(0)
    an = a.fillna(med).to_numpy(np.float32)
    en = df.iloc[eval_rows][nums].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(np.float32)
    keys = encode_keys(df)
    ky = y[fit_rows]
    prior = ky.mean()
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    if GROUP:
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        fold_ix = list(splitter.split(np.zeros(len(fit_rows)), ky, groups=df.seller_id.iloc[fit_rows]))
    else:
        fold_ix = list(splitter.split(np.zeros(len(fit_rows)), ky))
    for key in keys.values():
        fit_key = key.iloc[fit_rows].reset_index(drop=True)
        eval_key = key.iloc[eval_rows]
        oof = np.empty(len(fit_rows), dtype=np.float32)
        freq = np.empty(len(fit_rows), dtype=np.float32)
        for tr, va in fold_ix:
            stats = pd.DataFrame({'k': fit_key.iloc[tr].to_numpy(), 'y': ky[tr]}).groupby('k').y.agg(['sum', 'count'])
            prior_fold = ky[tr].mean()
            enc = (stats['sum'] + 20 * prior_fold) / (stats['count'] + 20)
            oof[va] = fit_key.iloc[va].map(enc).fillna(prior_fold).to_numpy(np.float32)
            freq[va] = np.log1p(fit_key.iloc[va].map(stats['count']).fillna(0).to_numpy(np.float32))
        stats = pd.DataFrame({'k': fit_key.to_numpy(), 'y': ky}).groupby('k').y.agg(['sum', 'count'])
        enc = (stats['sum'] + 20 * prior) / (stats['count'] + 20)
        an = np.column_stack((an, oof, freq)).astype(np.float32)
        en = np.column_stack((en, eval_key.map(enc).fillna(prior).to_numpy(np.float32),
                              np.log1p(eval_key.map(stats['count']).fillna(0).to_numpy(np.float32)))).astype(np.float32)
    return an, en


def text_teacher(docs, y, fit_rows, eval_rows, analyzer, ngram, seed, groups=None):
    fit_docs = docs[fit_rows]
    eval_docs = docs[eval_rows]
    yy = y[fit_rows]
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=3,
                          max_features=100000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    x = vec.fit_transform(fit_docs)
    xe = vec.transform(eval_docs)
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    oof = np.empty(len(fit_rows), dtype=np.float32)
    if GROUP:
        splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        fold_ix = splitter.split(x, yy, groups=groups[fit_rows])
    else:
        fold_ix = splitter.split(x, yy)
    for tr, va in fold_ix:
        clf = LogisticRegression(C=2.0, solver='liblinear', max_iter=400, random_state=seed)
        clf.fit(x[tr], yy[tr])
        oof[va] = clf.predict_proba(x[va])[:, 1]
    clf = LogisticRegression(C=2.0, solver='liblinear', max_iter=400, random_state=seed)
    clf.fit(x, yy)
    pred = clf.predict_proba(xe)[:, 1].astype(np.float32)
    return oof, pred, {'features': int(x.shape[1]), 'oof_acc': float(accuracy_score(yy, oof >= .5))}


def run_stage(df, y, fit_rows, eval_rows, label):
    docs = df.docs.fillna('').to_numpy(dtype=str)
    a, b = tabular_features(df, y, fit_rows, eval_rows)
    teacher_info = {}
    for name, analyzer, ngram in [('word', 'word', (1, 2)), ('char', 'char_wb', (2, 5))]:
        ao, be, info = text_teacher(docs, y, fit_rows, eval_rows, analyzer, ngram, 31011, df.seller_id.to_numpy())
        a = np.column_stack((a, ao)).astype(np.float32)
        b = np.column_stack((b, be)).astype(np.float32)
        teacher_info[name] = info
    if EXTRA:
        ao, be, info = text_teacher(df.extra_docs.fillna('').to_numpy(dtype=str), y, fit_rows, eval_rows,
                                    'word', (1, 2), 31017, df.seller_id.to_numpy())
        a = np.column_stack((a, ao)).astype(np.float32)
        b = np.column_stack((b, be)).astype(np.float32)
        teacher_info['extra_fields'] = info
    if BOOST=='cat':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=700,depth=6,learning_rate=.045,l2_leaf_reg=12,
                                  loss_function='Logloss',verbose=False,thread_count=2,random_seed=31021)
    elif BOOST=='lgbm':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=650,learning_rate=.035,num_leaves=20,max_depth=-1,
            min_child_samples=55,reg_lambda=18,colsample_bytree=.85,subsample=.85,subsample_freq=1,
            verbosity=-1,n_jobs=2,random_state=31021)
    else:
        model = HistGradientBoostingClassifier(max_iter=550 if WIDE else (300 if REG else 380), learning_rate=.045 if WIDE else .05,
            max_leaf_nodes=63 if WIDE else 31, min_samples_leaf=40 if WIDE else (70 if REG else 35),
            l2_regularization=25 if WIDE else (35 if REG else 15),
            max_bins=127, early_stopping=False, random_state=31021)
    seller_counts = df.seller_id.iloc[fit_rows].value_counts()
    seller_freq = df.seller_id.iloc[fit_rows].map(seller_counts).to_numpy(float)
    weights = np.clip(1 / np.sqrt(seller_freq), .2, 1.0)
    weights /= weights.mean()
    model.fit(a, y[fit_rows], sample_weight=weights)
    p = model.predict_proba(b)[:, 1]
    if label == 'test':
        np.savez_compressed(OUT / f'{label}.npz', ID=np.arange(1, len(eval_rows)+1), p=p)
    else:
        np.savez_compressed(OUT / f'{label}.npz', rows=eval_rows, p=p, y=y[eval_rows])
    return p, {'stage': label, 'n_fit': len(fit_rows), 'n_eval': len(eval_rows),
               'n_features': a.shape[1], 'teachers': teacher_info,
               'elapsed_sec': round(time.time() - START, 1)}


def main():
    compact = pd.read_csv(ROOT / 'gemini_bundle/gemini_full_compact_100k.csv')
    assert len(compact) == 100000 and compact.iloc[:70000].split.eq('train').all()
    if DEEP:
        df = pd.read_pickle(ROOT / 'recovery/extracted/tp_ap/results/training_round2/deep_teacher/features.pkl').copy()
        assert len(df) == len(compact)
        df['docs'] = compact.docs.to_numpy()
        df['y'] = compact.y.to_numpy()
        if IMAGE:
            import re
            image_rows=[]
            for filename in ['train_data.jsonlines','test_data.jsonlines']:
                for line in (ROOT/'raw'/filename).open():
                    d=json.loads(line); dates=[]; lengths=[];prefixes=[]
                    for pic in (d.get('pictures') or []):
                        match=re.match(r'([0-9]+)-[A-Z]+[0-9]+_([0-9]{2})([0-9]{4})',str(pic.get('id','')))
                        if match:
                            month,year=int(match[2]),int(match[3])
                            if 1<=month<=12 and 2000<=year<=2020:dates.append(12*year+month)
                            lengths.append(len(match[1]));prefixes.append(int(match[1]))
                    created=d.get('date_created') or ''
                    match=re.match(r'(\d{4})-(\d{2})',created)
                    created_month=12*int(match[1])+int(match[2]) if match else np.nan
                    image_rows.append({'img_oldest_month':min(dates) if dates else np.nan,
                                       'img_newest_month':max(dates) if dates else np.nan,
                                       'img_age_months':created_month-min(dates) if dates else np.nan,
                                       'img_date_spread':max(dates)-min(dates) if dates else np.nan,
                                       'img_first_prefix_length':lengths[0] if lengths else 0,
                                       'img_first_prefix_log':np.log1p(prefixes[0]) if prefixes else np.nan,
                                       'img_parsed_fraction':len(dates)/max(1,len(d.get('pictures') or []))})
            image_df=pd.DataFrame(image_rows)
            assert len(image_df)==len(df)
            for col in image_df:df[col]=image_df[col].to_numpy()
        if EXTRA:
            extra=[]
            for filename in ['train_data.jsonlines','test_data.jsonlines']:
                for line in (ROOT/'raw'/filename).open():
                    d=json.loads(line); parts=[]
                    for attr in (d.get('attributes') or [])[:20]:
                        if isinstance(attr,dict):parts.extend([str(attr.get('name') or ''),str(attr.get('value_name') or '')])
                    for variation in (d.get('variations') or [])[:8]:
                        if isinstance(variation,dict):parts.append(str(variation.get('attribute_combinations') or ''))
                    extra.append(' '.join(parts))
            df['extra_docs']=extra
    else:
        df = compact
    y = df.y.iloc[:70000].astype(np.int8).to_numpy()
    if FULL:
        train_rows = np.arange(70000)
        test_rows = np.arange(70000,100000)
        # Labels for the test partition are never read or used by the model.
        y_padded = np.r_[y, np.full(30000, -1, dtype=np.int8)]
        p, info = run_stage(df, y_padded, train_rows, test_rows, 'test')
        assert len(p) == 30000 and np.isfinite(p).all()
        report = {'protocol': 'All70k refit with 3fold OOF stage-one teachers and categorical target encodings; config fixed before full fit.',
                  'test_rows': len(p), 'p_range': [float(np.min(p)),float(np.max(p))],
                  'test_used_rate_at_0_5': float(np.mean(p>=.5)), 'elapsed_sec': round(time.time()-START,1),
                  'selection_from': 'text_deep_oof and text_deep_oof_fresh report.json'}
        (OUT / 'full_report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        (OUT / 'RUN_STATE.md').write_text('Full70k test probabilities complete. See test.npz and full_report.json. No Kaggle score.\n', encoding='utf8')
        print(json.dumps(report,indent=2),flush=True)
        return
    parts_path = ('recovery/extracted/results/training_round2/fresh/partitions.npz' if FRESH else
                  'recovery/extracted/results/new_models_20261021/partitions.npz')
    parts = np.load(ROOT / parts_path)
    fit, tune, confirm = [parts[k] for k in ('fit', 'tune', 'confirm')]
    assert not set(fit) & set(tune) and not set(fit) & set(confirm) and not set(tune) & set(confirm)
    p_t, report_t = run_stage(df, y, fit, tune, 'tune')
    thresholds = np.arange(.46, .541, .005)
    threshold = float(max(thresholds, key=lambda t: (accuracy_score(y[tune], p_t >= t), -abs(t-.5))))
    report_t['selected_threshold'] = threshold
    report_t['accuracy'] = float(accuracy_score(y[tune], p_t >= threshold))
    report_t['logloss'] = float(log_loss(y[tune], p_t))
    (OUT / 'RUN_STATE.md').write_text(f'Tune complete: {report_t}\nConfirmation is not yet opened.\n', encoding='utf8')

    dev = np.r_[fit, tune]
    p_c, report_c = run_stage(df, y, dev, confirm, 'confirm')
    decision = p_c >= threshold
    report_c['threshold_locked_on_tune'] = threshold
    report_c['accuracy'] = float(accuracy_score(y[confirm], decision))
    report_c['logloss'] = float(log_loss(y[confirm], p_c))
    comparisons = {}
    if FRESH:
        refs = [
            ('native', ROOT / 'recovery/extracted/tp_ap/results/training_round2/fresh/confirm_native.npz', .5),
            ('hist_te', ROOT / 'tp_ap/results/training_round4/diverse_model/fresh/confirm.npz', .5),
        ]
    else:
        refs = [
            ('native', ROOT / 'recovery/extracted/results/new_models_20261021/seller_native_confirm.npz', .5),
            ('fields', ROOT / 'recovery/extracted/tp_ap/results/training_round3/fields/confirm.npz', .475),
            ('hist_te', ROOT / 'tp_ap/results/training_round4/diverse_model/confirm.npz', .5),
        ]
    for name, path, ref_threshold in refs:
        if not path.exists():
            continue
        data = np.load(path)
        assert np.array_equal(data['rows'], confirm), name
        old = data['p'] >= ref_threshold
        new = decision
        yy = y[confirm].astype(bool)
        comparisons[name] = {'threshold': ref_threshold,
            'old_accuracy': float(accuracy_score(yy, old)),
            'fixes': int(np.sum((new == yy) & (old != yy))),
            'breaks': int(np.sum((new != yy) & (old == yy))),
            'net': int(np.sum(new == yy) - np.sum(old == yy))}
    report = {'protocol': 'Existing split49k/7k/14k, threshold selected on tune; 3fold OOF text logits and target encodings inside each fit stage. Holdout historically reused; public scores never used as labels.',
              'tune': report_t, 'confirm': report_c, 'comparisons': comparisons,
              'test_predictions': False, 'elapsed_sec': round(time.time()-START,1)}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    (OUT / 'RUN_STATE.md').write_text('Confirmation complete. See report.json. Full test fit only if validation earns it.\n', encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
