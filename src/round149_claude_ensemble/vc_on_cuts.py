"""CC-0038: AG variant_c (CatBoost with native seller_id + AG features, fixed 600 iterations as in its test refit) trained on the
fit rows of Claude's two cuts and predicted on their eval rows, using AG's own modules from a worktree (no code changes).
Cuts (original 70k row indices): holdout = fit T (56k, fit_train_rows.npy) / eval H (14k, holdout_train_rows.npy);
inner = fit T[~te] / eval T[te] with te = RandomState(145).choice(56000, 11200) (make_inner_split.py).
usage: python vc_on_cuts.py AG_WORKTREE ROWS_DIR(with fit_train_rows/holdout_train_rows) OUTDIR SEEDS(comma)"""
import sys, time, json
from pathlib import Path
wt, st, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(',')]
out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(wt))
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from src.benchmark_20261030.feature_extractor import compute_txt_sparse_p
from src.intensive_20261030.seller_te_ablation import compute_target_encodings_ablation
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

cache = build_or_load_cache(); F = cache["F_r151_tr"]; cat_tr = cache["cat_tr"]; y_tr = cache["y_tr"]; sellers_tr = cache["sellers_tr"]
T = np.load(st / 'fit_train_rows.npy'); H = np.load(st / 'holdout_train_rows.npy')
te = np.zeros(len(T), bool); te[np.random.RandomState(145).choice(len(T), len(T) // 5, replace=False)] = True
cuts = {'inner': (T[~te], T[te]), 'holdout': (T, H)}
rep = {}
for name, (fit, ev) in cuts.items():
    t0 = time.time(); y_fit = y_tr[fit]; sf = sellers_tr[fit]
    oof_sp, (p_sp_ev, _) = compute_txt_sparse_p(cache["combos_hist_tr"], cache["titles_tr"], fit, [ev, ev], y_fit, sf, seed=2026)
    fit_te, ev_te, _ = compute_target_encodings_ablation(cat_tr, fit, ev, ev, y_fit, sf, variant="variant_a")
    Xf = F.iloc[fit].copy().reset_index(drop=True); Xe = F.iloc[ev].copy().reset_index(drop=True)
    for c in ["seller_id", "category_id", "city"]:
        Xf[f"te_{c}"] = fit_te[c]; Xe[f"te_{c}"] = ev_te[c]
    Xf["txt_sparse_p"] = oof_sp; Xe["txt_sparse_p"] = p_sp_ev
    cbf = list(CAT_FEATURES) + ["seller_id"]
    for X, ix in ((Xf, fit), (Xe, ev)):
        X["seller_id"] = cat_tr["seller_id"].iloc[ix].astype(str).values
        for c in CAT_FEATURES:
            X[c] = cat_tr[c].iloc[ix].astype(str).values
    P = {}
    for s in seeds:
        m = CatBoostClassifier(iterations=600, learning_rate=0.07, depth=6, l2_leaf_reg=6.0, random_seed=s, thread_count=1,
                               allow_writing_files=False, verbose=0).fit(Pool(Xf, y_fit, cat_features=cbf))
        P[s] = m.predict_proba(Pool(Xe, cat_features=cbf))[:, 1]
    df = pd.DataFrame(dict(orig_row=ev, y=y_tr[ev], **{f'p_vc_s{s}': P[s] for s in seeds}))
    df.to_csv(out / f'vc_{name}.csv.gz', index=False)
    rep[name] = dict(n_fit=len(fit), n_eval=len(ev), acc={str(s): float(((P[s] >= .5) == y_tr[ev]).mean()) for s in seeds}, sec=round(time.time() - t0))
    print(name, rep[name], flush=True)
(out / 'vc_report.json').write_text(json.dumps(rep, indent=1)); print('VC DONE', flush=True)
