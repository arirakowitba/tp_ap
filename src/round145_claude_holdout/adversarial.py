"""Validación adversaria: ¿se distingue train de test con las features sin etiqueta de feats.py?
usage: python adversarial.py DATA_DIR  (sólo lee feats.pkl). AUC ~0,5 => partición aleatoria."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D = sys.argv[1].rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl"); F = P["F"].copy(); z = P["is_test"].astype(int)
F = F.drop(columns=["seller_id", "category_id", "city"])
for c in F.columns:
    if str(F[c].dtype) == "category": F[c] = F[c].cat.codes
oof = np.zeros(len(F)); imp = np.zeros(F.shape[1])
for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(F, z):
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, verbose=-1, num_threads=1, feature_fraction=0.7), lgb.Dataset(F.iloc[a], z[a]), 300)
    oof[b] = m.predict(F.iloc[b]); imp += m.feature_importance("gain")
print(D, "adversarial AUC", round(roc_auc_score(z, oof), 4))
print(pd.Series(imp, F.columns).sort_values(ascending=False).head(8).round(0).to_dict())
