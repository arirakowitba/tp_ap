import json
import time
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

print("Testing exact Podium replication and challenger on Fold 1...")
t0 = time.time()

# 1. Load Train
raw_path = "data/ap_2026_q_2/raw/train_data.jsonlines"
rows = []
with open(raw_path, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df_tr = pd.DataFrame(rows)
y_all = (df_tr["condition"] == "used").astype(int).values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
trn_idx, val_idx = next(cv.split(df_tr, y_all))
y_train = y_all[trn_idx]
y_val = y_all[val_idx]

# Baseline Podium OOF
podium_oof = pd.read_csv("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
p_pod_lgb = podium_oof.iloc[val_idx]["oof_lgb"].values
p_pod_cb = podium_oof.iloc[val_idx]["oof_cb"].values
p_pod_text = podium_oof.iloc[val_idx]["oof_text"].values
p_pod_blend = podium_oof.iloc[val_idx]["oof_blend"].values

print(f"Podium Fold 1 Blend: Acc = {accuracy_score(y_val, (p_pod_blend >= 0.5).astype(int)):.5f} ({int(round(accuracy_score(y_val, (p_pod_blend >= 0.5).astype(int))*len(y_val)))} hits), LL = {log_loss(y_val, p_pod_blend):.5f}")
print(f"Podium Fold 1 LGB:   Acc = {accuracy_score(y_val, (p_pod_lgb >= 0.5).astype(int)):.5f} ({int(round(accuracy_score(y_val, (p_pod_lgb >= 0.5).astype(int))*len(y_val)))} hits), LL = {log_loss(y_val, p_pod_lgb):.5f}")
print(f"Podium Fold 1 CB:    Acc = {accuracy_score(y_val, (p_pod_cb >= 0.5).astype(int)):.5f} ({int(round(accuracy_score(y_val, (p_pod_cb >= 0.5).astype(int))*len(y_val)))} hits), LL = {log_loss(y_val, p_pod_cb):.5f}")
print(f"Podium Fold 1 Text:  Acc = {accuracy_score(y_val, (p_pod_text >= 0.5).astype(int)):.5f} ({int(round(accuracy_score(y_val, (p_pod_text >= 0.5).astype(int))*len(y_val)))} hits), LL = {log_loss(y_val, p_pod_text):.5f}")
