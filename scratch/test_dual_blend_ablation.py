import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

df = pd.read_csv("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
y = df["y_true"].values
p_lgb = df["oof_lgb"].values
p_cb = df["oof_cb"].values
p_podium = df["oof_blend"].values

print("--- Grid Search of Dual Blend (LightGBM + CatBoost) vs Podium Engine ---")
best_hits = 0
best_w = 0.0

for w_lgb in np.linspace(0.40, 0.70, 31):
    w_cb = 1.0 - w_lgb
    p_dual = w_lgb * p_lgb + w_cb * p_cb
    acc = accuracy_score(y, (p_dual >= 0.5).astype(int))
    hits = int(round(acc * 70000))
    ll = log_loss(y, p_dual)
    auc = roc_auc_score(y, p_dual)
    if hits > best_hits:
        best_hits = hits
        best_w = w_lgb
    if hits >= 65965:
        print(f"w_lgb={w_lgb:.2f}, w_cb={w_cb:.2f} -> Hits={hits} ({hits - 65961:+d}), Acc={acc:.5f}, LL={ll:.5f}, AUC={auc:.5f}")

print(f"\nBest Dual Blend: w_lgb={best_w:.2f} with {best_hits} hits ({best_hits - 65961:+d} over Podium Engine)!")
