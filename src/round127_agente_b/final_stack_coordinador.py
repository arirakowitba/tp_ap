import numpy as np, pandas as pd, json, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
D="/home/claude/data/"; y=pd.read_pickle(D+"feats.pkl")["y"]
lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def ev(o): return ((o>.5)==y).mean(), -np.mean(y*np.log(np.clip(o,1e-6,1))+(1-y)*np.log(np.clip(1-o,1e-6,1)))
for m in ["b_cat_noso","b_cat_noso_b","b_cat_noso_c","b_cat_noso_d","b_cat_noso_lr03","b_cat_pf","a_cat_base_s81","a_cat_raw_s81","a_cat_raw_lr03_s83","a_cat_sraw_s81","a_cat_min_s81","a_cat_all_s81","b_xgb_noso","b_xgb_noso_d7","b_xgb_pf","c_nn_avg4"]:
    o=np.load(D+f"pred_{m}.npz")["oof"]; a,l=ev(o); print(f"{m:22s} {a:.5f} {l:.5f}")
cats=["b_cat_noso","b_cat_noso_b","b_cat_noso_c","b_cat_noso_d","b_cat_noso_lr03"]
avg_o=np.mean([np.load(D+f"pred_{m}.npz")["oof"] for m in cats],0); avg_t=np.mean([np.load(D+f"pred_{m}.npz")["te"] for m in cats],0)
print("avg5 honest cat", ev(avg_o))
np.savez(D+"pred_fin_cat5.npz",oof=avg_o,te=avg_t)
def stack(names, C=0.1, save=None):
    O=np.column_stack([lg(np.load(D+f"pred_{m}.npz")["oof"]) for m in names]); T=np.column_stack([lg(np.load(D+f"pred_{m}.npz")["te"]) for m in names])
    oof=np.zeros(len(y))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=2026).split(O,y):
        oof[b]=LogisticRegression(C=C,max_iter=2000).fit(O[a],y[a]).predict_proba(O[b])[:,1]
    m=LogisticRegression(C=C,max_iter=2000).fit(O,y); te=m.predict_proba(T)[:,1]
    print(names, ev(oof), m.coef_[0].round(2))
    if save: np.savez(D+f"pred_{save}.npz",oof=oof,te=te)
    return oof
stack(["fin_cat5","b_xgb_noso"])
stack(["fin_cat5","c_nn_avg4"])
stack(["fin_cat5","b_xgb_noso","c_nn_avg4"],save="fin_stack")
stack(["fin_cat5","b_xgb_noso","b_xgb_noso_d7","c_nn_avg4","b_cat_pf"])
