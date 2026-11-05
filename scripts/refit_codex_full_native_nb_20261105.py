"""Frozen full70k refit of native/NB recipe. Never uploads to Kaggle."""
import argparse,hashlib,json,time,zipfile,gc,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import HashingVectorizer
from catboost import CatBoostClassifier
try:
 from train_codex_raw_nested_text_histgb_20261104 import text_fit
 from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features
except ModuleNotFoundError:
 from train_raw_nested_text_histgb import text_fit
 from deep_reference import extract_deep_features
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("zip");ap.add_argument("out");args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);t0=time.time()
 with zipfile.ZipFile(args.zip) as z:
  tr=[json.loads(l) for l in z.open("train_data.jsonlines")]
  te=[json.loads(l) for l in z.open("test_data.jsonlines")]
 assert len(tr)==70000 and len(te)==30000
 train_ids=[str(r["id"]) for r in tr];test_ids=[str(r["id"]) for r in te]
 assert len(set(train_ids))==70000 and len(set(test_ids))==30000 and not set(train_ids)&set(test_ids)
 rows=tr+te;nt=len(tr);n=len(rows);fit=np.arange(nt);test=np.arange(nt,n)
 y=np.r_[np.asarray([int(r["condition"]=="used") for r in tr]),np.full(len(te),-1)]
 sellers=np.asarray([str(r["seller_id"]) for r in rows])
 af,cat,_,_,_=extract_deep_features(rows)
 df=af[[c for c in af if not c.startswith(("freq_","seller_","sell_mean_","rel_price_cat"))]].copy()
 for col in ["seller_id","category_id","city"]:
  counts=cat.iloc[fit][col].value_counts();df["fit_freq_"+col]=cat[col].map(counts).fillna(0).astype(float)
 med=df.iloc[fit].groupby(cat.iloc[fit].category_id).log_price.median()
 df["fit_rel_price"]=df.log_price-cat.category_id.map(med).fillna(float(df.iloc[fit].log_price.median()))
 text=np.asarray([str(r.get("title") or "")+" WARRANTY "+str(r.get("warranty") or "")+" ATTR "+
     " ".join(str(a.get("value_name") or a.get("name") or "") for a in r.get("attributes") or [] if isinstance(a,dict)) for r in rows],object)
 cats=["category_id","listing_type_id","buying_mode","currency_id","status","seller_id","state","ship_mode"]
 for key in cats[:6]:df[key]=[str(r.get(key) or "MISSING") for r in rows]
 df["state"]=[str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows]
 df["ship_mode"]=[str((r.get("shipping") or {}).get("mode") or "MISSING") for r in rows]
 del rows,tr,te,af;gc.collect();print("FEATURES_READY",time.time()-t0,flush=True)
 word=HashingVectorizer(n_features=2**16,alternate_sign=False,ngram_range=(1,2),norm="l2",dtype=np.float32)
 char=HashingVectorizer(n_features=2**16,alternate_sign=False,analyzer="char_wb",ngram_range=(3,5),norm="l2",dtype=np.float32)
 X=sparse.hstack([word.transform(text),char.transform(text)],format="csr")
 ptext=np.full(n,np.nan)
 for fold,(a,b) in enumerate(GroupKFold(3).split(fit,y[fit],sellers[fit])):
  assert not set(sellers[fit[a]])&set(sellers[fit[b]])
  ptext[fit[b]]=text_fit(X,y,fit[a],fit[b]);print("TEXT_FOLD",fold,time.time()-t0,flush=True)
 ptext[test]=text_fit(X,y,fit,test)
 assert np.isfinite(ptext).all()
 np.save(out/"nested_text_probabilities.npy",ptext)
 del X;gc.collect()
 preds={}
 for name,dd,tf in [("nested_nb",df.assign(nested_nb_text=ptext),[]),
                    ("native_text",df.assign(native_text=text),["native_text"])]:
  print("FIT_START",name,time.time()-t0,flush=True)
  m=CatBoostClassifier(iterations=900,depth=6,learning_rate=.06,l2_leaf_reg=6,
      random_seed=20261104,thread_count=6,verbose=False,loss_function="Logloss",allow_writing_files=False)
  m.fit(dd.iloc[fit],y[fit],cat_features=cats,text_features=tf)
  preds[name]=m.predict_proba(dd.iloc[test])[:,1]
  m.save_model(str(out/(name+".cbm")))
  np.save(out/(name+"_test_probs.npy"),preds[name])
  print("FIT_DONE",name,time.time()-t0,flush=True)
 blend=.5*preds["nested_nb"]+.5*preds["native_text"]
 prob=pd.DataFrame({"ID":np.arange(1,30001),"item_id":test_ids,"p_nested_nb":preds["nested_nb"],"p_native_text":preds["native_text"],"p_blend":blend})
 prob.to_csv(out/"test_probabilities.csv",index=False)
 cand=pd.DataFrame({"ID":np.arange(1,30001),"condition":np.where(blend>=.51,"used","new")})
 assert len(cand)==30000 and cand.ID.is_unique
 cand.to_csv(out/"candidate_full_native_nb_70k.csv",index=False)
 manifest={"recipe":".5 native CatBoost + .5 nested-NB CatBoost","threshold":.51,"iterations":900,"seed":20261104,
    "n_train":nt,"n_test":len(test),"source_zip_sha256":sha(args.zip),
    "submission_sha256":sha(out/"candidate_full_native_nb_70k.csv"),"test_probs_sha256":sha(out/"test_probabilities.csv"),
    "train_id_sha256":hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
    "test_id_sha256":hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
    "seconds":time.time()-t0,"warning":"Unsubmitted candidate. No public-score guarantee; refit/calibration transport remains unverified.",
    "test_labels_used":False}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print(json.dumps(manifest,indent=2),flush=True)
if __name__=="__main__":main()
