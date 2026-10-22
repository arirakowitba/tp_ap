from pathlib import Path
import sys,os,gc,json,joblib,numpy as np
from scipy.sparse import hstack
from threadpoolctl import threadpool_limits
threadpool_limits(2)
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from train_nbsvm_next import read,vectorize,ratio,model
D=R/'results/training_next/nbsvm';docs,y,_=read(R.parent/'recovered/data/train_data.jsonlines');test,_,_=read(R.parent/'recovered/data/test_data.jsonlines');docs=np.r_[docs,test]
x,[xt],vs=vectorize(docs,np.arange(70000),[np.arange(70000,100000)]);r=ratio(x,y);m=model(x.multiply(r).tocsr(),y,4);p=m.predict_proba(xt.multiply(r).tocsr())[:,1];old=np.load(D/'test.npz')['p'];diff=float(np.max(np.abs(p-old)));temp=D/'nb_model_atomic.joblib';joblib.dump({'vectorizers':vs,'ratio':r,'model':m,'native_weight':.85},temp,compress=3);check=joblib.load(temp);assert check['model'].coef_.shape==m.coef_.shape;os.replace(temp,D/'nb_model.joblib');(D/'checkpoint_integrity.json').write_text(json.dumps({'status':'repaired_and_reload_verified','max_abs_difference_test_vs_saved':diff,'model_bytes':(D/'nb_model.joblib').stat().st_size},indent=2));print('VERIFIED',diff,flush=True)
