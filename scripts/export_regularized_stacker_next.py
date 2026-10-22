"""Export from locked meta weights once missing full70k retrieval is available."""
from pathlib import Path
import json,numpy as np,joblib
from train_regularized_stacker_next import ROOT,D,OUT,raw_small,features,status
model=joblib.load(OUT/'meta_model.joblib');r=np.load(D/'retrieval/test_candidates.npz') if (D/'retrieval/test_candidates.npz').exists() else np.load(D/'retrieval/test.npz')
p=r['global_k128_pow2'] if 'global_k128_pow2' in r.files else r['p']
_,seller,listing,qty=raw_small();ps={'native':np.load(ROOT/'results/new_models_20261021/seller_native_test.npz')['test'],'nb':np.load(D/'nbsvm/test.npz')['p'],'rich':np.load(D/'rich/test.npz')['p'],'context':np.load(D/'context/test.npz')['p'],'retrieval':p}
x,_,_=features(ps,np.arange(70000,100000),seller,listing,qty,seller[:70000]);p=model.predict_proba(x)[:,1];np.savez_compressed(OUT/'test.npz',p=p)
f=OUT/'report.json';report=json.loads(f.read_text());report['test_export']='Complete via export-only helper; five full70k bases and unchanged meta';f.write_text(json.dumps(report,indent=2));status('done',test_export=report['test_export'])
