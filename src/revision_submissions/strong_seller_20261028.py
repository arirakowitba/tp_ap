"""Two-stage, seller-group honest baseline and seller-prior gate."""
import csv, json, zipfile
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score
from seller_diagnostic import load, split, prior

def transform(rows, indices, cats):
    out=[]
    for i in indices:
        r=rows[i]; pic=r.get('pictures') or []; attr=r.get('attributes') or []
        sh=r.get('shipping') or {}; adr=r.get('seller_address') or {}
        city=(adr.get('city') or {}).get('id') or ''
        title=r.get('title') or ''
        vals=[
            np.log1p(float(r.get('price') or 0)),
            np.log1p(float(r.get('initial_quantity') or 0)),
            np.log1p(float(r.get('available_quantity') or 0)),
            np.log1p(float(r.get('sold_quantity') or 0)),
            len(title),len(pic),len(attr),len(r.get('tags') or []),
            int(bool(sh.get('free_shipping'))),int(bool(sh.get('local_pick_up'))),
            int(bool(r.get('accepts_mercadopago'))),
            int(bool(r.get('warranty'))),int(bool(r.get('video_id'))),
            cats['category'].get(r.get('category_id'),0),
            cats['listing'].get(r.get('listing_type_id'),0),
            cats['city'].get(city,0),
        ]
        out.append(vals)
    return np.asarray(out,dtype=np.float32)

def categories(rows, fit):
    ans={}
    for name, fn in [
        ('category',lambda r:r.get('category_id')),
        ('listing',lambda r:r.get('listing_type_id')),
        ('city',lambda r:((r.get('seller_address') or {}).get('city') or {}).get('id'))]:
        count=Counter(fn(rows[i]) for i in fit)
        ans[name]={k:j+1 for j,(k,v) in enumerate(count.most_common()) if v>=3}
    return ans

def run(seed):
    rows,y,s,title=load(); fit,tune,confirm=split(s,seed)
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=90000,
                        sublinear_tf=True,strip_accents='unicode')
    xf=vec.fit_transform([title[i] for i in fit])
    xh={name:vec.transform([title[i] for i in ix]) for name,ix in [('tune',tune),('confirm',confirm)]}
    oof=np.zeros(len(fit))
    for tr,va in GroupKFold(n_splits=4).split(xf,y[fit],s[fit]):
        m=LogisticRegression(C=4,max_iter=120,solver='liblinear')
        m.fit(xf[tr],y[fit][tr]);oof[va]=m.predict_proba(xf[va])[:,1]
    mt=LogisticRegression(C=4,max_iter=120,solver='liblinear').fit(xf,y[fit])
    cats=categories(rows,fit)
    base=transform(rows,fit,cats)
    model=HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=22,
       learning_rate=.06,l2_regularization=8,random_state=42)
    model.fit(np.c_[base,oof],y[fit])
    res={'seed':seed,'fit':len(fit),'metrics':{}}
    for name,ix in [('tune',tune),('confirm',confirm)]:
        pt=mt.predict_proba(xh[name])[:,1]
        p=model.predict_proba(np.c_[transform(rows,ix,cats),pt])[:,1]
        sp,n=prior(s,y,fit,ix,3)
        gate=(n>=3)&(abs(sp-.5)>=.30)&((sp>=.5)!=(p>=.5))
        q=p.copy();q[gate]=.6*sp[gate]+.4*p[gate]
        vals=[]
        for segment,mask in [('all',np.ones(len(ix),bool)),('seen',n>0),('new',n==0),('gate',gate)]:
            if not mask.any():continue
            ya=y[ix][mask];a=p[mask]>=.5;b=q[mask]>=.5
            vals.append({'segment':segment,'n':int(mask.sum()),'base_acc':float(np.mean(a==ya)),
                         'gate_acc':float(np.mean(b==ya)),
                         'fix':int(np.sum((a!=ya)&(b==ya))),
                         'break':int(np.sum((a==ya)&(b!=ya))),
                         'base_ll':float(log_loss(ya,p[mask])),
                         'gate_ll':float(log_loss(ya,q[mask]))})
        res['metrics'][name]=vals
        if name=='confirm':
            with open(f'strong_pairs_{seed}.csv','w',newline='') as f:
                w=csv.writer(f);w.writerow(['id','seller_id','label_train','base_p','seller_prior','seller_n_fit','gate','gate_p','text_p'])
                for j,pr,spj,nj,g,qj,tj in zip(ix,p,sp,n,gate,q,pt):
                    w.writerow([rows[j]['id'],s[j],y[j],pr,spj,nj,int(g),qj,tj])
    return res

if __name__=='__main__':
    a=[run(seed) for seed in (2026,314159)]
    with open('strong_results.json','w') as f:json.dump(a,f,indent=2)
    print(json.dumps(a,indent=2))
