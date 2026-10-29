"""Train-only comparison of seller history against a richer sparse baseline."""
import csv
import json
import zipfile
from collections import defaultdict

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from seller_diagnostic import split, prior


def fields(r):
    city = (r.get("seller_address") or {}).get("city") or {}
    state = (r.get("seller_address") or {}).get("state") or {}
    ship = r.get("shipping") or {}
    return {
        "category=" + str(r.get("category_id")): 1,
        "listing=" + str(r.get("listing_type_id")): 1,
        "buying=" + str(r.get("buying_mode")): 1,
        "state=" + str(state.get("id")): 1,
        "ship=" + str(ship.get("mode")): 1,
        "free_shipping": int(bool(ship.get("free_shipping"))),
        "accepts_mp": int(bool(r.get("accepts_mercadopago"))),
        "log_price": np.log1p(float(r.get("price") or 0)),
        "log_initial_quantity": np.log1p(float(r.get("initial_quantity") or 0)),
        "log_sold_quantity": np.log1p(float(r.get("sold_quantity") or 0)),
        "has_warranty": int(bool(r.get("warranty"))),
        "has_description": int(bool(r.get("descriptions"))),
        "title_len": min(len(r.get("title") or ""), 200) / 100,
    }


def main():
    with zipfile.ZipFile("data.zip") as z, z.open("train_data.jsonlines") as f:
        rows = [json.loads(line) for line in f]
    title = [r.get("title") or "" for r in rows]
    y = np.array([r["condition"] == "new" for r in rows], dtype=int)
    seller = np.array([str(r["seller_id"]) for r in rows])
    metadata = [fields(r) for r in rows]
    results = []
    for seed in [2026, 314159]:
        fit, tune, confirm = split(seller, seed)
        word = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=70000,
                               strip_accents="unicode", sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=8,
                               max_features=110000, sublinear_tf=True)
        meta = DictVectorizer()
        wfit = word.fit_transform([title[i] for i in fit])
        cfit = char.fit_transform([title[i] for i in fit])
        mfit = meta.fit_transform([metadata[i] for i in fit])
        xfit = hstack([wfit,cfit,mfit],format="csr")
        model = LogisticRegression(C=3,solver="liblinear",max_iter=150)
        model.fit(xfit,y[fit])
        result={"seed":seed,"fit":len(fit),"features":xfit.shape[1]}
        for segment, ix in [("tune",tune),("confirm",confirm)]:
            xt = hstack([word.transform([title[i] for i in ix]),
                         char.transform([title[i] for i in ix]),
                         meta.transform([metadata[i] for i in ix])],format="csr")
            p = model.predict_proba(xt)[:,1]
            pp,n = prior(seller,y,fit,ix,3.)
            # A single fixed rule, no confirm-specific fitting.
            gate=(n>=3)&(np.abs(pp-.5)>=.3)&((pp>=.5)!=(p>=.5))
            mix=p.copy(); mix[gate]=.6*pp[gate]+.4*p[gate]
            records=[]
            for name,mask in [("all",np.ones(len(ix),bool)),("seen",n>0),
                              ("new",n==0),("gate",gate)]:
                if not mask.any(): continue
                a=(p[mask]>=.5); b=(mix[mask]>=.5); yy=y[ix[mask]]
                records.append({"segment":name,"n":len(yy),
                                "baseline_accuracy":float(accuracy_score(yy,a)),
                                "candidate_accuracy":float(accuracy_score(yy,b)),
                                "fix":int(np.sum((a!=yy)&(b==yy))),
                                "break":int(np.sum((a==yy)&(b!=yy))),
                                "baseline_logloss":float(log_loss(yy,p[mask])),
                                "candidate_logloss":float(log_loss(yy,mix[mask]))})
            result[segment]=records
            if segment=="confirm":
                with open(f"agent_signal_pairs_{seed}.csv","w",newline="") as f:
                    wr=csv.writer(f)
                    wr.writerow(["id","seller_id","condition_new_train","baseline_p","seller_prior_fit","seller_count_fit","gate","candidate_p"])
                    for rowid,j,pr,q,k,g,pm in zip(range(len(ix)),ix,p,pp,n,gate,mix):
                        wr.writerow([rows[j]["id"],seller[j],y[j],pr,q,k,int(g),pm])
        results.append(result)
        print(json.dumps(result),flush=True)
    with open("agent_signal_results.json","w") as f:json.dump(results,f,indent=2)

if __name__=="__main__": main()
