"""Fit the bracketed-seller model on all original labels; save probabilities privately."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from late_batch_seller import batch_features
from validate_structural_disjoint import (
    METADATA, PARAMS, PICTURES, StratifiedGroupKFold, ag, causal_features, load, sha,
)


def main():
    src, old = map(Path, sys.argv[1:3])
    with zipfile.ZipFile(src) as z:
        train, attr_train = load(z, "train")
        test, attr_test = load(z, "test")
    assert len(train) == 70000 and len(test) == 30000
    assert train.id.is_unique and test.id.is_unique and not set(train.id)&set(test.id)
    tr, te = map(ag.precompute_item_intrinsic_arrays, (train, test))
    fit, query = np.arange(len(train)), np.arange(len(test))
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"], tr["seller_id"]))
    assert all(not set(tr["seller_id"][a])&set(tr["seller_id"][b]) for a,b in folds)
    previous = old / "line3_today"
    assert sha(src) == json.loads((previous/"manifest.json").read_text())["archive_sha256"]
    ag.compute_causal_seller_features = causal_features([te["start_time"]])
    m = ag.build_fold_matrices(fit,[query],folds,tr,[te],previous/"text_70000.npz")
    all_rows = pd.concat([train, test],ignore_index=True)
    seller_hist = batch_features(all_rows,fit,len(train)+query)
    fields = sorted(METADATA+PICTURES)
    x = np.column_stack([m["X_m1_seen_causal_T"],attr_train[:,fields],seller_hist[:len(train)]])
    xv = np.column_stack([m["X_m1_seen_causal_evals"][0],attr_test[:,fields],seller_hist[len(train):]])
    by_seller = train.groupby("seller_id").start_time.agg(["min","max"])
    history = by_seller.reindex(test.seller_id)
    stamp = test.start_time.to_numpy()
    bracketed = (history["min"].to_numpy()<stamp)&(history["max"].to_numpy()>stamp)
    out = Path("late_batch_refit");out.mkdir(exist_ok=True)
    model = LGBMClassifier(**PARAMS).fit(x,m["y_T"])
    checkpoint = out/"batch.txt"
    model.booster_.save_model(str(checkpoint))
    dest = out/"test_probabilities_private.csv"
    pd.DataFrame({"ID":query+1,"item":test.id,"bracketed":bracketed,
                  "p_batch":model.predict_proba(xv)[:,1]}).to_csv(dest,index=False)
    stats = {"train_rows":len(train),"test_rows":len(test),"bracketed":int(bracketed.sum()),
             "test_sellers_seen":int(history["min"].notna().sum()),"test_with_later_fit":int((history["max"].to_numpy()>stamp).sum()),
             "test_with_earlier_fit":int((history["min"].to_numpy()<stamp).sum()),
             "archive_sha256":sha(src),"probabilities_sha256":sha(dest),"checkpoint_sha256":sha(checkpoint),
             "runner_sha256":sha(Path(__file__)),"scope":"Private prediction checkpoint. No public transfer claimed."}
    (out/"report.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(stats,flush=True)


if __name__=="__main__":
    main()
