"""Paired metadata ablation and video-subgroup diagnostics; no pixels implied."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
def paired(a,b,y):
    ca=(a>=.5)==y;cb=(b>=.5)==y
    repaired=int((~ca&cb).sum());introduced=int((ca&~cb).sum())
    return dict(n=len(y),original_accuracy=float(ca.mean()),metadata_accuracy=float(cb.mean()),
                corrected=repaired,introduced=introduced,net_correct=repaired-introduced,
                mcnemar_exact_p=float(binomtest(repaired,repaired+introduced,.5).pvalue) if repaired+introduced else 1.)

def run(output):
    out=Path(output);media=pd.read_csv(out/'media_features_train.csv').set_index('row_index')
    rows=[]
    for folder,base,new in [(out,'r4_lgb_original','r4_lgb_media'),
                            (out/'fixed_ablation','r4_lgb_fixed430_original','r4_lgb_fixed430_media'),
                            (out,'r4_cat_enriched_d9_original','r4_cat_enriched_d9_media'),
                            (out,'r4_cat_native_text_d7_original','r4_cat_native_text_d7_media')]:
        paths=[folder/(name+'_predictions.csv') for name in (base,new)]
        paths=[p if p.exists() else p.with_suffix('.csv.gz') for p in paths]
        if not all(p.exists() for p in paths):continue
        a,b=[pd.read_csv(p).rename(columns={'y':'y_used','p':'probability_used'}).set_index('row_index') for p in paths]
        if not a.index.is_unique or set(a.index)!=set(b.index):raise ValueError('Ablation row mismatch')
        b=b.loc[a.index]
        if not np.array_equal(a.y_used,b.y_used):raise ValueError('Ablation label mismatch')
        m=media.loc[a.index]
        for name,mask in [('all',np.ones(len(a),dtype=bool)),('video_present',(m.media_has_video>0).to_numpy()),
                          ('video_absent',(m.media_has_video==0).to_numpy()),('photos_present',(m.media_picture_n>0).to_numpy())]:
            if mask.sum():rows.append(dict(original=base,metadata=new,subset=name,
                **paired(a.probability_used.to_numpy()[mask],b.probability_used.to_numpy()[mask],a.y_used.to_numpy()[mask])))
    pd.DataFrame(rows).to_csv(out/'media_paired_diagnostics.csv',index=False)
    test=pd.read_csv(out/'media_features_test.csv').set_index('row_index')
    drift=[]
    for c in media:
        x,z=media[c],test[c]
        drift.append(dict(feature=c,train_missing=float(x.isna().mean()),test_missing=float(z.isna().mean()),
                          train_mean=float(x.mean()),test_mean=float(z.mean()),train_unique=int(x.nunique())))
    pd.DataFrame(drift).to_csv(out/'media_train_test_summary.csv',index=False)
    cache=out/'selection/native.pkl'
    if cache.exists():
        data=pd.read_pickle(cache);m=media.loc[data['t']].reset_index(drop=True)
        base=data['tr'].select_dtypes(include='number');novelty=[]
        for c in m:
            x=m[c].to_numpy(dtype=float)
            duplicates=[b for b in base if np.array_equal(x,base[b].to_numpy(dtype=float),equal_nan=True)]
            novelty.append(dict(feature=c,constant_in_development=bool(m[c].nunique(dropna=False)<=1),
                                identical_baseline_columns='|'.join(duplicates)))
        pd.DataFrame(novelty).to_csv(out/'media_feature_novelty.csv',index=False)
    (out/'media_diagnostic_protocol.json').write_text(json.dumps(dict(
        thresholds_fixed=.5,unit='same labeled row in frozen selection split',
        p_values='exploratory paired exact binomial; not corrected for multiple comparisons',
        video_present='historical video_id presence; no decoded video frames',
        photos_present='historical photo metadata; no decoded images',
        diagnostic_pairs=len(rows),drift_features=len(drift)),indent=2))
    print(json.dumps({'paired_rows':len(rows),'drift_features':len(drift)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/round_v4'));run(p.parse_args().output)
