"""Verify actual experiment artifacts without training or network requests."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
from finalize_round4 import aligned_predictions,load_frozen,row_hash,score
ROOT=Path(__file__).resolve().parents[1]
def verify(args):
    out=Path(args.output);reference=json.loads((ROOT/'results/round_v3/prior_prediction_metadata.json').read_text())
    checks=[]
    for filename,key in [('train_data.jsonlines','train_sha256'),('test_data.jsonlines','test_sha256')]:
        with (Path(args.data_dir)/filename).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==reference[key],filename+' changed';checks.append(filename+': SHA256')
    for split,n in [('train',70000),('test',30000)]:
        m=pd.read_csv(out/('media_features_'+split+'.csv'))
        assert len(m)==n and np.array_equal(m.row_index,np.arange(n))
        columns=[c for c in m if c!='row_index'];assert len(columns)==50 and all(c.startswith('media_') for c in columns)
        assert not np.isinf(m[columns].to_numpy()).any();checks.append(split+': 50 metadata features and exact row order')
    e=pd.read_csv(out/'selection/evaluation_rows.csv');rows,y=e.row_index.to_numpy(),e.y_used.to_numpy()
    assert len(rows)==21000 and len(set(rows))==21000
    baseline=aligned_predictions(ROOT/'optimization/holdout_predictions.csv',rows,y)
    assert np.isclose(score(y,baseline,.5)['accuracy'],.9420952380952381)
    checks.append('original baseline: 21k row/label alignment and recorded accuracy')
    seen=set();models=0
    for path in out.rglob('r4_*.json'):
        cfg=json.loads(path.read_text());name=cfg.get('name')
        if not name or cfg.get('family') not in ('cat','lgb') or name in seen:continue
        p=aligned_predictions(path.with_name(name+'_predictions.csv'),rows,y)
        assert np.isclose(score(y,p,.5)['accuracy'],cfg['accuracy']),name+' recorded accuracy mismatch'
        assert not any(cfg.get(k,False) for k in ['uses_pixels','uses_video_frames','image_pixels_used','video_frames_used'])
        if cfg.get('fingerprint'):assert cfg['fingerprint']['train_sha256']==reference['train_sha256']
        seen.add(name);models+=1
    checks.append(str(models)+' completed models: probabilities, labels, metrics and input claims')
    if (out/'round4_selected.json').exists():
        frozen=load_frozen(out/'round4_selected.json');p=aligned_predictions(out/'round4_selection_predictions.csv',rows,y)
        assert row_hash(rows)==frozen['selection_row_sha256']
        assert np.isclose(score(y,p,frozen['threshold'])['accuracy'],frozen['selection_metrics']['accuracy'])
        checks.append('frozen selection: digest, row hash and threshold')
    if (out/'round4_confirmation.json').exists():
        c=json.loads((out/'round4_confirmation.json').read_text());ce=pd.read_csv(out/'confirmation/evaluation_rows.csv')
        cr,cy=ce.row_index.to_numpy(),ce.y_used.to_numpy();assert len(cr)==14000
        p=aligned_predictions(out/'round4_confirmation_predictions.csv',cr,cy)
        bp=aligned_predictions(ROOT/'results/round_v3/prior_confirmation_predictions.csv',cr,cy)
        candidate=score(cy,p,frozen['threshold'])['accuracy'];prior=score(cy,bp,.5)['accuracy']
        assert np.isclose(candidate,c['candidate']['accuracy']) and np.isclose(prior,c['baseline']['accuracy'])
        assert c['promote']==bool(candidate>prior and frozen['selection_metrics']['accuracy']>frozen['selection_baseline']['accuracy'])
        checks.append('alternate grouped validation: aligned metrics and promotion decision')
    if (out/'round4_refit.json').exists():
        r=json.loads((out/'round4_refit.json').read_text());submission=Path(r['submission'])
        if not submission.is_absolute():submission=ROOT/submission
        s=pd.read_csv(submission)
        assert c['promote'] and r['selection_digest']==frozen['selection_digest']
        assert list(s)==['ID','condition'] and len(s)==30000 and np.array_equal(s.ID,np.arange(1,30001))
        assert set(s.condition)<=set(['new','used'])
        assert hashlib.sha256(submission.read_bytes()).hexdigest()==r['submission_sha256']
        checks.append('submission: authorized promotion, 30k IDs, labels and SHA256')
    result=dict(checks=checks,completed_models=models,all_passed=True,training_executed=False,network_requests=0)
    (out/'artifact_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v4'));verify(p.parse_args())
