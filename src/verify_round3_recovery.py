"""Lightweight checks for recovery and cache fallback; never trains a model."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json,time
import numpy as np,pandas as pd
import finalize_round3 as final
import learned_stacking_round as base3
import learned_stacking4_round as base4
import learned_stacking5_round as base5
import run_round3
from types import SimpleNamespace
import subprocess

def main():
    checks=[]
    prior=pd.read_csv(final.ROOT/'results/round_v3/prior_confirmation_predictions.csv')
    assert len(prior)==14000 and prior.row_index.is_unique
    assert abs(((prior.probability_used>=.5)==prior.y_used).mean()-.9421428571428572)<1e-12
    full=final.prior_probability('full_train');assert len(full)==30000 and np.isfinite(full).all()
    checks.append('Frozen baseline14k accuracy and30k public-submission agreement')
    n=14006;ids=np.arange(n);raw=pd.DataFrame({'id':ids,'parent_item_id':None,'condition':np.where(ids%2,'used','new')})
    t=ids[[2,4,6,8,10,12]];v=np.setdiff1d(ids,t);y=(raw.condition=='used').to_numpy(dtype='int8');g=final.groups_for(raw)
    original=[base3.partition,base4.partition,base5.partition]
    for count in [3,4,5]:
        calls=[]
        def fake(number):
            def run(frame,yy,gg,tt,vv,out,seed,deadline):
                assert np.array_equal(frame.id,ids) and np.array_equal(yy,y) and np.array_equal(gg,g)
                assert np.array_equal(tt,t) and np.array_equal(vv,v)
                calls.append(number);return np.full((len(t),number),.5),np.full((len(v),number),.5)
            return run
        base3.partition=fake(3);base4.partition=fake(4);base5.partition=fake(5)
        with TemporaryDirectory() as tmp:
            out=Path(tmp)/'confirmation';out.mkdir()
            pair=(np.zeros((len(t),384),dtype='float32'),np.zeros((len(v),384),dtype='float32')) if count==5 else None
            a,b=final.stack_features(raw.iloc[t].reset_index(drop=True),y[t],g[t],raw.iloc[v].reset_index(drop=True),out,t,v,time.time()+30,count,pair)
            assert a.shape==(len(t),count) and b.shape==(len(v),count)
            assert calls==([3,3] if count==3 else [3,4,4] if count==4 else [3,4,5])
            if count==5:assert np.load(out/'embedding_titles.npy').shape==(n,384)
        checks.append(f'{count}-base fallback preserves global row order, evaluation labels and grouped folds')
    base3.partition,base4.partition,base5.partition=original
    fit=raw.iloc[t].reset_index(drop=True);val=raw.iloc[v].reset_index(drop=True)
    key=final.cache_identity(fit,y[t],g[t],val)
    assert key!=final.cache_identity(fit.iloc[::-1].reset_index(drop=True),y[t][::-1],g[t][::-1],val)
    checks.append('Cache fingerprint rejects row reordering')
    # A search timeout must still leave the reserved confirmation stage usable.
    process_class=run_round3.subprocess.Popen;commands=[]
    class FakeProcess:
        def __init__(self,command,**kwargs):self.command=command;self.waits=0;commands.append(command)
        def wait(self,timeout=None):
            self.waits+=1
            if self.command[2].endswith('prepare_round3.py') and self.waits==1:raise subprocess.TimeoutExpired(self.command,timeout)
            return 0
        def terminate(self):pass
        def kill(self):pass
    run_round3.subprocess.Popen=FakeProcess
    try:
        with TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'round3_selected.json').write_text('{"promote_over_previous":false}')
            status=run_round3.run(SimpleNamespace(output=str(out),minutes=.1,data_dir=tmp,model_dir=tmp,trials=1))
            state=json.loads((out/'master_execution.json').read_text())
            assert status==1 and state['completed_stages']==['finalize_round3.py','finalize_round3.py']
            assert commands[-2][-1]=='select' and commands[-1][-1]=='confirm'
    finally:run_round3.subprocess.Popen=process_class
    checks.append('Search timeout preserves reserved selection and confirmation stages')
    (final.ROOT/'results/round_v3/recovery_verification.json').write_text(json.dumps({'checks_passed':checks,'models_trained':0},indent=2));print(checks)

if __name__=='__main__':main()
