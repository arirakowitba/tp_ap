"""Bounded, equal-capacity ablations of four media metadata blocks."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def run(args):
    stages=[];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for group in ('image','video','orientation','dimensions'):
        folder=out/group;folder.mkdir(exist_ok=True)
        cfg=folder/('r4_lgb_fixed430_media_'+group+'.json')
        if cfg.exists():status='cached'
        elif args.deadline-time.time()<30:status='not_started_deadline'
        else:
            command=[sys.executable,'-u',str(Path(__file__).with_name('multimodal_lgb_round4.py')),
                '--cache',args.cache,'--media-features',args.media_features,'--output',str(folder),
                '--fixed-only','--media-group',group,'--deadline',str(args.deadline)]
            with (folder/'run.log').open('w') as log:
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:status='completed' if p.wait(timeout=max(1,args.deadline-time.time()-3))==0 else 'failed'
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=2)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    status='terminated_deadline'
        stages.append(dict(group=group,status=status));(out/'execution.json').write_text(json.dumps(stages,indent=2))
        print(group,status,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--media-features',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline',type=float,required=True);run(p.parse_args())
