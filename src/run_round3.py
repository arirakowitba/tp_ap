"""Sequential, resumable orchestration with one enforced wall-clock budget."""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def run(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.time();deadline=started+args.minutes*60
    # Reserve one third of the budget, at most30min, for frozen confirmation/refit.
    search_deadline=deadline-min(30,args.minutes/3)*60
    completed=[];failed=[]
    def stage(name,extra,limit=deadline):
        seconds=limit-time.time()
        if seconds<=0:raise TimeoutError('Global stage budget exhausted')
        command=[sys.executable,'-u',str(ROOT/'src'/name),*extra]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        log=out/(Path(name).stem+'_master.log')
        print('RUN',name,round(seconds),'seconds available',flush=True)
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:status=process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise TimeoutError('Deadline reached in '+name)
        if status:
            failed.append(dict(script=name,exit_code=status,log=str(log.relative_to(ROOT))))
            raise RuntimeError(log.read_text()[-4000:])
        completed.append(name)
    shared=['--data-dir',str(Path(args.data_dir).resolve()),'--output',str(out),'--deadline',str(search_deadline)]
    cache=out/'cache_base';model=Path(args.model_dir).resolve()
    try:
        stage('prepare_round3.py',['--data-dir',str(Path(args.data_dir).resolve()),'--output',str(cache),'--deadline',str(search_deadline)],search_deadline)
        stage('normalize_domain_round.py',shared+['--cache',str(cache)],search_deadline)
        iso=datetime.datetime.fromtimestamp(search_deadline,datetime.timezone.utc).isoformat()
        stage('category_experts_round.py',['--data-dir',str(Path(args.data_dir).resolve()),'--output',str(out),'--cache',str(cache/'matrices.pkl'),'--deadline',iso],search_deadline)
        stage('learned_stacking_round.py',shared+['--no-confirm'],search_deadline)
        stage('probe_image_access.py',['--data-dir',str(Path(args.data_dir).resolve()),'--output',str(out/'image_access_probe.json')],search_deadline)
        if importlib.util.find_spec('openvino') and importlib.util.find_spec('tokenizers'):
            stage('download_embedding_model.py',['--model-dir',str(model)],search_deadline)
            stage('embedding_round.py',shared+['--model-dir',str(model),'--cache',str(cache)],search_deadline)
        else:
            failed.append(dict(script='embedding_round.py',reason='Install requirements_round3.txt to enable the pretrained CPU backend'))
        if (ROOT/'src/learned_stacking4_round.py').exists():stage('learned_stacking4_round.py',shared+['--no-confirm'],search_deadline)
        if (ROOT/'src/learned_stacking5_round.py').exists() and (out/'embedding_titles.npy').exists():stage('learned_stacking5_round.py',shared+['--no-confirm'],search_deadline)
        combined=shared+['--cache',str(cache),'--normalize','--trials',str(args.trials)]
        if (out/'embedding_titles.npy').exists():combined+=['--embedding']
        stage('combined_round3_search.py',combined,search_deadline)
        final=['--data-dir',str(Path(args.data_dir).resolve()),'--output',str(out),'--deadline',str(deadline),'--model-dir',str(model)]
        stage('finalize_round3.py',final+['--mode','select'])
        stage('finalize_round3.py',final+['--mode','confirm'])
        selected=json.loads((out/'round3_selected.json').read_text())
        if selected.get('promote_over_previous'):stage('finalize_round3.py',final+['--mode','refit'])
    except (TimeoutError,RuntimeError) as exc:
        failed.append(dict(reason=str(exc)));print(str(exc),flush=True)
    finally:
        status=dict(started_utc=datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),
                    deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
                    ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-started,
                    budget_minutes=args.minutes,completed_stages=completed,incomplete_stages=failed,
                    public_reference_user_report=.94428,status='stopped')
        (out/'master_execution.json').write_text(json.dumps(status,indent=2))
    return 1 if failed else 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v3'));p.add_argument('--model-dir',default=str(ROOT.parent/'models/title_minilm'));p.add_argument('--minutes',type=float,default=90);p.add_argument('--trials',type=int,default=12);args=p.parse_args()
    raise SystemExit(run(args))
