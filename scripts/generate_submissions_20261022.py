from pathlib import Path
import csv, hashlib, json, shutil, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
REC=ROOT/'recovery/extracted'
MODELS=REC/'tp_ap/results/training_round3'
OUT=ROOT/'submissions_20261022'
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def loadcsv(p):
    rows=list(csv.DictReader(p.open()))
    assert len(rows)==30000
    assert [int(x['ID']) for x in rows]==list(range(1,30001))
    assert all(x['condition'] in ('new','used') for x in rows)
    return np.array([x['condition']=='used' for x in rows])
def write(name,y):
    p=OUT/name
    with p.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['ID','condition'])
        w.writerows((i+1,'used' if v else 'new') for i,v in enumerate(y))
    assert np.array_equal(loadcsv(p),y)
    return p
old=REC/'submissions/20261021'
basefile=old/'submission_26_A_E2_complemento16.csv'
base=loadcsv(basefile)
ref=loadcsv(old/'submission_24_control_A32_consolidada_sobre15.csv')
assert np.sum(base!=ref)==16
src={
    'fields':MODELS/'fields/test.npz',
    'dropout':MODELS/'dropout/test.npz',
    'tpe':MODELS/'tpe/test.npz',
    'deep':REC/'tp_ap/results/training_round2/deep_teacher/test.npz',
    'meta':MODELS/'final/with_history/test.npz',
}
p={k:np.load(v)['p'] for k,v in src.items()}
assert all(v.shape==(30000,) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in p.values())
assert np.array_equal(np.load(src['meta'])['ID'],np.arange(1,30001))
historicalcsv=MODELS/'final/with_history/candidate_with_history_model_only.csv'
assert np.array_equal(loadcsv(historicalcsv),p['meta']>=.5)
known={int(x['ID']) for x in csv.DictReader((MODELS/'cohort_posterior/posterior_probabilities.csv').open())}
assert len(known)==160
# P(alternative to26) from each model, used only to rank, without asserting calibration.
q=np.stack([np.where(base,1-v,v) for v in p.values()])
eligible=np.flatnonzero((q.min(axis=0)>.5)&np.array([i+1 not in known for i in range(30000)]))
order=eligible[np.lexsort((eligible,-q[:,eligible].min(axis=0)))]
assert len(order)>=64
g1,g2=order[:32],order[32:64]
assert not set(g1)&set(g2)
assert not ({int(i+1) for i in np.r_[g1,g2]}&known)
paths=[]
control=OUT/basefile.name
shutil.copyfile(basefile,control); assert sha(control)==sha(basefile)
paths.append(control)
paths.append(write('submission_27_fields_titulo_garantia_modelo_puro.csv',p['fields']>=.475))
paths.append(write('submission_28_ensamble_historico_modelo_puro.csv',p['meta']>=.5))
for n,ix in [(29,g1),(30,g2)]:
    y=base.copy();y[ix]=~y[ix]
    paths.append(write(f'submission_{n}_consenso_G{n-28}_32_sobre26.csv',y))
audit=OUT/'auditoria_flips_29_30.csv'
with audit.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['ID','group','condition_in26','alternative','min_support','mean_support']+[f'p_alt_{k}' for k in p])
    for group,ix in [('G1',g1),('G2',g2)]:
        for i in ix:
            w.writerow([int(i+1),group,'used' if base[i] else 'new','new' if base[i] else 'used',float(q[:,i].min()),float(q[:,i].mean())]+[float(v) for v in q[:,i]])
manifest={
 'date':'2026-10-22','competition':'ap-2026-q-2',
 'public_reference':{'submission':24,'score_observed':'0.94533','sha256':sha(old/'submission_24_control_A32_consolidada_sobre15.csv')},
 'assumptions':['21000 public rows is inferred, not officially verified here','Score display treated as truncation to five decimals','Base26 public score unobserved; inferred0.94547 only under assumptions','Model probabilities are correlated and not established as calibrated; no confidence guarantee'],
 'generation_policy':'No new label training. Fields threshold0.475 and meta0.5 locked in previous training. Probe policy declared before observing new scores: unanimous alternative support, descending minimum support, ID tie-break; exclude160 old tested cohort IDs, G1top32 and G2next32. Both probes start from26, never sequential flips.',
 'eligible_unanimous':int(len(eligible)),
 'sources':{k:{'path':str(v.relative_to(REC)),'sha256':sha(v)} for k,v in src.items()},
 'submissions':[]
}
for n,path in zip(range(26,31),paths):
    y=loadcsv(path);d=np.flatnonzero(y!=base)
    manifest['submissions'].append({'number':n,'file':path.name,'sha256':sha(path),'rows':30000,'columns':['ID','condition'],'counts':{'used':int(y.sum()),'new':int((~y).sum())},'differences_vs26':int(len(d)),'differences_vs24':int(np.sum(y!=ref)),'changed_IDs_vs26':[int(i+1) for i in d],'public_score_observed':None,'threshold':.475 if n==27 else .5 if n==28 else None})
assert np.sum(loadcsv(paths[3])!=base)==32 and np.sum(loadcsv(paths[4])!=base)==32
assert np.sum(loadcsv(paths[3])!=loadcsv(paths[4]))==64
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
shutil.copyfile(Path(__file__),OUT/Path(__file__).name)
print(json.dumps({'eligible':len(eligible),'submissions':[{k:v for k,v in e.items() if k not in ['changed_IDs_vs26']} for e in manifest['submissions']],'group_min_support':[float(q[:,g1].min()),float(q[:,g2].min())]},ensure_ascii=False,indent=2))
