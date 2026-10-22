from pathlib import Path
import json,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'results/training_next';O.mkdir(exist_ok=True,parents=True)
t=pd.read_json(R.parent/'recovered/data/train_data.jsonlines',lines=True)
a=np.load(R/'results/new_models_20261021/seller_native_confirm.npz');b=np.load(R/'results/new_models_20261021/seller_free_depthwise_confirm.npz');part=np.load(R/'results/new_models_20261021/partitions.npz')
y=t.condition.eq('used').astype(int); rows=a['rows'];dev=np.concatenate([part['fit'],part['tune']]);d=t.iloc[dev].copy();d['y']=y.iloc[dev].values
v=t.iloc[rows].copy();v['y']=a['y'];v['p']=a['p'];v['error']=(a['p']>=.5)!=a['y'];v['free_error']=(b['p']>=.5)!=a['y'];v['seen']=a['seen'];v['conf']=np.maximum(a['p'],1-a['p'])
s=d.groupby('seller_id').y.agg(['mean','count']);v['seller_train_n']=v.seller_id.map(s['count']).fillna(0);v['seller_train_used_rate']=v.seller_id.map(s['mean']);v['seller_label_mix']=pd.cut(v.seller_train_used_rate,[-.01,0,.1,.9,.999,1.01],labels=['pure_new','mostly_new','mixed','mostly_used','pure_used']).astype(str)
v['seller_size']=pd.cut(v.seller_train_n,[-1,0,1,3,10,100,100000],labels=['unseen','1','2-3','4-10','11-100','100+'])
v['warranty_present']=v.warranty.fillna('').str.len()>0;v['quantity']=pd.cut(v.initial_quantity,[-1,1,2,10,100,1e10],labels=['1','2','3-10','11-100','100+']);v['conf_bin']=pd.cut(v.conf,[.499,.6,.7,.8,.9,.95,.99,1.01]);v['title_used']=v.title.str.lower().str.contains(r'\busad[oa]s?\b|segunda mano',regex=True)
outs=[]
for col in ['seen','seller_size','seller_label_mix','warranty_present','quantity','conf_bin','title_used','listing_type_id','buying_mode']:
 for key,q in v.groupby(col,observed=True,dropna=False):
  outs.append({'slice':col,'value':str(key),'n':len(q),'errors':int(q.error.sum()),'accuracy':float(1-q.error.mean()),'seller_free_accuracy':float(1-q.free_error.mean()),'used_rate':float(q.y.mean()),'mean_confidence':float(q.conf.mean()),'free_fixes_native':int((q.error&~q.free_error).sum()),'free_breaks_native':int((~q.error&q.free_error).sum())})
pd.DataFrame(outs).to_csv(O/'confirmation_error_slices.csv',index=False)
cat=v.groupby('category_id').agg(n=('y','size'),errors=('error','sum'),used_rate=('y','mean'));cat['error_rate']=cat.errors/cat.n;cat.sort_values('errors',ascending=False).to_csv(O/'confirmation_category_errors.csv')
v[['title','category_id','initial_quantity','warranty','seller_train_n','seller_train_used_rate','y','p','conf']].loc[v.error].sort_values('conf',ascending=False).head(120).to_csv(O/'confirmation_high_confidence_errors.csv',index=False)
# Cohorts are diagnostics of aggregate leaderboard effects, not rowwise labels.
records=[]
for filename in ['cohorte_16_auditoria.csv','cohorte_C_auditoria.csv','cohorte_D_auditoria.csv','cohorte_E_auditoria.csv']:
 q=pd.read_csv(R/'submissions/20261020'/filename)
 if filename.startswith('cohorte_16'):q['group']=np.where(q.full70k_seed_votes_alternative_out_of_7>=4,'A','B')
 else:q['group']=q['subgroup'] if 'subgroup' in q else filename
 for name,z in q.groupby('group'):
  rec={'group':name,'n':len(z)}
  for c in z.select_dtypes('number'):
   if c not in ['ID','seller_id']:rec[c+'_mean']=float(z[c].mean())
  records.append(rec)
pd.DataFrame(records).to_csv(O/'flip_group_features.csv',index=False)
print(pd.DataFrame(outs).to_string(index=False))
