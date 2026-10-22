from pathlib import Path
import json,re,collections
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/training_round2/relations';O.mkdir(parents=True,exist_ok=True)
a=[];keys=collections.defaultdict(lambda: [0,0])
for line in open(R.parent/'recovered/data/train_data.jsonlines'):
 x=json.loads(line);y=int(x['condition']=='used');title=re.sub(r'\s+',' ',str(x.get('title') or '').lower()).strip();q=float(x.get('initial_quantity') or 0);lst=str(x.get('listing_type_id',''));w=str(x.get('warranty') or '').lower();key=(str(x['seller_id']),str(x.get('category_id')),title,float(x.get('price') or 0),q,lst,w)
 keys[key][y]+=1;a.append({'used':y,'listing':lst,'qty1':q<=1,'category':x.get('category_id'),'seller':str(x['seller_id']),'mentions_new':bool(re.search(r'\b(nuevo|nueva|sin uso|a estrenar)\b',title)),'mentions_used':bool(re.search(r'\b(usado|usada|segunda mano)\b',title)),'warranty_missing':not w,'n_pictures':len(x.get('pictures') or [])})
d=pd.DataFrame(a);z=d.groupby(['listing','qty1']).used.agg(n='size',used_rate='mean').reset_index();z.to_csv(O/'listing_quantity_target.csv',index=False)
t=d.groupby(['mentions_new','mentions_used']).used.agg(n='size',used_rate='mean').reset_index();t.to_csv(O/'title_semantics.csv',index=False)
mixed=[c for c in keys.values() if min(c)>0];o={'n_train':len(d),'exact_context_groups':len(keys),'mixed_context_groups':len(mixed),'rows_in_mixed_contexts':sum(sum(c) for c in mixed),'minority_labels_in_mixed_contexts':sum(min(c) for c in mixed),'interpretation':'These are collisions under selected observed fields, not proof of noisy labels or an irreducible error ceiling; omitted attributes may distinguish rows.'};(O/'summary.json').write_text(json.dumps(o,indent=2));print(json.dumps(o));print(z.to_string(index=False));print(t.to_string(index=False))
