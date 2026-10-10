"""Adaptive local experiment sweep. No Kaggle credentials or submissions sent."""
import os,sys,gc,json,time,re,unicodedata,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,log_loss
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'optimization';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from ap2026q2_advanced_optimized import build_advanced_features
from ap2026q2_deep_features import build_deep_features,parse_nested

def split(y,groups,seed=42,n_holdout=3):
 f=list(StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))
 v=np.concatenate([f[i][1] for i in range(n_holdout)]);t=np.setdiff1d(np.arange(len(y)),v)
 assert not set(groups[t]).intersection(groups[v]);return t,v

def enrich(raw,attr_ids=None):
 x,attr_ids=build_advanced_features(raw,attribute_ids=attr_ids,top_k=100)
 x=build_deep_features(raw,x,include_post_listing=True)
 title=raw.title.fillna('').astype(str).str.lower()
 warranty=raw.warranty.fillna('').astype(str).str.lower()
 x['full_text']=(title+' warranty '+warranty+' attributes '+x.attributes_text).str.replace(r'[^\w\s]',' ',regex=True).str.replace(r'\s+',' ',regex=True).str.strip().replace('','empty')
 x['warranty_text']=warranty.replace('','__MISSING__')
 for pattern,name in [(r'sin garant|no.*garant','none'),(r'f[aá]brica|fabricante','factory'),(r'\bmes|\baño|\bd[ií]a','duration'),(r'oficial','official'),(r'\busad|sin uso|nuevo','condition_word')]:x['warranty_'+name]=warranty.str.contains(pattern,regex=True).astype('int8')
 x['warranty_chars']=warranty.str.len().astype('int16')
 x['title_first']=title.str.split().str[0].fillna('__MISSING__')
 x['title_first2']=title.str.split().map(lambda z:' '.join(z[:2]))
 for col in ['tags','sub_status','deal_ids','coverage_areas']:
  x[col+'_joined']=raw[col].map(lambda z:'|'.join(sorted(map(str,parse_nested(z) or [])))).replace('','__MISSING__')
 for col in ['accepts_mercadopago','automatic_relist']:
  x[col]=raw[col].astype('string').fillna('__MISSING__')
 for part in ['state','city']:
  x['address_'+part+'_name']=raw.seller_address.map(lambda z:(z or {}).get(part,{}).get('name','__MISSING__')).astype(str)
 x['shipping_tags_joined']=raw.shipping.map(lambda z:'|'.join(sorted(map(str,(parse_nested(z) or {}).get('tags') or [])))).replace('','__MISSING__')
 for col in ['date_created','start_time','last_updated','stop_time']:
  dt=pd.to_datetime(raw[col],errors='coerce',utc=True)
  x[col+'_days_epoch']=(dt-pd.Timestamp('2010-01-01',tz='UTC')).dt.total_seconds()/86400
 for col in ['price','base_price','original_price','initial_quantity','available_quantity','sold_quantity']:
  x[col+'_raw']=pd.to_numeric(raw[col],errors='coerce')
 for a,b,name in [('price','initial_quantity','price_per_initial'),('sold_quantity','available_quantity','sold_per_available')]:
  x[name]=pd.to_numeric(raw[a],errors='coerce')/(1+pd.to_numeric(raw[b],errors='coerce').clip(lower=0))
 x['image_aspect']=x.deep_picture_width_mean/(x.deep_picture_height_mean.replace(0,np.nan))
 x['image_max_aspect']=x.deep_picture_max_width_mean/(x.deep_picture_max_height_mean.replace(0,np.nan))
 return x,attr_ids

def data_folder():
 override=os.getenv('AP2026Q2_DATA_DIR')
 if override:
  folder=Path(override).expanduser()
  if (folder/'train_data.jsonlines').exists() and (folder/'test_data.jsonlines').exists():return folder
 from ap2026q2_common import download_competition
 return Path(download_competition())

def prepare():
 cache=OUT/'feature_cache.pkl'
 if cache.exists():return pd.read_pickle(cache)
 folder=data_folder();raw=pd.read_json(folder/'train_data.jsonlines',lines=True)
 y=(raw.condition=='used').astype('int8').to_numpy()
 groups=raw.parent_item_id.astype('string').fillna('__ROW__'+raw.id.astype(str)).to_numpy()
 t,v=split(y,groups)
 print('BUILD FEATURES',len(t),len(v),flush=True)
 xf,ids=enrich(raw.iloc[t].reset_index(drop=True));xv,_=enrich(raw.iloc[v].reset_index(drop=True),ids)
 x=pd.concat([xf.set_axis(t),xv.set_axis(v)]).sort_index().reset_index(drop=True)
 result={'x':x,'y':y,'groups':groups,'t':t,'v':v,'attr_ids':ids}
 pd.to_pickle(result,cache);print('FEATURES',x.shape,flush=True);return result

def record(name,p,y,protocol='selection_seed42',**extra):
 rows_path=OUT/'metrics.csv';rows=pd.read_csv(rows_path).to_dict('records') if rows_path.exists() else []
 r=dict(model=name,protocol=protocol,accuracy=accuracy_score(y,p>=.5),balanced_accuracy=balanced_accuracy_score(y,p>=.5),macro_f1=f1_score(y,p>=.5,average='macro'),logloss=log_loss(y,p),**extra)
 rows=[z for z in rows if not(z['model']==name and z['protocol']==protocol)]+[r]
 pd.DataFrame(rows).to_csv(rows_path,index=False);np.save(OUT/(name+'_'+protocol+'.npy'),p)
 print('RESULT',json.dumps(r),flush=True);return r

def numeric_text(x,t,v):
 tf=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=90000,strip_accents='unicode',sublinear_tf=True,dtype=np.float32)
 a=tf.fit_transform(x.full_text.iloc[t]);b=tf.transform(x.full_text.iloc[v])
 svd=TruncatedSVD(n_components=96,n_iter=5,random_state=42)
 da=svd.fit_transform(a).astype('float32');db=svd.transform(b).astype('float32')
 tr=x.iloc[t].drop(columns=['title_text','attributes_text','full_text']).reset_index(drop=True).copy()
 va=x.iloc[v].drop(columns=['title_text','attributes_text','full_text']).reset_index(drop=True).copy()
 tr=pd.concat([tr,pd.DataFrame(da,columns=[f'svd_{i}' for i in range(96)])],axis=1)
 va=pd.concat([va,pd.DataFrame(db,columns=[f'svd_{i}' for i in range(96)])],axis=1)
 cats=tr.select_dtypes(include=['object','string','category']).columns.tolist()
 for c in cats:tr[c]=tr[c].astype('string').fillna('__MISSING__').astype(str);va[c]=va[c].astype('string').fillna('__MISSING__').astype(str)
 for c in tr.columns.difference(cats):tr[c]=tr[c].replace([np.inf,-np.inf],np.nan);va[c]=va[c].replace([np.inf,-np.inf],np.nan)
 return tr,va,cats,tf,svd

def sweep():
 data=prepare();x,y,t,v=data['x'],data['y'],data['t'],data['v'];gc.collect()
 tr,va,cats,tf,svd=numeric_text(x,t,v)
 configs=[('cat_enriched_d8',dict(depth=8,iterations=1800,learning_rate=.035,l2_leaf_reg=8,random_seed=51)),('cat_enriched_d6',dict(depth=6,iterations=2000,learning_rate=.04,l2_leaf_reg=5,random_seed=42)),('cat_enriched_d9',dict(depth=9,iterations=1500,learning_rate=.035,l2_leaf_reg=12,random_seed=73)),('cat_native_text_d7',dict(depth=7,iterations=1400,learning_rate=.04,l2_leaf_reg=8,random_seed=51))]
 for name,params in configs:
  if (OUT/(name+'_selection_seed42.npy')).exists():continue
  a,b=tr.copy(),va.copy();texts=[]
  if 'native' in name:a['full_text']=x.full_text.iloc[t].to_numpy();b['full_text']=x.full_text.iloc[v].to_numpy();texts=['full_text']
  model=CatBoostClassifier(**params,loss_function='Logloss',eval_metric='Accuracy',thread_count=4,allow_writing_files=False,early_stopping_rounds=130,one_hot_max_size=10,random_strength=.5)
  ts=time.time();model.fit(a,y[t],cat_features=cats,text_features=texts,eval_set=(b,y[v]),verbose=200)
  record(name,model.predict_proba(b)[:,1],y[v],seconds=time.time()-ts,best_iteration=model.best_iteration_)
  model.save_model(str(OUT/(name+'.cbm')))
  (OUT/(name+'_params.json')).write_text(json.dumps(params));del a,b,model;gc.collect()
 del tr,va;gc.collect()
 # A complementary linear model: word/character text + one-hot IDs and numeric features.
 name='logistic_word_char_tabular'
 if not (OUT/(name+'_selection_seed42.npy')).exists():
  word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=120000,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
  char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=140000,sublinear_tf=True,dtype=np.float32)
  columns=['seller_id','category_id','listing_type_id','shipping_mode','seller_state','title_first','address_city_name']
  enc=OneHotEncoder(handle_unknown='ignore',min_frequency=3,dtype=np.float32)
  nums=x.select_dtypes(include=np.number).columns
  imp=SimpleImputer(strategy='median',add_indicator=True);scale=StandardScaler()
  nd=scale.fit_transform(imp.fit_transform(x.iloc[t][nums].replace([np.inf,-np.inf],np.nan)));nv=scale.transform(imp.transform(x.iloc[v][nums].replace([np.inf,-np.inf],np.nan)))
  a=sparse.hstack([word.fit_transform(x.full_text.iloc[t]),char.fit_transform(x.full_text.iloc[t]),enc.fit_transform(x.iloc[t][columns].astype(str)),sparse.csr_matrix(nd,dtype=np.float32)],format='csr')
  b=sparse.hstack([word.transform(x.full_text.iloc[v]),char.transform(x.full_text.iloc[v]),enc.transform(x.iloc[v][columns].astype(str)),sparse.csr_matrix(nv,dtype=np.float32)],format='csr')
  for C in [1.,4.]:
   if (OUT/(name+f'_C{C:g}'+'_selection_seed42.npy')).exists():continue
   model=LogisticRegression(C=C,solver='liblinear',max_iter=350);ts=time.time();model.fit(a,y[t]);record(name+f'_C{C:g}',model.predict_proba(b)[:,1],y[v],seconds=time.time()-ts)
  if (OUT/(name+'_C4_selection_seed42.npy')).exists():np.save(OUT/(name+'_selection_seed42.npy'),np.load(OUT/(name+'_C4_selection_seed42.npy')))

if __name__=='__main__':sweep()
