"""Data-backed visual explanations for the Mercado Libre classification case."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay,accuracy_score
from sklearn.calibration import calibration_curve
COLORS={'new':'#227c9d','used':'#f28f3b'}
def style():
    plt.rcParams.update({'figure.figsize':(11,4.5),'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':14,
                         'axes.labelsize':11,'font.size':10,'figure.dpi':110,'axes.grid':True,'grid.alpha':.16,'axes.axisbelow':True})
def show(fig):
    fig.tight_layout();plt.show()
def overview(raw,test):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].bar(['Train','Test'],[len(raw),len(test)],color=['#227c9d','#65758b']);axes[0].set(title='Disponibilidad de datos',ylabel='Publicaciones')
    for i,n in enumerate([len(raw),len(test)]):axes[0].text(i,n+800,f'{n:,}',ha='center')
    counts=raw.condition.value_counts().reindex(['new','used']);axes[1].bar(counts.index,counts,color=list(COLORS.values()));axes[1].set(title='Etiquetas de entrenamiento',ylabel='Publicaciones')
    for i,n in enumerate(counts):axes[1].text(i,n+500,f'{n/len(raw):.1%}',ha='center')
    axes[2].axis('off');axes[2].text(0,.9,'Una fila = una publicación\n\nX: texto, precio, vendedor, logística...\ny: new / used\n\nTest: tiene X, no tiene y\n\nAccuracy = aciertos / total',va='top',fontsize=13)
    show(fig)
def missing(raw):
    values=(raw.isna().mean()*100).sort_values().tail(15)
    fig,ax=plt.subplots(figsize=(11,6));ax.barh(values.index,values,color='#65758b');ax.set(title='Faltantes: ausencia y posible señal',xlabel='% faltante',xlim=(0,105))
    for i,v in enumerate(values):ax.text(v+1,i,f'{v:.1f}%',va='center')
    show(fig)
def distributions(raw):
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for label in ['new','used']:
        subset=raw[raw.condition==label]
        axes[0].hist(np.log1p(subset.price.clip(lower=0)),bins=60,alpha=.5,density=True,color=COLORS[label],label=label)
        axes[1].hist(subset.title.fillna('').str.len(),bins=np.arange(0,151,5),alpha=.5,density=True,color=COLORS[label],label=label)
    axes[0].set(title='Distribución de precios',xlabel='log(1 + precio)',ylabel='Densidad')
    axes[1].set(title='Longitud de título',xlabel='Caracteres',ylabel='Densidad')
    for ax in axes:ax.legend()
    show(fig)
def categories(raw):
    keys=raw.category_id.value_counts().head(14).index;table=pd.crosstab(raw.category_id,raw.condition).reindex(keys)
    fig,axes=plt.subplots(1,2,figsize=(14,5));table[['new','used']].plot.barh(stacked=True,ax=axes[0],color=list(COLORS.values()))
    axes[0].set(title='Categorías más frecuentes',xlabel='Publicaciones')
    (table.used/table.sum(axis=1)*100).plot.barh(ax=axes[1],color=COLORS['used']);axes[1].set(title='Tasa de usados dentro de categoría',xlabel='% usados',xlim=(0,100));show(fig)
def commercial(raw):
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,c in zip(axes,['listing_type_id','accepts_mercadopago']):
        table=pd.crosstab(raw[c].fillna('__MISSING__'),raw.condition,normalize='index')
        table[['new','used']].plot.bar(stacked=True,ax=ax,color=list(COLORS.values()));ax.set(title=c,ylabel='Proporción dentro de cada valor',ylim=(0,1));ax.tick_params(axis='x',rotation=25)
    show(fig)
def families(raw,t,v):
    from competition_pipeline_v2 import groups_for
    groups=groups_for(raw);sizes=pd.Series(groups).value_counts()
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    table=pd.cut(sizes,[0,1,2,5,10,np.inf],labels=['1','2','3–5','6–10','11+']).value_counts(sort=False)
    axes[0].bar(table.index.astype(str),table,color='#65758b');axes[0].set(title='Tamaño de familia',xlabel='Publicaciones por familia',ylabel='Familias')
    axes[1].bar(['Desarrollo','Validación'],[len(t),len(v)],color=list(COLORS.values()));axes[1].set_title('Partición agrupada de la ronda nueva')
    for i,n in enumerate([len(t),len(v)]):axes[1].text(i,n+400,f'{n:,}',ha='center')
    overlap=len(set(groups[t])&set(groups[v]));axes[2].axis('off');axes[2].text(.05,.8,f'{overlap} familias compartidas',fontsize=18,color='#227c9d')
    axes[2].text(.05,.55,'Publicaciones hermanas quedan juntas.\nEvita evaluar memorización\nde variantes del mismo producto.',fontsize=12,va='top');show(fig)
def signals(raw):
    title=raw.title.fillna('').str.lower()
    patterns={'Usado / segunda mano':r'\busad[oa]s?\b|segunda mano','Nuevo / estrenar':r'\bnuev[oa]s?\b|a estrenar',
              'Como nuevo / impecable':r'como nuev|impecable','Reparar / repuestos':r'reparar|repuesto|no funciona',
              'Mayorista / stock':r'mayorista|stock|por mayor','Sin uso / sellado':r'sin uso|sellad'}
    rows=[]
    for name,pattern in patterns.items():
        mask=title.str.contains(pattern,regex=True);rows.append((name,int(mask.sum()),float((raw.loc[mask,'condition']=='used').mean())))
    table=pd.DataFrame(rows,columns=['señal','n','used_rate']);fig,axes=plt.subplots(1,2,figsize=(14,4.5))
    axes[0].barh(table['señal'],table.n,color='#65758b');axes[0].set_title('Frecuencia de expresiones')
    axes[1].barh(table['señal'],table.used_rate*100,color=COLORS['used']);axes[1].axvline((raw.condition=='used').mean()*100,color='#227c9d',ls='--',label='Tasa global')
    axes[1].set(title='Tasa de usados cuando aparece',xlabel='% usados',xlim=(0,100));axes[1].legend();show(fig);return table
def representation(raw):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    sample=raw.sample(min(2500,len(raw)),random_state=42);tf=TfidfVectorizer(min_df=3,max_features=3000,strip_accents='unicode')
    matrix=tf.fit_transform(sample.title.fillna(''));idx=np.argsort(np.asarray(matrix.sum(axis=0)).ravel())[-18:]
    embedding=TruncatedSVD(2,random_state=42).fit_transform(matrix);fig,axes=plt.subplots(1,2,figsize=(14,5))
    image=axes[0].imshow(matrix[:12,idx].toarray(),aspect='auto',cmap='Blues');axes[0].set_xticks(np.arange(len(idx)),tf.get_feature_names_out()[idx],rotation=65,ha='right')
    axes[0].set(title='TF-IDF: títulos × términos',ylabel='Publicación de muestra');fig.colorbar(image,ax=axes[0],fraction=.03)
    for label in ['new','used']:
        mask=(sample.condition==label).to_numpy();axes[1].scatter(embedding[mask,0],embedding[mask,1],s=9,alpha=.3,color=COLORS[label],label=label)
    axes[1].set(title='SVD de dos componentes: ilustración',xlabel='Componente 1',ylabel='Componente 2');axes[1].legend();show(fig)
def historical(root):
    table=pd.read_csv(Path(root)/'optimization/metrics.csv');table=table[table.protocol=='selection_seed42'].sort_values('accuracy',ascending=False).head(12).iloc[::-1]
    fig,axes=plt.subplots(1,2,figsize=(15,5));axes[0].barh(table.model,table.accuracy*100,color='#227c9d');axes[0].set(title='Modelos históricos: umbral 0,50',xlabel='Accuracy (%) en 21.000 filas',xlim=(90,95.3));axes[0].axvline(95,color='#b33b3b',ls='--')
    names=['Selección\n21.000 filas','Comprobación\n14.000 filas','Kaggle público\ninformado por usuario'];values=[94.2095238095,94.2142857143,94.428]
    axes[1].bar(names,values,color=['#227c9d','#3a9d91','#f28f3b']);axes[1].set(title='Tres evaluaciones del ensamble previo',ylim=(92,95.4));axes[1].axhline(95,color='#b33b3b',ls='--',label='Objetivo 95%')
    for i,value in enumerate(values):axes[1].text(i,value+.04,f'{value:.3f}%',ha='center')
    axes[1].legend();show(fig)
def tpe(root):
    table=pd.read_csv(Path(root)/'optimization/tpe_trials.csv');fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].plot(table.number,table.value*100,'o',alpha=.65,label='Ensayo');axes[0].plot(table.number,table.value.cummax()*100,label='Mejor acumulado')
    axes[0].set(title='TPE aprende de sus ensayos',xlabel='Ensayo',ylabel='Accuracy de selección (%)');axes[0].legend()
    points=axes[1].scatter(table.params_num_leaves,table.params_learning_rate,c=table.value,cmap='viridis',s=80)
    axes[1].set(title='Hiperparámetros de ensayos reales',xlabel='Hojas por árbol',ylabel='Learning rate');fig.colorbar(points,ax=axes[1],label='Accuracy');show(fig)
def blend(root):
    selected=json.loads((Path(root)/'optimization/selected_ensemble.json').read_text());fig,axes=plt.subplots(1,2,figsize=(14,4))
    axes[0].bar(selected['names'],np.array(selected['weights'])*100,color=['#227c9d','#3a9d91','#f28f3b']);axes[0].set(title='Pesos congelados del ensamble previo',ylabel='Peso (%)');axes[0].tick_params(axis='x',rotation=15)
    p=pd.read_csv(Path(root)/'optimization/holdout_predictions.csv').head(8);axes[1].bar(np.arange(len(p)),p.probability_used,color=np.where(p.y_used==1,COLORS['used'],COLORS['new']))
    axes[1].axhline(.5,color='#b33b3b',ls='--',label='Umbral 0,50');axes[1].set(title='Probabilidad → etiqueta',xlabel='Primeras filas del holdout',ylabel='P(used)',ylim=(0,1));axes[1].legend();show(fig)
def evaluation(predictions,q=.5):
    y=predictions.y_used.to_numpy();p=predictions.probability_used.to_numpy();fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    ConfusionMatrixDisplay.from_predictions(y,p>=q,display_labels=['new','used'],ax=axes[0],colorbar=False,cmap='Blues');axes[0].set_title(f'Confusión: umbral {q:.2f}')
    RocCurveDisplay.from_predictions(y,p,ax=axes[1],color='#227c9d');axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set_title('ROC: ordenamiento de los casos')
    PrecisionRecallDisplay.from_predictions(y,p,ax=axes[2],color='#f28f3b');axes[2].set_title('Precisión y recall de used');show(fig)
def confidence(predictions,q=.5):
    y=predictions.y_used.to_numpy();p=predictions.probability_used.to_numpy();correct=(p>=q)==y;fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for label in [0,1]:axes[0].hist(p[y==label],bins=np.linspace(0,1,31),density=True,alpha=.5,label=['new','used'][label],color=COLORS[['new','used'][label]])
    axes[0].set(title='Separación de probabilidades',xlabel='P(used)',ylabel='Densidad');axes[0].legend()
    bins=pd.cut(np.maximum(p,1-p),[.5,.6,.7,.8,.9,.95,1.00001],right=False);frame=pd.DataFrame({'bin':bins,'correct':correct})
    rates=frame.groupby('bin',observed=False).correct.mean();sizes=frame.groupby('bin',observed=False).size()
    axes[1].bar(np.arange(len(rates)),rates*100,color='#227c9d');axes[1].set_xticks(np.arange(len(rates)),['50–60','60–70','70–80','80–90','90–95','95–100'],rotation=30)
    axes[1].set(title='Accuracy según confianza',xlabel='Confianza (%)',ylabel='Accuracy (%)',ylim=(0,105))
    for i,n in enumerate(sizes):axes[1].text(i,5,f'n={n}',ha='center',rotation=90,color='white')
    observed,estimated=calibration_curve(y,p,n_bins=10,strategy='quantile');axes[2].plot(estimated,observed,'o-',color='#f28f3b');axes[2].plot([0,1],[0,1],'--',color='gray')
    axes[2].set(title='Calibración',xlabel='Probabilidad promedio',ylabel='Frecuencia real de used');show(fig)
def threshold(predictions):
    y=predictions.y_used.to_numpy();p=predictions.probability_used.to_numpy();values=np.arange(.3,.701,.005)
    fig,ax=plt.subplots();ax.plot(values,[accuracy_score(y,p>=q)*100 for q in values],color='#227c9d');ax.axvline(.5,ls='--',color='#f28f3b',label='Umbral previo congelado')
    ax.set(title='Cambiar el umbral cambia el tipo de error',xlabel='Umbral para used',ylabel='Accuracy en holdout conocido (%)');ax.legend();show(fig)
def errors(raw,predictions,q=.5):
    subset=raw.iloc[predictions.row_index.to_numpy()].copy().reset_index(drop=True);subset['wrong']=(predictions.probability_used.to_numpy()>=q)!=predictions.y_used.to_numpy()
    table=subset.groupby('category_id').wrong.agg(['mean','size','sum']).query('size >= 60').sort_values('sum',ascending=False).head(14).iloc[::-1]
    fig,axes=plt.subplots(1,2,figsize=(13,5));axes[0].barh(table.index,table['sum'],color='#f28f3b');axes[0].set(title='Cantidad de errores por categoría',xlabel='Errores')
    axes[1].barh(table.index,table['mean']*100,color='#227c9d');axes[1].set(title='Cantidad de errores ≠ tasa de error',xlabel='% incorrecto');show(fig)
    subset['prob_used']=predictions.probability_used.to_numpy();return subset[subset.wrong]
def new_results(out):
    out=Path(out)
    if not (out/'metrics.csv').exists():print('La ronda nueva todavía no registró métricas.');return
    table=pd.read_csv(out/'metrics.csv').sort_values('accuracy');fig,ax=plt.subplots(figsize=(12,max(4,len(table)*.3)))
    ax.barh(table.name,table.accuracy*100,color='#3a9d91');ax.set_xlim(max(85,table.accuracy.min()*100-1),96);ax.axvline(95,color='#b33b3b',ls='--')
    ax.set(title='Nueva ronda: selección sobre datos íntegros',xlabel='Accuracy (%)');show(fig);return table
def importance(out):
    paths=list(Path(out).glob('*_importance.csv'))
    if not paths:print('Importancias aún pendientes.');return
    table=pd.read_csv(paths[0]).head(22).iloc[::-1];fig,ax=plt.subplots(figsize=(12,6));ax.barh(table.feature,table.gain,color='#3a9d91')
    ax.set(title='Variables que utiliza el primer LightGBM nuevo',xlabel='Ganancia en splits: no causalidad');show(fig)
