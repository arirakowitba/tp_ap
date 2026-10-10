"""Report actual experiment artifacts; never infer a Kaggle score from a holdout."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix,roc_curve

def summarize(out):
    out=Path(out);parts=[]
    for path in sorted(out.glob('*metrics.csv')):
        tab=pd.read_csv(path)
        if 'name' not in tab or 'accuracy' not in tab:continue
        tab['family']=tab['name'].str.split('_').str[0].replace({'stacking4':'stacking','stacking5':'stacking'})
        tab['source']=path.name;parts.append(tab)
    if not parts:return pd.DataFrame()
    return pd.concat(parts,ignore_index=True).drop_duplicates('name').sort_values('accuracy',ascending=False)

def overview(out):
    table=summarize(out)
    if table.empty:print('Todavía no hay configuraciones completas.');return
    best=table.sort_values('accuracy').groupby('family',sort=False).tail(1).sort_values('accuracy')
    fig,ax=plt.subplots(figsize=(10,4.5));values=100*best.accuracy.to_numpy()
    ax.barh(best.family,values,color=['#577590','#43aa8b','#f9c74f','#f9844a','#9b5de5'][:len(best)])
    for i,value in enumerate(values):ax.text(value+.02,i,f'{value:.3f}%',va='center')
    ax.set(xlim=(max(85,values.min()-.8),max(95.2,values.max()+.5)),xlabel='Accuracy de selección (%) · mismo holdout agrupado de14.000 filas',title='Mejor configuración de cada rama · selección, todavía no Kaggle')
    ax.axvline(95,color='#d62828',linestyle='--',label='Objetivo local95%');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2);fig.tight_layout();plt.show()
    return best[['family','name','accuracy','threshold']]

def ablations(out):
    out=Path(out);table=summarize(out);fig,axes=plt.subplots(2,2,figsize=(12,8))
    tab=table[table.family=='normalize'].sort_values('name')
    if len(tab):
        axes[0,0].barh(tab.name.str.replace('normalize_','',regex=False),100*tab.accuracy,color='#43aa8b');axes[0,0].set_xlim(89,95)
    axes[0,0].set_title('Normalización: texto solo y contexto tabular')
    cat=table[table.family=='category']
    if len(cat):
        for text in ['exact_300','cluster_8_l2','cluster_16_l2']:
            rows=cat[cat.name.str.contains(text)].sort_values('expert_weight')
            if len(rows):axes[0,1].plot(rows.expert_weight,100*rows.accuracy,'o-',label=text)
        base=cat[cat.name=='category_global_baseline']
        if len(base):axes[0,1].axhline(100*base.accuracy.iloc[0],color='gray',linestyle='--',label='General')
        axes[0,1].legend(fontsize=8)
    axes[0,1].set(xlabel='Peso del especialista',ylabel='Accuracy (%)',title='Más especialización no garantiza más aciertos')
    stack=table[table.family=='stacking']
    for prefix in ['stacking_','stacking4_','stacking5_']:
        for design in ['probability','logit']:
            rows=stack[stack.name.str.startswith(prefix+design+'_C')].copy()
            if len(rows):
                rows=rows.sort_values('C');axes[1,0].semilogx(rows.C,100*rows.accuracy,'o-',label=prefix+design)
    if len(stack):axes[1,0].legend(fontsize=7)
    axes[1,0].set(xlabel='C de la regresión logística',ylabel='Accuracy (%)',title='Stacking: combinación aprendida de probabilidades OOF')
    combined=table[table.family=='combined']
    for phase in ['normstack','stackbase','joint']:
        rows=combined[combined.name.str.startswith('combined_'+phase+'_')].copy()
        if len(rows):
            rows['trial']=rows.name.str.rsplit('_',n=1).str[-1].astype(int);rows=rows.sort_values('trial')
            axes[1,1].plot(rows.trial,100*rows.accuracy,'o-',label=phase)
    if len(combined):axes[1,1].legend(fontsize=8)
    axes[1,1].set(xlabel='Ensayo TPE',ylabel='Accuracy (%)',title='Representaciones conjuntas: aporte complementario')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.tight_layout();plt.show()

def embedding_atlas(out):
    out=Path(out);path=out/'embedding_titles.npy';rows=out/'cache_base/selection_rows.csv'
    if not path.exists() or not rows.exists():print('Regenerar el cache de embeddings para visualizar el espacio.');return
    from sklearn.decomposition import PCA
    tab=pd.read_csv(rows).sample(n=min(2500,len(pd.read_csv(rows))),random_state=2026)
    vectors=np.load(path,mmap_mode='r')[tab.row_index.to_numpy()]
    pca=PCA(2,random_state=2026);xy=pca.fit_transform(vectors)
    fig,ax=plt.subplots(figsize=(8,5))
    for label,color,name in [(0,'#577590','Nuevo'),(1,'#f9844a','Usado')]:
        mask=tab.y_used.to_numpy()==label;ax.scatter(xy[mask,0],xy[mask,1],s=9,alpha=.3,color=color,label=name)
    ax.set(title='Embeddings preentrenados de títulos · proyección ilustrativa',xlabel='Componente1',ylabel='Componente2');ax.legend()
    ax.text(.01,-.15,f'PCA2D conserva {100*pca.explained_variance_ratio_.sum():.1f}% de la varianza; el modelo usa384 dimensiones.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();plt.show()

def frozen_evaluation(out):
    out=Path(out);path=out/'round3_confirmation_predictions.csv'
    if not path.exists():print('Confirmación congelada todavía pendiente.');return
    cfg=json.loads((out/'round3_selected.json').read_text());tab=pd.read_csv(path)
    y=tab.y_used.to_numpy();p=tab.probability_used.to_numpy();old=tab.prior_probability.to_numpy();q=cfg['threshold']
    new_correct=(p>=q)==y;old_correct=(old>=.5)==y
    corrected=int((new_correct&~old_correct).sum());introduced=int((~new_correct&old_correct).sum())
    fig,axes=plt.subplots(1,3,figsize=(13,4));cm=confusion_matrix(y,p>=q)
    im=axes[0].imshow(cm,cmap='Blues');axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['Nuevo','Usado'],yticklabels=['Nuevo','Usado'],xlabel='Predicción',ylabel='Real',title='Confirmación: matriz de confusión')
    for (i,j),value in np.ndenumerate(cm):axes[0].text(j,i,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else 'black')
    for values,label in [(old,'Anterior'),(p,'Candidato congelado')]:
        fpr,tpr,_=roc_curve(y,values);axes[1].plot(fpr,tpr,label=label)
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(xlabel='Falsos positivos',ylabel='Verdaderos positivos',title='ROC · no cambia el umbral congelado');axes[1].legend(fontsize=8)
    axes[2].bar(['Corrige','Introduce'],[corrected,introduced],color=['#43aa8b','#f9844a']);axes[2].set(title='Errores respecto al anterior',ylabel='Cantidad de publicaciones')
    for i,n in enumerate([corrected,introduced]):axes[2].text(i,n+2,str(n),ha='center')
    fig.tight_layout();plt.show()
    paired_p=float(binomtest(corrected,corrected+introduced,.5).pvalue) if corrected+introduced else 1.
    return dict(prior_accuracy=float(old_correct.mean()),new_accuracy=float(new_correct.mean()),corrected=corrected,introduced=introduced,net_corrected=corrected-introduced,paired_exact_p=paired_p,threshold_frozen=q,public_kaggle_score=None)
