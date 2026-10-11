"""Close the interrupted round from durable artifacts; execute report cells only.

No data download, training, or invented confirmation takes place here.
Historical outputs in notebook22 are retained and identified as historical.
"""
from pathlib import Path
import json,os,sys,datetime
import nbformat
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from round3_visuals import summarize
from execute_project_notebooks import execute

def main():
    out=ROOT/'results/round_v3';table=summarize(out)
    count=int(table.name.nunique());assert count==57
    best=table.sort_values('accuracy').groupby('family').tail(1)
    stack=json.loads((out/'stacking_selected.json').read_text())
    assert stack['status']=='confirmed'
    closure=dict(status='interrupted_closed_from_durable_checkpoint',
        original_deadline_utc='2026-10-11T00:43:51.416Z',closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        last_durable_commit='c3eb04b92633d7c66401eff4f04435ac80085fd4',completed_configurations_with_saved_metrics=count,
        best_by_family=best[['family','name','accuracy']].to_dict('records'),
        stacking_confirmation=stack['confirmation'],prior_confirmation_accuracy=.9421428571428572,
        public_reference_user_report=.94428,goal_met_local=False,new_submission=None,
        retained_submission='submissions/submission_optimized_ensemble.csv',
        missing_artifacts=['stacking4 final selection/confirmation','stacking5 results','joint embeddings TPE results','final blend confirmation','new full-data refit'],
        embedding_note='70k×384 inference and three text-only LR scores were reported before interruption; prediction/cache artifacts did not reach durable checkpoint. Do not treat missing results as confirmed.',
        image_note='14 image URL requests returned403; no visual model trained.',
        verification_note='Report cells executed after recovery; historical notebook22 outputs retained. Full master training cycle not reexecuted.')
    (out/'closure.json').write_text(json.dumps(closure,indent=2))
    manifest=json.loads((out/'run_manifest.json').read_text());manifest.update(status=closure['status'],closed_utc=closure['closed_utc'],new_submission=None)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    selected=json.loads((out/'round3_selected.json').read_text());selected.update(status='interim_selection_interrupted_unconfirmed',promote_over_previous=False,goal_met_local=False,submission=None,
        caution='Interim blend predates combined trials and uses an imbalanced cluster router. It was never confirmed and is not promoted.')
    (out/'round3_selected.json').write_text(json.dumps(selected,indent=2))
    category=json.loads((out/'category_summary.json').read_text());category['fixed_threshold_caveat']='The +3 selected cases also change threshold .495→.475. At .495 both global and exact300w25 score93.9429%; no net improvement attributable solely to specialization.'
    (out/'category_summary.json').write_text(json.dumps(category,indent=2))
    report='''# Cierre de la ronda complementaria

La sesión se interrumpió. El último respaldo durable conserva **57 configuraciones con métricas**:5 de normalización,25 especialistas,11 bases/meta-modelos de stacking y16 búsquedas conjuntas. No hay evidencia de una confirmación final del blend ni un refit nuevo. Se conserva la submission cuyo **94,428% público** informó el usuario.

| Rama | Mejor selección local | Confirmación |
|---|---:|---|
| Normalización y tabular |94,0357%|Pendiente|
| Especialistas |93,9643%|Pendiente|
| Stacking aprendido,3 bases |93,8643%|94,0143%|
| Normalización + OOF + tabular |94,0571%|Pendiente|
| Referencia previa |94,0214% sobre estas filas|94,2143%|

La selección de la referencia proviene de un modelo entrenado con49.000 filas; los candidatos usan56.000. La comparación en confirmación utiliza56.000 para ambos. Las semillas42 y731 son particiones agrupadas del mismo dataset ya explorado, no un test externo intacto. El objetivo95% no quedó demostrado.

Los especialistas exactos corrigen17 errores e introducen14 con su umbral elegido; esa diferencia de3 incluye cambiar el umbral. Al fijar .495, empatan al modelo general. Los clusters sin normalizar se colapsaron; los ensayos con centroides L2 no mejoraron. El blend intermedio de94,0714% no es una selección final válida ni una mejora confirmada.

Se informó inferencia real de70.000 títulos con384 dimensiones y regresiones logísticas de71,5%,72,3643% y72,4286%. Sus predicciones no llegaron al respaldo. Los vectores no eran constantes, pero el backend cuantizado mostró pequeñas diferencias entre títulos duplicados según el batch. Los ensayos incrementales y el stacking ampliado quedaron sin resultados finales recuperables. No se fabrican métricas para completarlos.

Las14 solicitudes a imágenes devolvieron403: esta hipótesis quedó bloqueada por acceso y no se entrenó un modelo visual. ONNX Runtime fue rechazado por revisión automática debido a telemetría; se cambió a OpenVINO CPU, con inferencia local y pesos preentrenados congelados.

El notebook maestro ofrece full y report, presupuesto global y confirmación antes del refit. En esta entrega se verificó report, no se volvió a ejecutar full. El ilustrativo conserva18 figuras históricas y agrega la comparación de esta ronda. Caches/modelos se regeneran con los datos oficiales; no deben confundirse los estados pendientes con ejecuciones completas.
'''
    (out/'README.md').write_text(report)
    os.chdir(ROOT);os.environ.update(AP_RUN_MODE='report',AP_PIPELINE='round3',AP_OUTPUT_DIR=str(out),MPLBACKEND='Agg')
    path=ROOT/'notebooks/21_maestro_pipeline.ipynb';master=nbformat.read(path,4)
    master.cells[2].source=master.cells[2].source.replace('from competition_pipeline_v2 import load_data,groups_for\nfrom optimize_leaderboard import split','')
    master.cells[4].source=master.cells[4].source.replace("os.getenv('AP_RUN_MODE','full')","os.getenv('AP_RUN_MODE','report')")
    master.cells[6].source="""if RUN_MODE=='report':
    audit=json.loads((OUT/'data_audit.json').read_text())
    display(pd.DataFrame(audit))
    print('Auditoría histórica guardada; no se vuelven a leer datos en report.')
else:
    from competition_pipeline_v2 import load_data,groups_for
    from optimize_leaderboard import split
    train,test,audit=load_data(DATA_DIR,OUT,ALLOW_PARTIAL)
    display(pd.DataFrame(audit));display(train[['id','title','condition','price','category_id']].head())
"""
    master.cells[8].source="""if RUN_MODE=='report':
    protocol=json.loads((OUT/'run_manifest.json').read_text())
    display(protocol['selection']);display(protocol['confirmation'])
    print('Protocolos históricos. La comprobación por familias se realiza al entrenar.')
else:
    y=(train.condition=='used').to_numpy(dtype='int8');groups=groups_for(train)
    t,v=split(y,groups,42,2)
    assert not set(groups[t])&set(groups[v])
    print({'desarrollo':len(t),'validacion':len(v),'familias_compartidas':0})
"""
    master.cells[13].source+="\nif (OUT/'closure.json').exists():display(json.loads((OUT/'closure.json').read_text()))"
    master.cells[16].source+='\n\n**Estado de esta ronda:** interrumpida;57 configuraciones respaldadas. No hay nueva submission confirmada. Modo report ejecutado después de recuperar el último commit.'
    execute(master);nbformat.validate(master);nbformat.write(master,path)
    path=ROOT/'notebooks/22_caso_desde_cero.ipynb';case=nbformat.read(path,4)
    historical=sum('image/png' in o.get('data',{}) for c in case.cells if c.cell_type=='code' for o in c.outputs)
    appendix=[c for c in case.cells if c.metadata.get('round3_appendix')]
    appendix[0].source+='\n\n**Cierre:** esta sesión se interrumpió. El último checkpoint guarda57 métricas; la mejor búsqueda conjunta quedó en94,0571% de selección, sin confirmación. Se conserva94,428% como referencia pública. Los capítulos anteriores conservan sus salidas históricas; este apéndice se ejecutó tras la recuperación.'
    appendix[4].source+='\n\nEn especialistas, la diferencia de3 aciertos incorpora cambiar el umbral. Con umbral .495 fijo, el experto exacto empata al general; no es evidencia de ganancia por especialización.'
    appendix[7].source+="\ndisplay(json.loads((OUT3/'closure.json').read_text()))\nfrom round3_visuals import confirmation_overview\nconfirmation_overview(OUT3)"
    subset=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("from pathlib import Path\nimport sys,json\nimport pandas as pd\nROOT=Path.cwd()\nsys.path.insert(0,str(ROOT/'src'))"),*appendix])
    execute(subset)
    case.metadata['round3_execution_verification']=dict(mode='report appendix only',historical_figures_retained=historical,full_training_reexecuted=False)
    nbformat.validate(case);nbformat.write(case,path)
    result=[]
    for name in ['21_maestro_pipeline.ipynb','22_caso_desde_cero.ipynb']:
        n=nbformat.read(ROOT/'notebooks'/name,4);nbformat.validate(n)
        errors=sum(o.output_type=='error' for c in n.cells if c.cell_type=='code' for o in c.outputs)
        figures=sum('image/png' in o.get('data',{}) for c in n.cells if c.cell_type=='code' for o in c.outputs)
        assert errors==0;result.append(dict(notebook=name,errors=errors,figures=figures))
    (out/'notebook_verification.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
