"""Add round3 orchestration and actual-result visual narrative to project notebooks."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
master=ROOT/'notebooks/21_maestro_pipeline.ipynb';n=nbformat.read(master,4)
n.cells[0].source='# Notebook maestro · De JSONLines a la submission\n\nAudita los datos y ejecuta una ronda reproducible con presupuesto global. Por defecto, `round3` prueba normalización, especialistas, stacking aprendido y embeddings preentrenados; también verifica el acceso a imágenes. `v2` conserva el ciclo anterior. Los resultados públicos de Kaggle y las métricas locales son mediciones diferentes.'
n.cells[2].source=n.cells[2].source.replace("DATA_DIR=Path(os.getenv('AP2026Q2_DATA_DIR',str(ROOT/'data/ap_2026_q_2/raw')))","DATA_DIR=Path(os.getenv('AP2026Q2_DATA_DIR',str(ROOT.parent/'data' if (ROOT.parent/'data/train_data.jsonlines').exists() else ROOT/'data/ap_2026_q_2/raw')))\nPIPELINE=os.getenv('AP_PIPELINE','round3')")
n.cells[2].source=n.cells[2].source.replace("str(ROOT/'results/round_v2')","str(ROOT/'results'/('round_v3' if PIPELINE=='round3' else 'round_v2'))")
n.cells[3].source='## 2. Configuración explícita\n`full` ejecuta entrenamientos; `report` lee resultados existentes. `AP_PIPELINE=round3` ejecuta las ramas complementarias; `v2` recupera la búsqueda anterior. `AP_MINUTES=90` limita los subprocesses y reserva tiempo para confirmación/refit. Instalar `requirements_round3.txt` para la nueva ronda. Los pesos preentrenados se descargan con revisión y SHA256 congelados; la inferencia es local.'
n.cells[10].source='## 6. Ejecutar el ciclo completo\n\nEl orquestador reutiliza caches compatibles, ejecuta las ramas secuencialmente para reducir memoria, selecciona con el holdout42, congela configuración/pesos/umbral y confirma con731. Solo refitea una propuesta si mejora la referencia local. Las URLs de imágenes se verifican antes de prometer una prueba visual. Si falta acceso o se agota el presupuesto, se conserva el estado incompleto; no se inventa una métrica. La confirmación usa otra partición del mismo dataset y ya se utilizó en rondas anteriores: no es un test ciego independiente.'
n.cells[11].source="""import subprocess,time
if RUN_MODE=='full':
    if ALLOW_PARTIAL:raise ValueError('El maestro requiere70.000 filas completas.')
    if PIPELINE=='round3':
        command=[sys.executable,'-u',str(ROOT/'src/run_round3.py'),'--data-dir',str(DATA_DIR),'--output',str(OUT),'--minutes',str(MINUTES),'--trials',str(min(TRIALS,12))]
    elif PIPELINE=='v2':
        command=[sys.executable,'-u',str(ROOT/'src/competition_pipeline_v2.py'),'--data-dir',str(DATA_DIR),'--output',str(OUT),'--trials',str(TRIALS)]
    else:raise ValueError('PIPELINE debe ser round3 o v2')
    log_path=OUT/'master_run.log';env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    with log_path.open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:returncode=process.wait(timeout=MINUTES*60+5)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            returncode=124
    print(log_path.read_text()[-4500:])
    if returncode:print('Ejecución incompleta: revisar logs y estado; la referencia anterior sigue disponible.')
elif RUN_MODE=='report':print('Modo reporte: se leen resultados existentes, no se entrena.')
else:raise ValueError('RUN_MODE debe ser full o report')
"""
n.cells[13].source="""if PIPELINE=='round3':
    from round3_visuals import summarize,overview,frozen_evaluation
    results=summarize(OUT);display(results[[c for c in ['name','family','accuracy','threshold'] if c in results]].head(15))
    overview(OUT)
    if (OUT/'round3_selected.json').exists():
        selected=json.loads((OUT/'round3_selected.json').read_text());display(selected)
    display(frozen_evaluation(OUT))
else:
    if (OUT/'metrics.csv').exists():display(pd.read_csv(OUT/'metrics.csv').sort_values('accuracy',ascending=False))
    if (OUT/'selected.json').exists():display(json.loads((OUT/'selected.json').read_text()))
    from case_visuals import style,new_results,importance
    style();new_results(OUT);importance(OUT)
"""
n.cells[14].source='## 8. Submission\n\nLa referencia pública comprobada sigue identificada por separado. Un CSV nuevo solo se propone si tiene confirmación local; su puntaje de Kaggle queda pendiente hasta que se lo evalúe en la competencia. El refit usa todas las filas etiquetadas y transforma el test con lo aprendido en entrenamiento.'
n.cells[15].source="""candidate=ROOT/'submissions/submission_optimized_ensemble.csv'
if PIPELINE=='round3' and (OUT/'round3_selected.json').exists():
    state=json.loads((OUT/'round3_selected.json').read_text())
    if state.get('promote_over_previous') and (OUT/'submission_round3.csv').exists():
        candidate=OUT/'submission_round3.csv';print('CSV experimental confirmado localmente; Kaggle pendiente.')
sub=pd.read_csv(candidate)
assert list(sub.columns)==['ID','condition'] and len(sub)==30000
assert sub.ID.tolist()==list(range(1,30001)) and set(sub.condition)<={'new','used'} and not sub.isna().any().any()
display(sub.head());print(candidate)
print('Referencia pública informada por el usuario:94.428%.')
"""
n.cells[16].source+='\n\nLa ejecución de los notebooks entregados es en modo reporte. Los modelos se entrenaron en procesos separados, documentados en sus logs. Esto no equivale a reejecutar integralmente el modo full del maestro. Los caches y pesos grandes se regeneran; fuentes, configuraciones, métricas y predicciones congeladas quedan versionados.'
for c in n.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
nbformat.write(n,master)
case=ROOT/'notebooks/22_caso_desde_cero.ipynb';n=nbformat.read(case,4)
# Keep the previous round's section intact and append the complementary work.
n.cells[2].source=n.cells[2].source.replace("OUT=Path(os.getenv('AP_OUTPUT_DIR',str(ROOT/'results/round_v2')))","OUT=ROOT/'results/round_v2'")
n.cells=[c for c in n.cells if not c.metadata.get('round3_appendix')]
cells=[
nbformat.v4.new_markdown_cell('## 16. Cinco hipótesis nuevas: intentar mejorar con información complementaria\n\n| Hipótesis | Intuición | Evaluación |\n|---|---|---|\n| Normalizar texto | `s/uso` y `sin uso` deberían compartir señal | Texto original, normalizado y flags separados |\n| Embeddings de títulos | Un modelo preentrenado puede captar significado y sinónimos |384 números por título; solo título y combinado con tabular |\n| Imágenes | Embalaje y desgaste podrían ser informativos | Primero comprobar acceso real a las fotos |\n| Especialistas | Las señales pueden cambiar entre categorías | Modelos generales más expertos, con fallback |\n| Stacking aprendido | Los modelos pueden cometer errores distintos | Regresión logística sobre predicciones OOF estrictas |\n\nLos hiperparámetros se eligen con selección42. Después se congelan y se verifican con731; ambas particiones provienen del mismo conjunto y no garantizan el score público. La misma evaluación se reutilizó en rondas anteriores.'),
nbformat.v4.new_code_cell("OUT3=ROOT/'results/round_v3'\nfrom round3_visuals import summarize,overview,ablations,embedding_atlas,frozen_evaluation\nround3_table=summarize(OUT3)\ndisplay(round3_table[[c for c in ['name','family','accuracy','threshold'] if c in round3_table]].head(15))\ndisplay(overview(OUT3))"),
nbformat.v4.new_markdown_cell('**Interpretación:** estas barras comparan selección local sobre las mismas14.000 filas. No son puntajes de Kaggle. Buscar muchas configuraciones puede mejorar esta tabla por azar: por eso importa la confirmación congelada, el análisis de errores y conservar la referencia anterior.'),
nbformat.v4.new_code_cell('ablations(OUT3)'),
nbformat.v4.new_markdown_cell('**Qué aprendemos:** normalizar puede borrar detalles útiles; aumentar el peso de un experto puede empeorar; un meta-modelo no crea información que sus bases no contienen. Los ensayos TPE combinan las nuevas señales sin utilizar probabilidades de entrenamiento del propio ejemplo: cada predicción usada como variable proviene de un modelo que excluyó su familia.'),
nbformat.v4.new_code_cell('embedding_atlas(OUT3)'),
nbformat.v4.new_markdown_cell('**Embeddings:** cada punto representa un título, calculado localmente con pesos públicos preentrenados. La proyección2D permite mirar el espacio, pero no conserva toda la información ni implica separación perfecta. Las primeras ejecuciones con ONNX Runtime fueron bloqueadas por telemetría; se utilizó otro backend local. Las imágenes devolvieron403: se documenta el bloqueo, sin presentarlo como un modelo visual evaluado.'),
nbformat.v4.new_code_cell("display(frozen_evaluation(OUT3))\nif (OUT3/'round3_selected.json').exists():\n    final_round3=json.loads((OUT3/'round3_selected.json').read_text());display({k:final_round3.get(k) for k in ['names','weights','threshold','accuracy','confirmation','prior_confirmation','promote_over_previous','goal_met_local','submission']})\nprint('Mejor público previamente informado:94.428%; cualquier CSV nuevo necesita evaluación en Kaggle.')"),
nbformat.v4.new_markdown_cell('**Cómo cerrar la explicación:** mostrar qué hipótesis se probaron, cuáles aportaron señal, cuántos errores corrigió o introdujo el candidato congelado y qué falta verificar. Un aumento pequeño no debe describirse como garantía de mejora. El resultado público y la confirmación local se reportan por separado. Código, decisiones y resultados quedan en `results/round_v3`; el maestro permite repetir la ronda con un presupuesto global.')]
for c in cells:c.metadata['round3_appendix']=True
n.cells.extend(cells);nbformat.write(n,case)
print('Notebooks actualizados; falta ejecutar modo reporte y verificar figuras.')
