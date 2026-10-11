"""Report round 4 and append executed report cells, preserving old notebook outputs.

Default CLI is report only. No weights, training or remote requests occur while
reporting. Missing final artifacts are explicitly pending, never fabricated.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def summarize(output):
    import pandas as pd
    output = Path(output)
    rows = []
    for path in sorted(output.rglob('r4_*.json')):
        row = read_json(path)
        if isinstance(row, dict) and isinstance(row.get('accuracy'), (float, int)) and row.get('name'):
            row = dict(row)
            row['use_media'] = row.get('use_media', row.get('media_metadata'))
            row['uses_pixels'] = row.get('uses_pixels', row.get('image_pixels_used'))
            row['uses_video_frames'] = row.get('uses_video_frames', row.get('video_frames_used'))
            rows.append({key: value for key, value in row.items() if key not in
                         ('params', 'requested_params', 'fingerprint', 'cat_features', 'feature_sets', 'text_features')})
    for name in ('lgb_round4_metrics.csv', 'cat_round4_metrics.csv'):
        path = output / name
        if path.exists():
            try:
                rows.extend(pd.read_csv(path).to_dict('records'))
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
    if not rows:
        return pd.DataFrame(columns=['name', 'family', 'accuracy', 'threshold'])
    table = pd.DataFrame(rows).drop_duplicates('name', keep='first')
    table['accuracy'] = pd.to_numeric(table.accuracy, errors='coerce')
    return table.dropna(subset=['accuracy']).sort_values('accuracy', ascending=False).reset_index(drop=True)


def frozen_state(output):
    output = Path(output)
    for name in ('round4_selected.json', 'r4_selected.json', 'selected.json'):
        row = read_json(output / name)
        if row:
            row = dict(row)
            row['status'] = row.get('status', row.get('stage', 'selection_available'))
            confirmation = read_json(output/'round4_confirmation.json')
            if confirmation:
                row.update(confirmation=confirmation.get('candidate', {}),
                           confirmation_baseline=confirmation.get('baseline', {}),
                           confirmation_record=confirmation,
                           promote_over_previous=confirmation.get('promote', False),
                           goal_met_local=confirmation.get('local_goal_reached', False),
                           status=confirmation.get('stage', 'confirmation_available'))
            refit = read_json(output/'round4_refit.json')
            if refit:
                row['refit'] = refit
            return row
    return {'status': 'pending_final_selection', 'promote_over_previous': False, 'goal_met_local': False}


def coverage(output):
    import pandas as pd
    inventory = read_json(Path(output) / 'media_inventory.json')
    rows = []
    for split in ('train', 'test'):
        item = inventory.get(split, {})
        total = item.get('rows', 0)
        fields = item.get('nonempty_fields', {})
        for name, count in [('Metadatos de fotos', fields.get('pictures', 0)),
                            ('ID de video presente', fields.get('video_id', 0)),
                            ('Fotos decodificadas', 0), ('Frames decodificados', 0)]:
            rows.append({'split': split, 'informacion': name, 'filas': count,
                         'porcentaje': 100*count/total if total else None})
    # Actual decoded coverage only enters after a real local extractor run.
    local = read_json(Path(output) / 'local_visual_run.json')
    if local.get('inference_executed'):
        path = Path(output) / 'local_visual_coverage.csv'
        if path.exists():
            actual = pd.read_csv(path)
            for row in rows:
                subset = actual[actual.split == row['split']]
                field = {'Fotos decodificadas': 'image_missing', 'Frames decodificados': 'video_missing'}.get(row['informacion'])
                if field and field in subset:
                    row['filas'] = int((subset[field] == 0).sum())
                    total = inventory.get(row['split'], {}).get('rows', 0)
                    row['porcentaje'] = 100*row['filas']/total if total else None
    return pd.DataFrame(rows)


def coverage_plot(output):
    import matplotlib.pyplot as plt
    table = coverage(output)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
    for axis, split in zip(axes, ('train', 'test')):
        data = table[table.split == split]
        values = data.porcentaje.fillna(0).to_numpy()
        axis.barh(data.informacion, values, color=['#227c9d', '#7b2cbf', '#999999', '#999999'])
        for number, value in enumerate(values):
            axis.text(min(value+1, 93), number, f'{value:.2f}% · {int(data.iloc[number].filas):,}', va='center')
        axis.set(xlim=(0, 110), xlabel='% de filas originales', title=split.capitalize())
        axis.invert_yaxis()
        axis.grid(axis='x', alpha=.2)
    fig.suptitle('Tener una URL o medidas de foto no equivale a tener sus píxeles', fontsize=13)
    fig.tight_layout()
    plt.show()
    return table


def matched_ablations(output):
    import pandas as pd
    table = summarize(output).set_index('name')
    pairs = [('LightGBM', 'r4_lgb_original', 'r4_lgb_media'),
             ('LightGBM · 430 árboles fijos', 'r4_lgb_fixed430_original', 'r4_lgb_fixed430_media'),
             ('CatBoost enriquecido d9', 'r4_cat_enriched_d9_original', 'r4_cat_enriched_d9_media'),
             ('CatBoost texto nativo d7', 'r4_cat_native_text_d7_original', 'r4_cat_native_text_d7_media')]
    rows = []
    for label, original, media in pairs:
        row = {'modelo': label, 'original_name': original, 'media_name': media, 'estado': 'pendiente'}
        if original in table.index:
            row['original_accuracy'] = float(table.loc[original, 'accuracy'])
        if media in table.index:
            row['media_accuracy'] = float(table.loc[media, 'accuracy'])
        if 'original_accuracy' in row and 'media_accuracy' in row:
            row.update(estado='par completo', delta_puntos_porcentuales=100*(row['media_accuracy']-row['original_accuracy']))
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_plot(output):
    import matplotlib.pyplot as plt
    import numpy as np
    table = matched_ablations(output)
    complete = table[table.estado == 'par completo']
    if complete.empty:
        print('Todavía no hay un par original/metadatos completo; no se dibujan resultados inexistentes.')
        return table
    x = np.arange(len(complete))
    fig, axis = plt.subplots(figsize=(10, 4))
    a, b = 100*complete.original_accuracy.to_numpy(), 100*complete.media_accuracy.to_numpy()
    axis.bar(x-.18, a, .36, label='Representación original', color='#486581')
    axis.bar(x+.18, b, .36, label='+ 50 metadatos multimedia', color='#2a9d8f')
    for positions, values in ((x-.18, a), (x+.18, b)):
        for position, value in zip(positions, values):
            axis.text(position, value+.03, f'{value:.3f}%', ha='center', fontsize=9)
    axis.axhline(95, color='#bc4749', ls='--', label='Objetivo 95%')
    labels = complete.modelo.str.replace(' · ', '\n', regex=False).str.replace('CatBoost ', 'CatBoost\n', regex=False)
    axis.set(xticks=x, xticklabels=labels, ylabel='Accuracy de selección (%)',
             ylim=(max(0, min(min(a), min(b))-1), max(95.4, max(max(a), max(b))+.4)),
             title='Misma partición y umbral: ¿añadir medidas de fotos ayuda?')
    axis.legend(fontsize=8)
    axis.grid(axis='y', alpha=.2)
    fig.tight_layout()
    plt.show()
    return table


def paired_diagnostics(output):
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest
    output = Path(output)
    metrics = summarize(output).set_index('name')
    pairs = matched_ablations(output)
    rows = []
    for row in pairs.to_dict('records'):
        if row['estado'] != 'par completo':
            continue
        names = [row['original_name'], row['media_name']]
        paths = []
        for name in names:
            path = next(output.rglob(f'{name}_predictions.csv'), None)
            if path is None:
                path = next(output.rglob(f'{name}_predictions.csv.gz'), output/f'{name}_predictions.csv')
            paths.append(path)
        if not all(path.exists() for path in paths):
            continue
        original, media = [pd.read_csv(path).sort_values('row_index').reset_index(drop=True) for path in paths]
        original = original.rename(columns={'y': 'y_used', 'p': 'probability_used'})
        media = media.rename(columns={'y': 'y_used', 'p': 'probability_used'})
        if not np.array_equal(original.row_index, media.row_index) or not np.array_equal(original.y_used, media.y_used):
            raise ValueError('Paired prediction rows/labels disagree')
        thresholds = [float(metrics.loc[name, 'threshold']) if 'threshold' in metrics else .5 for name in names]
        original_correct = (original.probability_used.to_numpy() >= thresholds[0]) == original.y_used.to_numpy()
        media_correct = (media.probability_used.to_numpy() >= thresholds[1]) == media.y_used.to_numpy()
        corrected = int((~original_correct & media_correct).sum())
        introduced = int((original_correct & ~media_correct).sum())
        discordant = corrected + introduced
        rows.append({'modelo': row['modelo'], 'filas': len(original), 'corregidos': corrected,
                     'introducidos': introduced, 'ganancia_neta': corrected-introduced,
                     'mcnemar_exacto_p': float(binomtest(corrected, discordant, .5).pvalue) if discordant else 1.,
                     'threshold_original': thresholds[0], 'threshold_media': thresholds[1]})
    return pd.DataFrame(rows)


def paired_plot(output):
    import matplotlib.pyplot as plt
    import numpy as np
    table = paired_diagnostics(output)
    if table.empty:
        print('Predicciones pareadas completas pendientes; no se fabrican errores corregidos/introducidos.')
        return table
    x = np.arange(len(table))
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(x-.18, table.corregidos, .36, color='#2a9d8f', label='Errores corregidos')
    axis.bar(x+.18, table.introducidos, .36, color='#e76f51', label='Errores introducidos')
    labels = table.modelo.str.replace(' · ', '\n', regex=False).str.replace('CatBoost ', 'CatBoost\n', regex=False)
    axis.set(xticks=x, xticklabels=labels, ylabel='Publicaciones de selección',
             title='Una mejora pequeña puede ocultar muchos cambios de decisión')
    axis.legend()
    axis.grid(axis='y', alpha=.2)
    fig.tight_layout()
    plt.show()
    return table


def group_ablation_plot(output):
    """Block effects at fixed capacity; not model feature importance."""
    import pandas as pd
    import matplotlib.pyplot as plt
    output = Path(output)
    baseline = read_json(output/'fixed_ablation/r4_lgb_fixed430_original.json').get('accuracy')
    labels = {'image': 'Metadatos de fotos', 'video': 'Solo presencia de video',
              'orientation': 'Orientación y aspecto', 'dimensions': 'Dimensiones y resolución'}
    rows = []
    for group, label in labels.items():
        path = output/f'media_groups/{group}/r4_lgb_fixed430_media_{group}.json'
        model = read_json(path)
        if 'accuracy' in model:
            rows.append({'bloque': label, 'accuracy': model['accuracy'], 'arboles': 430,
                         'delta_pp': 100*(model['accuracy']-baseline) if baseline is not None else None})
    table = pd.DataFrame(rows)
    if table.empty:
        print('Ablaciones de bloques a capacidad fija pendientes.')
        return table
    fig, axis = plt.subplots(figsize=(10, 4))
    values = table.delta_pp if baseline is not None else 100*table.accuracy
    colors = ['#2a9d8f' if value >= 0 else '#e76f51' for value in values]
    axis.barh(table.bloque, values, color=colors)
    axis.axvline(0, color='#555555', linewidth=1)
    axis.set(xlabel='Cambio en accuracy (puntos porcentuales)' if baseline is not None else 'Accuracy de selección (%)',
             title='Aporte de bloques: mismo LightGBM, 430 árboles y umbral 0,5')
    axis.invert_yaxis()
    axis.grid(axis='x', alpha=.2)
    fig.tight_layout()
    plt.show()
    return table


def extended_progress_plot(output):
    """Real selection-only complementarity objective; no confirmation reuse."""
    import pandas as pd
    import matplotlib.pyplot as plt
    rows = []
    for path in Path(output).glob('extended/r4_lgb_extended_*.json'):
        row = read_json(path)
        if 'objective_blend_accuracy' in row:
            row = dict(row)
            row['trial'] = int(row['name'].rsplit('_', 1)[-1])
            rows.append(row)
    if not rows:
        print('Búsqueda de complementariedad pendiente; no se dibuja una evolución simulada.')
        return pd.DataFrame()
    table = pd.DataFrame(rows).sort_values('trial')
    objective = 100*table.objective_blend_accuracy
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.scatter(table.trial, objective, s=28, color='#227c9d', alpha=.65,
                 label='Cada prueba: 0,6 referencia + 0,4 nuevo')
    axis.plot(table.trial, objective.cummax(), color='#2a9d8f', lw=2, label='Mejor objetivo acumulado')
    if 'prior_accuracy' in table:
        axis.axhline(100*float(table.prior_accuracy.iloc[0]), ls=':', color='#486581', label='Referencia en selección')
    axis.axhline(95, ls='--', color='#bc4749', label='Objetivo 95%')
    axis.set(xlabel='Número real de trial TPE', ylabel='Accuracy de selección (%)',
             title='Buscar un modelo que complemente a la referencia, con pesos fijos y umbral 0,5')
    axis.legend(fontsize=8, loc='lower right')
    axis.grid(alpha=.2)
    fig.tight_layout()
    plt.show()
    return table[['trial', 'media_group', 'accuracy', 'objective_blend_accuracy', 'prior_accuracy']]


def evaluation_table(output):
    import pandas as pd
    state = frozen_state(output)
    rows = []
    for phase in ('selection', 'confirmation'):
        data = state.get('selection_metrics', state.get(phase, {})) if phase == 'selection' else state.get(phase, {})
        if isinstance(data, (float, int)):
            data = {'accuracy': data}
        if not isinstance(data, dict):
            data = {}
        accuracy = data.get('accuracy', state.get(f'{phase}_accuracy'))
        baseline = state.get(f'{phase}_baseline', {}).get('accuracy',
            data.get('prior_accuracy', data.get('baseline_accuracy', state.get(f'prior_{phase}_accuracy'))))
        rows.append({'evaluacion': phase, 'candidate_accuracy': accuracy, 'prior_accuracy': baseline,
                     'estado': 'disponible' if accuracy is not None else 'pendiente'})
    return pd.DataFrame(rows)


def evaluation_plot(output):
    import matplotlib.pyplot as plt
    import numpy as np
    table = evaluation_table(output)
    complete = table[table.candidate_accuracy.notna()]
    if complete.empty:
        print('Selección congelada/confirmación final todavía no disponibles. Referencia pública informada: 94,428%.')
        return table
    x = np.arange(len(complete))
    fig, axis = plt.subplots(figsize=(8, 4))
    candidate = 100*complete.candidate_accuracy.astype(float).to_numpy()
    prior = 100*complete.prior_accuracy.astype(float).to_numpy()
    axis.bar(x-.18, prior, .36, color='#486581', label='Referencia previa · evaluación local')
    axis.bar(x+.18, candidate, .36, color='#2a9d8f', label='Candidato congelado')
    axis.axhline(95, color='#bc4749', ls='--', label='95% local')
    axis.set(xticks=x, xticklabels=complete.evaluacion, ylabel='Accuracy local (%)',
             ylim=(max(0, float(np.nanmin(candidate))-1), max(95.4, float(np.nanmax(candidate))+.4)),
             title='Selección y confirmación tienen distinta función')
    axis.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return table


def frozen_paired_table(output):
    import pandas as pd
    state = frozen_state(output)
    rows = []
    for phase, record in (('selection', state.get('selection_paired', {})),
                           ('confirmation', state.get('confirmation_record', {}).get('paired', {}))):
        if record:
            rows.append({'fase': phase, 'corregidos': record.get('repairs'),
                         'introducidos': record.get('introduces'), 'ganancia_neta': record.get('net_correct'),
                         'p_exacto': record.get('exact_two_sided_p')})
    return pd.DataFrame(rows)


def stage(command, deadline, log_path):
    remaining = deadline-time.time()
    if remaining <= 1:
        return {'status': 'not_started_deadline', 'command': command}
    with Path(log_path).open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
            return {'status': 'completed' if code == 0 else 'failed', 'returncode': code, 'command': command}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {'status': 'terminated_deadline', 'command': command}


def finalizer_command(script, mode, values):
    """Adapt optional arguments to the finalizer's real advertised CLI."""
    help_text = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True,
                               check=True, timeout=10).stdout
    available = set(re.findall(r'--[a-zA-Z0-9_-]+', help_text))
    command = [sys.executable, '-u', str(script)]
    command.extend(['--mode', mode] if '--mode' in available else [mode])
    for option, value in values.items():
        if option in available:
            command.extend([option, str(value)])
    return command


def run_training_pipeline(data_dir, output, minutes=120., trials=12, reserve_minutes=50., extended_trials=50):
    """Explicitly invoked training path; sequential and globally bounded."""
    output = Path(output).resolve()
    data_dir = Path(data_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.time()+60*minutes
    selection_deadline = max(time.time(), deadline-60*min(reserve_minutes, minutes*.5))
    stages = []
    script = ROOT / 'src'
    common = [sys.executable, '-u']
    commands = [('media_access', common+[str(script/'multimodal_access_round4.py'), '--data-dir', str(data_dir),
                '--output', str(output), '--max-probes', '20', '--video-followup', '--cdn-migration']),
                ('prepare_selection', common+[str(script/'prepare_round4.py'), '--data-dir', str(data_dir),
                '--output', str(output/'selection'), '--seed', '42', '--holdout-folds', '3', '--deadline', str(selection_deadline)]),
                ('lightgbm_selection', common+[str(script/'multimodal_lgb_round4.py'), '--cache', str(output/'selection/stack.pkl'),
                '--media-features', str(output/'media_features_train.csv'), '--output', str(output), '--trials', str(trials), '--deadline', str(selection_deadline)]),
                ('fixed_capacity_ablation', common+[str(script/'multimodal_lgb_round4.py'), '--cache', str(output/'selection/stack.pkl'),
                '--media-features', str(output/'media_features_train.csv'), '--output', str(output/'fixed_ablation'), '--fixed-only', '--deadline', str(selection_deadline)]),
                ('metadata_group_ablations', common+[str(script/'run_media_groups_round4.py'), '--cache', str(output/'selection/stack.pkl'),
                '--media-features', str(output/'media_features_train.csv'), '--output', str(output/'media_groups'), '--deadline', str(selection_deadline)]),
                ('catboost_selection', common+[str(script/'multimodal_cat_round4.py'), '--cache', str(output/'selection/native.pkl'),
                '--media-features', str(output/'media_features_train.csv'), '--output', str(output), '--deadline', str(selection_deadline)]),
                ('extended_complementarity_search', common+[str(script/'multimodal_extended_round4.py'), '--cache', str(output/'selection/stack.pkl'),
                '--media-features', str(output/'media_features_train.csv'), '--output', str(output/'extended'),
                '--trials', str(extended_trials), '--deadline', str(selection_deadline)])]
    for name, command in commands:
        result = stage(command, selection_deadline, output/f'{name}_master.log')
        stages.append({'stage': name, **result})
        (output/'round4_master_execution.json').write_text(json.dumps({'deadline': deadline, 'stages': stages}, indent=2))
        if result['status'] != 'completed':
            break
    finalizer = script/'finalize_round4.py'
    if finalizer.exists() and time.time() < deadline:
        values = {'--data-dir': data_dir, '--output': output, '--deadline': deadline,
                  '--media-train': output/'media_features_train.csv', '--media-test': output/'media_features_test.csv',
                  '--candidates': output, '--selected': output/'round4_selected.json',
                  '--confirmation': output/'round4_confirmation.json', '--cache': output/'selection',
                  '--submission': output/'submission_round4_confirmed.csv'}
        for mode in ('select', 'confirm', 'refit'):
            if mode == 'refit' and not frozen_state(output).get('promote_over_previous', False):
                stages.append({'stage': mode, 'status': 'not_promoted_no_refit'})
                break
            if mode in ('confirm', 'refit'):
                folder = output/('confirmation' if mode == 'confirm' else 'full')
                prepare = common+[str(script/'prepare_round4.py'), '--data-dir', str(data_dir),
                           '--output', str(folder), '--deadline', str(deadline)]
                prepare += ['--seed', '731', '--holdout-folds', '2'] if mode == 'confirm' else ['--full']
                preparation = stage(prepare, deadline, output/f'prepare_{mode}_master.log')
                stages.append({'stage': f'prepare_{mode}', **preparation})
                if preparation['status'] != 'completed':
                    break
                values['--cache'] = folder
            result = stage(finalizer_command(finalizer, mode, values), deadline, output/f'{mode}_master.log')
            stages.append({'stage': mode, **result})
            if result['status'] != 'completed':
                break
    else:
        stages.append({'stage': 'finalizer', 'status': 'missing_source_or_deadline_no_confirmation_claimed'})
    verification = script/'verify_round4_artifacts.py'
    if verification.exists() and all((data_dir/f'{split}_data.jsonlines').exists() for split in ('train', 'test')):
        result = stage(common+[str(verification), '--data-dir', str(data_dir), '--output', str(output)],
                       deadline, output/'verify_master.log')
        stages.append({'stage': 'verify_artifacts', **result})
    else:
        stages.append({'stage': 'verify_artifacts', 'status': 'missing_source_or_raw_data'})
    archive = script/'archive_round4_predictions.py'
    if archive.exists():
        result = stage(common+[str(archive), '--output', str(output)], deadline, output/'archive_master.log')
        stages.append({'stage': 'archive_predictions', **result})
    report = script/'report_round4.py'
    result = stage(common+[str(report), '--output', str(output)], deadline, output/'report_master.log')
    stages.append({'stage': 'report', **result})
    result = {'deadline': deadline, 'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'stages': stages,
              'training_explicitly_requested': True, 'pixels_claimed_analyzed': False}
    (output/'round4_master_execution.json').write_text(json.dumps(result, indent=2))
    return result


def appendix_cells(master=False):
    import nbformat
    def md(source):
        return nbformat.v4.new_markdown_cell(source, metadata={'round4_appendix': True})
    def code(source):
        return nbformat.v4.new_code_cell(source, metadata={'round4_appendix': True})
    cells = [md('''## 17. Imágenes, videos y texto: qué pudimos medir realmente

Para decidir si una publicación describe un producto nuevo o usado, una foto podría mostrar desgaste; un video podría mostrar el objeto funcionando. Primero hay que obtener los píxeles reales. Este dataset entrega URLs antiguas de fotos y algunos IDs de YouTube. **En esta ronda no se recuperaron imágenes ni frames:** los enlaces de Mercado Libre devolvieron 403 y las comprobaciones de YouTube no acreditaron un stream utilizable.

Hay tres clases de información distintas: **texto** (palabras del título y representaciones SVD/OOF ajustadas dentro del entrenamiento), **metadatos multimedia** (cantidad, dimensiones y orientación de fotos) y **contenido visual** (píxeles/frames, ausentes hoy). Una URL, un ID o el tamaño de una foto no permiten afirmar que analizamos su contenido. Los capítulos anteriores conservan sus figuras y resultados históricos; este capítulo lee los artifacts de la ronda 4.''')]
    if master:
        cells.extend([md('''### Ejecutar la ronda completa de forma explícita

Por defecto este bloque solamente muestra el reporte guardado. Instalar `requirements_round4.txt`. `AP_ROUND4_RUN_TRAINING=1` habilita acceso acotado, preparación de splits, ablaciones LightGBM/CatBoost, capacidad fija de 430 árboles, cuatro bloques y hasta 50 trials de complementariedad, selección congelada, confirmación y refit cuando corresponde; al terminar verifica los artifacts, comprime las probabilidades guardadas y genera el reporte. `AP_ROUND4_MINUTES` controla el presupuesto global (120 por defecto); se reservan 50 minutos para confirmación/refit. El presupuesto puede interrumpir búsquedas: solo se reportan ejecuciones terminadas. La reproducción usa `results/round_v4_reproduction` o `AP_ROUND4_OUTPUT_DIR`: elegir una carpeta nueva si ya hay una selección congelada. No hay entrenamiento visual mientras no existan assets válidos; la rama opcional de CLIP local está documentada por separado. Caches/modelos grandes se regeneran, no se incorporan al repositorio.'''),
            code('''import os
from report_round4 import run_training_pipeline
OUT4=ROOT/'results/round_v4'
RUN_ROUND4_TRAINING=os.getenv('AP_ROUND4_RUN_TRAINING','0')=='1'
if RUN_ROUND4_TRAINING:
    ROUND4_DATA=Path(os.getenv('AP2026Q2_DATA_DIR',str(ROOT.parent/'data')))
    OUT4=Path(os.getenv('AP_ROUND4_OUTPUT_DIR',str(ROOT/'results/round_v4_reproduction')))
    display(run_training_pipeline(ROUND4_DATA,OUT4,
        minutes=float(os.getenv('AP_ROUND4_MINUTES','120')),
        trials=int(os.getenv('AP_ROUND4_TRIALS','12')),
        extended_trials=int(os.getenv('AP_ROUND4_EXTENDED_TRIALS','50')),
        reserve_minutes=float(os.getenv('AP_ROUND4_RESERVE_MINUTES','50'))))
else:
    print('Modo reporte: no se descargan assets ni se entrenan modelos.')
''')])
    output_setup = "OUT4=globals().get('OUT4',ROOT/'results/round_v4')" if master else "OUT4=ROOT/'results/round_v4'"
    cells.extend([code(output_setup+'''
from report_round4 import summarize,coverage_plot,ablation_plot,paired_plot,group_ablation_plot,extended_progress_plot,evaluation_plot,frozen_state,frozen_paired_table,read_json
display(coverage_plot(OUT4))
'''), md('''**Cómo leerlo:** casi todas las filas traen información descriptiva de fotos, pero solamente alrededor de 3% identifica un video. Las barras de píxeles/frames recuperados permanecen en cero. Se comprobó también la migración oficial al CDN actual: cambiar un dominio antiguo por el actual no resolvió el bloqueo. No se inventan fotografías para llenar este faltante.

Construimos 50 medidas numéricas: ancho, alto, área, relación ancho/alto, dispersiones, proporciones de fotos verticales/horizontales y expansión de resolución. **No son 50 señales nuevas independientes:** sobre las 49.000 filas de desarrollo, una es constante y seis coinciden exactamente con columnas previas (incluida esa constante). El inventario de redundancia queda guardado en `media_feature_novelty.csv`. Si los detalles restantes ayudan, podemos añadirlos a los mejores diseños previos de LightGBM y CatBoost. Eso es una prueba de **metadatos**, no de visión artificial.'''),
        code('''round4_metrics=summarize(OUT4)
display(round4_metrics[[column for column in ['name','family','accuracy','threshold','use_media','uses_pixels','uses_video_frames'] if column in round4_metrics]])
display(ablation_plot(OUT4))
if (OUT4/'media_feature_novelty.csv').exists():
    display(pd.read_csv(OUT4/'media_feature_novelty.csv'))
'''), md('''**Experimento pareado:** cada par conserva el algoritmo, la partición de selección y el umbral, y añade las medidas multimedia. La cantidad efectiva de árboles puede cambiar por early stopping; los parámetros base siguen siendo los mismos. Luego, TPE explora configuraciones adicionales. Los mejores resultados de esa búsqueda no sustituyen a la comparación original/metadatos. Las barras son selección local, no Kaggle.'''),
        code('''display(paired_plot(OUT4))
display(group_ablation_plot(OUT4))
if (OUT4/'media_paired_diagnostics.csv').exists():
    display(pd.read_csv(OUT4/'media_paired_diagnostics.csv'))
'''), md('''**Aciertos y errores:** un modelo puede corregir publicaciones que antes confundíamos y, al mismo tiempo, equivocarse en otras. La ganancia neta es corregidos menos introducidos. El p-valor exacto pareado pregunta si esas dos cantidades son compatibles con una diferencia de azar; tras muchas búsquedas no basta para demostrar una mejora general. Por eso congelamos modelo, pesos y umbral antes de la confirmación.

La comparación de bloques usa exactamente 430 árboles y umbral 0,5: pregunta qué pasa al añadir presencia de video, orientación o dimensiones. Es una **ablación de información**, no importancia interna de variables ni prueba de que el modelo vio un video. Los diagnósticos del subconjunto con video presente comparan esa marca histórica, no frames.'''),
        code('''display(extended_progress_plot(OUT4))
'''), md('''**Complementariedad:** la búsqueda extendida evalúa cada árbol nuevo mezclado con la referencia previa: 60% referencia + 40% nuevo, con umbral fijo 0,5. Un modelo aislado apenas mejor puede ser útil si corrige errores diferentes. TPE usa solamente la partición de selección para elegir parámetros y bloques. El ranking por arquitectura considera el mayor de su score aislado y ese blend fijo; después se congela el blend final antes de mirar la confirmación. La curva muestra ejecuciones reales terminadas, no trials interrumpidos ni resultados públicos.'''),
        code('''display(evaluation_plot(OUT4))
display(frozen_paired_table(OUT4))
round4_state=frozen_state(OUT4)
display(round4_state)
display(read_json(OUT4/'run_manifest.json'))
'''), md('''**Selección, confirmación y objetivo:** selección usa 49.000 filas para aprender y 21.000 para comparar candidatos; confirmación usa otra partición de 56.000/14.000. Las familias se separan por `parent_item_id` dentro de cada split. **Las filas de evaluación de las semillas 42 y 731 se solapan:** no son dos holdouts disjuntos ni un test externo intacto; ambas particiones vienen del mismo dataset ya explorado. El solapamiento exacto queda registrado en `selection_confirmation_overlap`. El 94,428% informado por el usuario es una referencia pública de Kaggle y se muestra por separado. No se declara 95% ni una nueva submission hasta que exista evidencia correspondiente.

**Continuación visual real:** `local_visual_embeddings_round4.py` acepta un manifest de imágenes/videos locales alineados por fila. Verifica archivos y pesos oficiales CLIP ViT-B/32, obtiene embeddings de imágenes o tres frames reproducibles por video y registra cobertura. En esta sesión solo se verificó su validación/formato y decodificación local de QA; no se ejecutó inferencia sobre la competencia. El deadline y los estados pendientes forman parte del resultado, no se completan con valores simulados.''')])
    return cells


def update_notebooks(output):
    import nbformat
    from execute_project_notebooks import execute
    output = Path(output)
    result = []
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for name in ('21_maestro_pipeline.ipynb', '22_caso_desde_cero.ipynb'):
            path = ROOT/'notebooks'/name
            notebook = nbformat.read(path, as_version=4)
            historical = [cell for cell in notebook.cells if not cell.metadata.get('round4_appendix')]
            retained_figures = sum('image/png' in item.get('data', {}) for cell in historical
                                   if cell.cell_type == 'code' for item in cell.outputs)
            appendix = appendix_cells(master=name.startswith('21'))
            setup = nbformat.v4.new_code_cell("from pathlib import Path\nimport sys,json\nimport pandas as pd\nfrom IPython.display import display\nROOT=Path.cwd()\nsys.path.insert(0,str(ROOT/'src'))")
            subset = nbformat.v4.new_notebook(cells=[setup, *appendix])
            old_flag = os.environ.get('AP_ROUND4_RUN_TRAINING')
            os.environ['AP_ROUND4_RUN_TRAINING'] = '0'
            try:
                execute(subset)
            finally:
                if old_flag is None:
                    os.environ.pop('AP_ROUND4_RUN_TRAINING', None)
                else:
                    os.environ['AP_ROUND4_RUN_TRAINING'] = old_flag
            notebook.cells = [*historical, *appendix]
            notebook.metadata['round4_execution_verification'] = {'mode': 'report appendix only',
                'historical_figures_retained': retained_figures, 'training_reexecuted': False,
                'visual_competition_inference_executed': False,
                'executed_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
            nbformat.validate(notebook)
            errors = sum(item.output_type == 'error' for cell in notebook.cells if cell.cell_type == 'code' for item in cell.outputs)
            figures = sum('image/png' in item.get('data', {}) for cell in notebook.cells if cell.cell_type == 'code' for item in cell.outputs)
            if errors:
                raise ValueError(f'{name}: {errors} saved notebook errors')
            nbformat.write(notebook, path)
            result.append({'notebook': name, 'errors': errors, 'figures': figures,
                           'historical_figures_retained': retained_figures, 'new_figures': figures-retained_figures})
    finally:
        os.chdir(previous_cwd)
    (output/'notebook_verification.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/round_v4')
    parser.add_argument('--update-notebooks', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = summarize(args.output)
    pairs = matched_ablations(args.output)
    diagnostics = paired_diagnostics(args.output)
    table.to_csv(args.output/'report_metrics.csv', index=False)
    pairs.to_csv(args.output/'report_ablations.csv', index=False)
    diagnostics.to_csv(args.output/'report_paired_diagnostics.csv', index=False)
    state = frozen_state(args.output)
    report = {'completed_configurations': len(table), 'frozen_state': state,
              'manifest': read_json(args.output/'run_manifest.json'), 'pixel_model_trained': False,
              'video_model_trained': False, 'public_reference_user_report': .94428,
              'report_generated_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    (args.output/'report_state.json').write_text(json.dumps(report, indent=2, default=str))
    text = '# Ronda 4: texto y metadatos multimedia\n\n'
    text += f'Configuraciones con métricas guardadas: **{len(table)}**. No se recuperaron píxeles de fotos ni frames de la competencia: no hubo entrenamiento visual o temporal. Se probaron 50 medidas de metadatos sobre los mejores diseños previos; una es constante y seis coinciden con columnas previas (incluida la constante) en las 49.000 filas de desarrollo. No son 50 señales nuevas independientes.\n\n'
    text += 'Las métricas siguientes son de selección local; no se atribuyen a Kaggle. Referencia pública informada por el usuario: **94,428%**.\n\n'
    text += '```text\n'+table[[c for c in ['name','accuracy','threshold'] if c in table]].to_string(index=False)+'\n```\n\n'
    text += 'Estado congelado/confirmación (un artifact ausente queda pendiente):\n\n```json\n'+json.dumps(state,indent=2,default=str)+'\n```\n\n'
    text += 'Fuentes de evidencia: media_access.md, media_protocol.json, métricas por modelo, probabilidades por fila y report_paired_diagnostics.csv. El notebook conserva capítulos históricos y agrega gráficos reales de cobertura, ablaciones y confirmación disponible. La rama local CLIP es opcional y está documentada en local_visual_pipeline.md.\n'
    (args.output/'README.md').write_text(text)
    if args.update_notebooks:
        report['notebooks'] = update_notebooks(args.output)
    print(json.dumps({'completed_configurations': len(table), 'state': state.get('status', 'available'),
                      'notebooks': report.get('notebooks')}, indent=2))


if __name__ == '__main__':
    main()
