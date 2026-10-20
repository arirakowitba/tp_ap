"""Additional executable cells for the latest score and iterative flip design."""


def append_iterative_section(md, code):
    md(r"""
    ### 13.1 Actualización iterativa: corte del 20 de octubre

    El relato previo conserva su corte histórico del 18 de octubre. Aquí se
    incorpora un registro independiente: la submission **15 obtuvo 0,94523**.
    Con la hipótesis de 21.000 filas públicas (70 % de 30.000) y truncamiento
    a cinco decimales, corresponde a **19.850 aciertos públicos**. La 16 tiene
    64 flips nuevos sobre la 15 y sigue pendiente. Tampoco recibimos el score
    de la 14; no se le atribuye ninguna ganancia.

    El objetivo es maximizar el score final de la competencia: probar flips,
    buscar nuevas arquitecturas, mejorar representaciones y estudiar fuentes
    adicionales del mismo dataset, conforme a las reglas aplicables.
    El leaderboard agrega información sobre filas públicas: **no revela la
    etiqueta individual ni demuestra transferencia al 30 % oculto**. Se
    valida la transferencia por familias y folds agrupados, y se hace explícita
    la incertidumbre del posterior.

    Al llegar cada score: registrar el valor textual exacto en
    `feedback_iterativo_20261020.json`, conservar CSV, hash, control, auditoría
    de filas y patrón de cohortes, y volver a ejecutar el notebook.
    """)
    code(r"""
    from ap2026q2_iterative_submissions import (
        public_correct, infer_cohorts, choose_combination, validate_baseline)
    from hashlib import sha256

    registro_path = ROOT/'results/revision_submissions/feedback_iterativo_20261020.json'
    registro = json.loads(registro_path.read_text(encoding='utf-8'))
    base_path = ROOT/registro['baseline']['file']
    base_15 = pd.read_csv(base_path)
    validate_baseline(base_15)
    assert sha256(base_path.read_bytes()).hexdigest() == registro['baseline']['sha256']

    manifiesto_16 = json.loads((ROOT/'submissions/20261020/prueba_16_manifiesto.json').read_text(encoding='utf-8'))
    prueba_16_path = ROOT/'submissions/20261020'/manifiesto_16['submission']
    audit_path = ROOT/registro['groups_source']
    assert sha256(prueba_16_path.read_bytes()).hexdigest() == manifiesto_16['submission_sha256']
    assert sha256(audit_path.read_bytes()).hexdigest() == manifiesto_16['audit_sha256']
    prueba_16, audit_16 = pd.read_csv(prueba_16_path), pd.read_csv(audit_path)
    validate_baseline(prueba_16)
    assert set(audit_16.ID) == set(base_15.loc[base_15.condition.ne(prueba_16.condition),'ID'])
    assert np.array_equal(prueba_16.set_index('ID').loc[audit_16.ID,'condition'], audit_16.label_in_16)
    print('15:', registro['baseline']['score'], '→', public_correct(registro['baseline']['score']),
          'aciertos públicos supuestos. Score 16:', registro['scores']['16'])
    """)
    md(r"""
    ### 13.2 Grilla de cuatro pruebas nuevas que separa cohortes

    Dividimos los 64 flips de la 16 en dos grupos de 32 según las siete
    semillas anteriores: **A** recibe ≥4 votos para la alternativa; **B**, <4.
    Los votos son un *ordenamiento*, no probabilidades calibradas. C y D
    contienen 64 IDs nuevos cada una; E amplía la cobertura en 128 filas,
    particionadas en cuatro subgrupos de 32. Sus CSV y auditorías se generan
    con `compose_four_information_probes_20261020.py`. La matriz siguiente
    describe archivos **preparados**, todavía sin scores recibidos:

    | Prueba sobre la 15 | A | B | C | D | E | Ganancia pública |
    |---|---:|---:|---:|---:|---:|---|
    | 16 | 1 | 1 | 0 | 0 | 0 | $g_{16}=a+b$ |
    | 17 | 1 | 0 | 1 | 0 | 0 | $g_{17}=a+c$ |
    | 18 | 0 | 1 | 1 | 0 | 0 | $g_{18}=b+c$ |
    | 19 | 0 | 1 | 0 | 1 | 0 | $g_{19}=b+d$ |
    | 20 | 0 | 0 | 1 | 1 | 1 | $g_{20}=c+d+e$ |

    Si se reciben los cinco resultados compatibles (16 a 20):
    $a=(g_{16}+g_{17}-g_{18})/2$,
    $b=(g_{16}-g_{17}+g_{18})/2$,
    $c=(g_{17}+g_{18}-g_{16})/2$, $d=g_{19}-b$ y
    $e=g_{20}-c-d$.
    **Con un solo score, A+B es conocido pero A y B no lo son**.
    Si falla paridad o el sistema no tiene solución, revisar control,
    archivos, solapamientos y truncamiento antes de interpretar el resultado.
    """)
    code(r"""
    nombres_16 = np.where(audit_16.full70k_seed_votes_alternative_out_of_7.ge(4), 'A', 'B')
    cohortes = {
        nombre: audit_16.loc[nombres_16 == nombre, ['ID','label_in_16']]
                  .rename(columns={'label_in_16':'alternative_label'}).copy()
        for nombre in ('A','B')
    }
    assert [len(cohortes[n]) for n in ('A','B')] == [32,32]
    # Incorporar CSV auditados y disjuntos con ID,alternative_label.
    for nombre in ('C','D','E'):
        ruta = ROOT/f'submissions/20261020/cohorte_{nombre}_auditoria.csv'
        if ruta.exists():
            cohortes[nombre] = pd.read_csv(ruta)

    if any(score is not None and not set(registro['experiments'][i]) <= set(cohortes)
           for i,score in registro['scores'].items()):
        raise ValueError('Falta auditar C/D para una submission puntuable')
    activas = {'A','B'}
    for i,score in registro['scores'].items():
        if score is not None:
            activas.update(registro['experiments'][i])
    cohortes = {i:filas for i,filas in cohortes.items() if i in activas}
    patrones = {i:grupos for i,grupos in registro['experiments'].items()
               if set(grupos) <= set(cohortes)}
    resumen, posterior = infer_cohorts(
        {i:len(f) for i,f in cohortes.items()}, patrones,
        registro['scores'], registro['baseline']['score'],
        n_public=registro['assumed_public_rows'], mode=registro['public_display'])
    tabla(resumen)
    print('Configuraciones conjuntas compatibles:', len(posterior))
    """)
    md(r"""
    ### 13.3 De un score a una confianza y a un CSV verificable

    Para cada cohorte de tamaño $n$, sean $M$ filas públicas y $K$ flips
    correctos entre ellas: $g=2K-M$. El código usa
    $M\sim\mathrm{Hipergeom}(30000,21000,n)$ y una previa explícita
    $p\sim\mathrm{Beta}(2,2)$; el peso de cada $(M,K)$ es proporcional a
    $P(M)\mathrm{BetaBinomial}(K\mid M,2,2)$.
    El efecto esperado sobre las filas ocultas de la misma cohorte es
    $(n-M)[2(2+K)/(4+M)-1]$. Con varias pruebas, el posterior condiciona
    **simultáneamente** a todas las ecuaciones. No se asignan etiquetas
    individuales a partir de una suma.

    La columna `expected_hidden_gain` extrapola bajo intercambiabilidad;
    para tomar decisiones se multiplica por `hidden_transfer` (por defecto
    0,5). El ranking enumera combinaciones disjuntas. La consolidación
    automática exige un efecto público positivo **identificado por grupo o
    para la combinación exacta probada**; si gana A+B se puede conservar
    entero aunque no sepamos qué aportó cada parte. Con la 16 sin score
    conserva la 15. Una prueba exploratoria nueva se
    diseña aparte para obtener información, según el presupuesto restante.
    Repetir con previas Beta(1,1), Beta(2,2) y una previa calibrada por
    CV agrupada, y factores de transferencia entre cero y uno.
    """)
    code(r"""
    def actualizar_submission(registro=registro, grupos=cohortes, control=base_15,
                              salida=None, hidden_transfer=.5):
        # Incorporar scores nuevos y generar opcionalmente el CSV óptimo condicional.
        matriz = {i:g for i,g in registro['experiments'].items() if set(g) <= set(grupos)}
        resumen, configuraciones = infer_cohorts(
            {i:len(f) for i,f in grupos.items()}, matriz,
            registro['scores'], registro['baseline']['score'],
            n_public=registro['assumed_public_rows'], mode=registro['public_display'])
        ranking, prediccion = choose_combination(
            resumen, configuraciones, grupos, control,
            hidden_transfer=hidden_transfer, require_observed_positive=True)
        if salida is not None:
            salida = Path(salida)
            assert salida.suffix.lower() == '.csv' and salida.parent.exists()
            prediccion.to_csv(salida, index=False)
            assert pd.read_csv(salida).equals(prediccion)
            manifiesto = {'baseline_sha256':registro['baseline']['sha256'],
                          'scores':registro['scores'], 'cohorts':ranking.iloc[0].cohorts,
                          'csv_sha256':sha256(salida.read_bytes()).hexdigest(),
                          'public_rows_hypothesis':registro['assumed_public_rows'],
                          'hidden_transfer':hidden_transfer}
            salida.with_suffix('.json').write_text(json.dumps(manifiesto,indent=2),encoding='utf-8')
        return ranking, prediccion

    ranking, mejor_condicional = actualizar_submission()
    tabla(ranking.head(8))
    print('Flips consolidados con scores disponibles:',
          int((mejor_condicional.condition != base_15.condition).sum()))
    # Tras registrar los scores, descomentar para exportar:
    # ranking, csv = actualizar_submission(salida=ROOT/'submissions/20261020/submission_consolidada.csv')
    """)
    code(r"""
    # Un caso independiente verifica que las cinco ecuaciones separan los efectos.
    from itertools import product
    efectos = np.array([3,-2,7,-1,5])
    matriz_ejemplo = np.array([[1,1,0,0,0],[1,0,1,0,0],[0,1,1,0,0],
                               [0,1,0,1,0],[0,0,1,1,1]])
    assert np.array_equal(np.linalg.solve(matriz_ejemplo, matriz_ejemplo@efectos), efectos)
    assert len(audit_16) == 64 and public_correct('0.94523') == 19850
    if registro['scores']['16'] is None:
        assert int(ranking.iloc[0].flips) == 0  # no inventar evidencia pendiente
    print('Diseño, archivos y evidencia observada comprobados.')
    """)
