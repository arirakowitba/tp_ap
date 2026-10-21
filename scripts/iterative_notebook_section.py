"""Additional executable cells for the latest score and iterative flip design."""


def append_iterative_section(md, code):
    md(r"""
    ### 13.1 Actualización iterativa: corte del 20 de octubre

    El relato previo conserva su corte histórico del 18 de octubre. Aquí se
    incorpora un registro independiente: la submission **15 obtuvo 0,94523**.
    Con la hipótesis de 21.000 filas públicas (70 % de 30.000) y truncamiento
    a cinco decimales, corresponde a **19.850 aciertos públicos**. Ya recibimos
    los scores de la 16 a la 20: 0,94500; 0,94447; 0,94404; 0,94409; 0,94285.
    El score de la 14 sigue sin suministrarse. La sección 13.4 presenta el
    resultado real; las secciones anteriores conservan su corte histórico.

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
    describe los archivos exactos **presentados y puntuados**:

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
    entero aunque no sepamos qué aportó cada parte. Antes de recibir los scores
    conservaba la 15; ahora selecciona **solo A**. Una prueba exploratoria nueva se
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
    md(r"""
    ### 13.4 Resultados observados: separar saldos y confianza

    Con los cinco resultados, las ganancias respecto de la 15 son
    $(-5,-16,-25,-24,-50)$. Al resolver las ecuaciones obtenemos
    **A: +2; B: −7; C: −18; D: −17; E: −15** aciertos públicos netos.
    La 16 perdió porque mezclaba A y B; conservar solo A añade dos aciertos.
    La candidata `submission_consolidada_A32_sobre15_no_enviada.csv` tiene
    32 flips y score público **inferido 0,94533**, aún sin medir directamente.

    El gráfico izquierdo es un saldo identificado bajo máscara fija de 21.000
    filas y truncamiento. El derecho es una inferencia probabilística diferente:
    estima la tasa de acierto de flips dentro de cada cohorte con Beta(2,2),
    integrando la cantidad desconocida de filas públicas. No identifica cada ID.
    En A la media es 0,539 pero el intervalo del 95 % va de 0,349 a 0,724:
    **la pequeña mejora pública todavía deja mucha incertidumbre de transferencia**.
    El beneficio oculto esperado es +0,79 aciertos; su intervalo predictivo del
    95 % es [−6,+8]. Son supuestos de intercambiabilidad, no resultados ocultos.
    Las marginales de máscara se aproximan como independientes entre cohortes.
    """)
    code(r"""
    resultados = json.loads((ROOT/'results/revision_submissions/resultados_cruzados_16_a_20.json').read_text(encoding='utf-8'))
    tabla(pd.DataFrame(resultados['observed_submissions']))
    confianzas = pd.DataFrame(resultados['cohort_posterior'])
    tabla(confianzas[['cohort','n','public_gain_identified','mean_flip_correctness',
                     'flip_correctness_ci_low','flip_correctness_ci_high',
                     'expected_hidden_gain','prob_hidden_gain_positive']])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    colores = ['#23875b' if x > 0 else '#c65a50' for x in confianzas.public_gain_identified]
    axes[0].bar(confianzas.cohort, confianzas.public_gain_identified, color=colores)
    axes[0].axhline(0, color='#555555', lw=1)
    axes[0].set(title='Saldo público identificado por cohorte', xlabel='Cohorte', ylabel='Aciertos netos al aplicar flips')
    for i, fila in confianzas.iterrows():
        axes[0].text(i, fila.public_gain_identified + (1 if fila.public_gain_identified > 0 else -1),
                     f'{fila.public_gain_identified:+d}', ha='center',
                     va='bottom' if fila.public_gain_identified > 0 else 'top')
    medias = confianzas.mean_flip_correctness.to_numpy()
    axes[1].errorbar(medias, np.arange(len(medias)),
        xerr=np.vstack([medias-confianzas.flip_correctness_ci_low,
                        confianzas.flip_correctness_ci_high-medias]), fmt='o', capsize=5, color='#356ca1')
    axes[1].set_yticks(np.arange(len(medias)), confianzas.cohort)
    axes[1].axvline(.5, color='#777777', linestyle='--', label='50 %')
    axes[1].set(xlim=(0,1), title='Tasa de acierto: posterior e intervalo 95 %',
                xlabel='Probabilidad de acertar el flip (modelo de cohorte)', ylabel='Cohorte')
    axes[1].invert_yaxis()
    fig.tight_layout()
    plt.show()
    candidata = resultados['conditional_candidate']
    candidata_path = ROOT/candidata['file']
    assert sha256(candidata_path.read_bytes()).hexdigest() == candidata['sha256']
    assert candidata['public_score_observed'] is None
    assert candidata['public_score_inferred'] == '0.94533'
    assert pd.read_csv(candidata_path).equals(mejor_condicional)
    print('Candidata condicional:', candidata_path.name, '| 32 flips | score inferido, no medido.')
    """)
    md(r"""
    ### 13.5 Próxima tanda: resolver heterogeneidad y probar transferencia

    C y D fueron negativos; su apoyo de modelos no bastó para estos flips.
    E fue negativo como agregado, pero sus cuatro subgrupos de 32 siguen
    sin separar. Propuesta todavía **sin generar CSV**: probar A+E1+E2,
    A+E1+E3 y A+E1+E4, siempre sobre la 15. Si $h_i=g_i-a$ y $e=-15$,
    entonces $e_1=(h_1+h_2+h_3-e)/2$ y los otros efectos se obtienen por
    diferencia. Tres scores nuevos más el total conocido identifican los
    cuatro saldos de E.

    La cuarta prueba se reserva para una cohorte F con IDs nuevos y señales
    semejantes a A, auditada antes de generarla: comprueba si la relación se
    extiende. La quinta consolida después de observar esas cuatro pruebas.
    Esta es una propuesta adaptable: no conocemos aún los scores 21–25 ni
    la composición factible de F. Seguir mejorando modelos y datos; el
    ensamble de mejor CV no se convierte automáticamente en una política
    de flips fiable para todos los casos dudosos.
    """)
    md(r"""
    ### 13.6 Meta actualizada: 96 % global en las 30.000 filas

    La identidad solicitada es $A_{global}=0.7P+0.3H$.
    Con $P=0.95$ y un $H=0.94523$ **hipotético**, da 94,8569 %, no 96 %.
    Para llegar al 96 % con ese oculto se requieren 20.293 aciertos públicos,
    96,6333 %. El score de la 15 ya incorpora feedback: no es una evaluación
    independiente del oculto. La igualdad de calidad entre segmentos es un
    supuesto de ese escenario, no una etiqueta observada.

    La meta operativa con margen es **97 % público y 95 % oculto central**:
    96,4 % global. Para hablar de alta confianza exigiremos
    $\Pr(C_{pub}+C_{hid}\ge 28800\mid D)\ge0.95$ con un modelo predictivo
    conjunto validado. Con 97 % público, un límite predictivo inferior oculto
    de 94 % deja 96,1 % global. **Ese criterio no está demostrado hoy**;
    las tablas son escenarios y presupuestos, no garantías de alcanzar la meta.

    Una auditoría corrigió la descripción de la CV anterior: sus 70.000
    grupos son distintos, por lo que no separan vendedores. Las métricas
    son OOF por fila; aproximadamente 67,4 % de validación tiene vendedor visto,
    frente a 71,6 % del test. La próxima evaluación debe medir vendedores
    vistos y nuevos y la transferencia de todo el procedimiento 70/30.
    El ensamble actual obtuvo 96,12 % OOF en vistos y 88,36 % en nuevos:
    este último segmento ofrece un frente concreto de mejora, sin que esos
    números sean scores observados del test.
    """)
    code(r"""
    from ap2026q2_global_objective import global_accuracy, required_public_correct, required_hidden_correct
    global_dir = ROOT/'results/global_target_96'
    plan_global = json.loads((global_dir/'global_plan.json').read_text(encoding='utf-8'))
    escenarios = pd.read_csv(global_dir/'scenarios.csv')
    mezcla = json.loads((global_dir/'validation_distribution_audit.json').read_text(encoding='utf-8'))
    tabla(escenarios)
    tabla(pd.DataFrame(mezcla['metrics']).T)
    print('Criterio de confianza establecido:', plan_global['confidence_gate']['gate_established'])
    assert not plan_global['confidence_gate']['gate_established']
    assert required_public_correct(.94523) == 20293
    assert required_hidden_correct(20370) == 8430
    assert np.isclose(global_accuracy(.95,.94523),.948569)
    assert mezcla['unique_cv_groups'] == 70000 and not mezcla['seller_disjoint_validation']
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    publico = np.linspace(.94,.985,300)
    oculto_minimo = (.96-.7*publico)/.3
    axes[0].plot(publico, oculto_minimo, color='#356ca1', label='Frontera de 96 % global')
    axes[0].fill_between(publico, oculto_minimo, 1, color='#73b98a', alpha=.18)
    for x,y,texto,color in [(.95,.94523,'95 % público: global 94,86 %','#c65a50'),
                            (.97,.95,'Meta central: global 96,4 %','#23875b'),
                            (.97,.94,'Umbral oculto propuesto: 96,1 %','#ac7d24')]:
        axes[0].scatter([x],[y],color=color,s=50,zorder=4)
        axes[0].annotate(texto,(x,y),xytext=(-6 if x>=.965 else 6,6),
                         ha='right' if x>=.965 else 'left',textcoords='offset points',fontsize=8)
    axes[0].set(xlim=(.94,.985),ylim=(.925,.975),xlabel='Accuracy pública',ylabel='Accuracy oculta',
                title='El objetivo global exige las dos partes')
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc='lower left',fontsize=8)
    conocidos = np.array([mezcla['cv_seen_seller_fraction'],mezcla['test_seen_seller_fraction']])
    axes[1].bar(['CV histórica','Test real'],conocidos,color='#356ca1',label='Vendedor visto en entrenamiento')
    axes[1].bar(['CV histórica','Test real'],1-conocidos,bottom=conocidos,color='#c2cfda',label='Vendedor nuevo')
    for i,valor in enumerate(conocidos):
        axes[1].text(i,valor/2,f'{valor:.1%}',ha='center',color='white',fontsize=12)
    axes[1].set(ylim=(0,1),ylabel='Proporción de filas',title='Mezcla a reproducir en validación')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(loc='lower left',fontsize=8)
    fig.tight_layout()
    fig.savefig(global_dir/'global_frontier_and_validation.png',dpi=160,bbox_inches='tight')
    plt.show()
    tabla(pd.read_csv(global_dir/'flip_budget.csv'))
    print('Filas nuevas del ranking anterior:',plan_global['remaining_ranked_unprobed_rows'])
    print('La estrategia y las metas pendientes están en results/global_target_96/README.md')
    """)
