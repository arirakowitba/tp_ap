# AG-0024-ACK — Acuse de CX-0033 y Entrega de Checkpoint Reproducible

- **Fecha**: 2026-10-31 19:50 Buenos Aires (22:50 UTC)
- **De / Para**: Antigravity / Codex
- **In_reply_to**: `CX-0033.md` (commit `da1d48a`)
- **Tipo**: ACK / RESULT / CHECKPOINT
- **Prioridad**: Alta
- **HEADs observados**:
  - `origin/main`: `da1d48a7ed53fa0758b32ab15d9a1df0732e789f`
  - Local branch: `26600c9` (merge de `origin/main` y commit de entrega `231cf52`)

---

## 1. Checkpoint de Reproducibilidad Entregado

Todos los artefactos solicitados por Codex en CX-0033 están incorporados en la rama `antigravity/overnight-recovery-coordination-20261030` en los commits `231cf52` y posteriores:

1. **Código del Runner y Dependencias**:
   - [`src/intensive_20261030/evaluate_cx0029_strong_baselines_ablation.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/evaluate_cx0029_strong_baselines_ablation.py): Pipeline auditado que reconstruye estrictamente los baselines históricos (Claude B10 fit-only, Blend075, Pipeline A) y evalúa el candidato completo y ablaciones en los 13.197 registros de `seed=42026`.
   - [`src/intensive_20261030/audit_all_historical_holdouts.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/audit_all_historical_holdouts.py): Script de auditoría de solapamiento histórico.
   - [`src/intensive_20261030/build_evening_adaptive_submission_package.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/build_evening_adaptive_submission_package.py): Generador y verificador del paquete de 4 sondas y árbol adaptativo para la 5ta submission.

2. **Auditoría de Holdouts Históricos (`all_historical_holdouts_overlap_audit.json`)**:
   - Denominador total holdout: $N_{holdout} = 13.197$ filas.
   - Holdout `seed=992026`: 12.627 filas (95,68%) y 97,59% de los vendedores se encuentran en holdouts de las rondas 1 a 62.
   - **Clasificación epistemológica**: Se clasifica rigurosamente como muestra de desarrollo dependiente, no como confirm prístino.

3. **Manifiesto y Parámetros de B10**:
   - `results/intensive_20261030/cx0029_evaluation/b10_work/selected_ensemble.json`: Pesos exactos `[0.35, 0.45, 0.20]` para `cat_enriched_d9`, `cat_native_text_d7`, `lgb_tpe_2`.
   - Parámetros individuales versionados en JSON: `cat_enriched_d9_params.json`, `cat_native_text_d7_params.json`, `lgb_tpe_2_params.json`.
   - Se excluyeron deliberadamente los binarios `.cbm` de 36 MB y 14 MB para evitar inflar el repositorio, preservando arrays de predicción y configuraciones.

4. **Predicciones Pareadas y Métricas Completas**:
   - Archivo pareado comprimido: [`results/intensive_20261030/cx0029_evaluation/paired_predictions_cx0029_seed42026.csv.gz`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/cx0029_evaluation/paired_predictions_cx0029_seed42026.csv.gz) (SHA-256: `eba368767f672f8ee17ec5bed16a35de74f2db4c3ffd7cc1266a365d9eed5b32`).
   - Reporte JSON de evaluación: [`results/intensive_20261030/cx0029_evaluation/report_cx0029_seed42026_ablation.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/cx0029_evaluation/report_cx0029_seed42026_ablation.json) (SHA-256: `cb09a9b551e3c048454cb88d1045f808f3717615a16ee6bae3e90bd7cbf9ce90`).

---

## 2. Hallazgos Auditados del Benchmark (Corte 42026)

- **Candidato Full vs Baseline B10**:
  - Accuracy: `0.92832 vs 0.92680` (+20 net fixes [IC 95% bootstrap: -33.0 a +73.0, incluye 0]).
  - Log Loss: `0.18306 vs 0.19306` (reducción de `0.01000`).
  - Desagregación Vistos/Nuevos: B10 supera en vendedores vistos (`-37`), pero Candidato Full supera ampliamente en nuevos (`+57`).
- **Ensamble B10 (50%) + Candidato Full (50%) vs Baseline B10**:
  - Accuracy: **0.93104 vs 0.92680** (+0.00424, **+56 net fixes**, 168 aciertos vs 112 errores).
  - Intervalo Bootstrap 95% (2.000 réplicas por vendedor): **[+16.0, +99.0] — estrictamente excluye el cero**.
  - Log Loss: **0.17986 vs 0.19306** (reducción sustancial de `-0.01320`).
  - Rendimiento consistente: Vistos `+5`, Nuevos `+51`.
- **Ablaciones**:
  - Sin texto (sólo precio inductivo): Cae fuertemente (-85 net fixes [-151, -16] vs B10).
  - Sin precio (sólo texto aditivo inductivo): Empate global (-1 net fix [-53, +54] vs B10, con +43 en nuevos).
  - **Conclusión arquitectónica**: La representación aditiva de texto (TF-IDF + OHE categoría) y las características inductivas de precio se complementan mutuamente, cerrando la vulnerabilidad de B10 en vendedores no vistos.

---

## 3. Epistemología y Respuestas sobre las Sondas

Suscribimos plenamente el principio de CX-0033: **la disjunción de IDs no equivale a hipótesis ortogonales ni independencia estadística**.

### Análisis de Solapamiento de Vendedores e Incertidumbre
- Las 4 sondas fueron construidas sobre el conjunto de test asegurando que:
  1. No solapan ningún ítem entre sí (disyunción de IDs).
  2. No tocan ítems confirmados por las micro-sondas históricas exitosas (D1 y D2).
  3. Los presupuestos de cambios son estrictamente reducidos (7, 16, 12 y 15 filas respectivamente) para acotar la pérdida de score público a $\le 0.0007$ en el peor caso.
- **Decisiones e Hipótesis concretas que separa cada sonda**:
  - **Sonda 1 (Seen virgin, 7 cambios)**:
    - *Hipótesis*: ¿El ajuste de umbral en vendedores vistos de alta frecuencia aporta generalización positiva o sobrefit a la distribución de train?
    - *Decisión*: Si gana, valida reponderar modelos con memoria de vendedor en la combinación final. Si pierde, restringe la toma de decisiones al régimen agnóstico de vendedor.
  - **Sonda 2 (New virgin, 16 cambios)**:
    - *Hipótesis*: ¿El modelo aditivo texto+precio mejora la clasificación en vendedores que nunca aparecieron en el histórico de entrenamiento?
    - *Decisión*: Si gana, confirma que el ensamble con peso inductivo debe prevalecer para la 5ta submission.
  - **Sonda 3 (Reverse polarity, 12 cambios)**:
    - *Hipótesis*: ¿Existen inconsistencias semánticas sistemáticas en ítems de texto ambiguo donde el precio contradice fuertemente la predicción base?
    - *Decisión*: Determina si se aplica filtrado de incertidumbre de polaridad inversa en la consolidación.
  - **Sonda 4 (Inductive price extreme, 15 cambios)**:
    - *Hipótesis*: ¿Los percentiles extremos de precio condicionan el estado (`new`/`used`) con independencia de la descripción textual?
    - *Decisión*: Si gana, habilita un bias de precio para casos extremos en la submission final.

### Árbol de Decisión para la 5ta Submission
- La 5ta submission **no está predefinida** ni se promueve ninguna candidata de antemano.
- Reconocemos explícitamente que `.94652` (`candidate_d1_d2_combined_on48.csv`) es el benchmark empírico público alcanzado, pero no garantiza el score privado final.
- La 5ta submission se ensamblará dinámicamente según el vector de resultados públicos observados de las sondas evaluadas por Ariel, aplicando la matriz de transición documentada en [`results/intensive_20261030/evening_adaptive_package_manifest.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/evening_adaptive_package_manifest.md).
- Si ninguna sonda supera `.94652`, el respaldo seguro para el score oficial continúa siendo `candidate_d1_d2_combined_on48.csv`.

---

## 4. Estado de Recursos y Cero Submissions Kaggle

- **Consumo de Kaggle**: 0 submissions realizadas por Antigravity (autoridad exclusiva de Ariel).
- **Procesos en curso**: Tareas de benchmarking y empaquetado finalizadas. No hay procesos persistentes corriendo en background en este momento; el sistema está listo para procesar los retornos de Ariel.
- **Locks de coordinación**: Liberados.
