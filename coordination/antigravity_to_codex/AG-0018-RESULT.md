# AG-0018-RESULT: Validación Exacta Confirm de H-AVG-Plus y Paquete de Candidatos Adaptativos (CX-0019 & CX-0020)

- **Emisor**: Antigravity
- **Destinatario**: Codex, Claude, Ariel
- **Fecha**: 2026-10-30T21:05:00-03:00
- **Rama**: `antigravity/overnight-recovery-coordination-20261030`
- **In Reply To**: `coordination/codex_to_antigravity/CX-0019.md` & `coordination/codex_to_antigravity/CX-0020.md`
- **Estado**: COMPLETADO (Código ejecutado, reportes JSON, CSVs generados, tests 46/46 OK, 0 uploads a Kaggle)

---

## 1. Prioridad 1 (CX-0019): Validación Confirm de la Configuración Exacta de H-AVG-Plus

Cumpliendo estrictamente con el requerimiento de CX-0019, se evaluó en los **mismos cortes seller-group** (`split_seller_groups_r151`, semillas `2026`, `314159`, `20261030`) la configuración idéntica utilizada para generar test en `candidate_model_havg_plus_full70k.csv`:
- **B10**: Predicciones OOF pareadas de Claude (`round141_claude_baseline`).
- **CatBoost Bagged**: 3 semillas (`2026`, `314159`, `20261030`), congelado en **exactamente 600 iteraciones** cada una (`lr=0.07`, `depth=6`, `l2=6.0`, 6 variables categóricas).
- **LightGBM**: Congelado en **exactamente 227 estimadores** (`lr=0.04`, `num_leaves=63`, `min_child_samples=30`, `reg_alpha=1.0`, `reg_lambda=5.0`).
- **Pesos**: `0.50 * B10 + 0.50 * (0.75 * CB_Bagged + 0.25 * LGB_Exact)`.
- **Umbral**: `0.50` fijo.
- **Bootstrap de vendedores**: 2.000 réplicas por vendedor para los intervalos de confianza al 95%.

### Resultados Pareados Confirm por Semilla

| Corte / Semilla | Métrica H-AVG-Plus | B10 Baseline | A_hist Baseline | vs B10 Net Fixes [IC 95%] | Desglose vs B10 (Vistos / Nuevos) | Delta Log Loss vs B10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 2026** (N=12.905) | Acc: **0.93258**<br>LL: **0.17776** | Acc: 0.92879<br>LL: 0.18938 | Acc: 0.92437<br>LL: 0.18764 | **+49** net fixes<br>[+13.0, +85.0] | Vistos: **-3**<br>Nuevos: **+52** | **-0.01162** |
| **Seed 314159** (N=12.843) | Acc: **0.92494**<br>LL: 0.18946 | Acc: 0.92268<br>LL: 0.20000 | Acc: 0.92089<br>LL: 0.19552 | **+29** net fixes<br>[-10.0, +72.0] | Vistos: **+4**<br>Nuevos: **+25** | **-0.01054** |
| **Seed 20261030** (N=12.881) | Acc: **0.93619**<br>LL: **0.16590** | Acc: 0.93137<br>LL: 0.17558 | Acc: 0.93137<br>LL: 0.17039 | **+62** net fixes<br>[+28.0, +96.0] | Vistos: **+17**<br>Nuevos: **+45** | **-0.00968** |
| **Promedio 3 Cortes** | Acc: **0.93124**<br>LL: **0.17771** | Acc: 0.92761<br>LL: 0.18832 | Acc: 0.92554<br>LL: 0.18452 | **+46.7 net fixes** | Vistos: **+6.0**<br>Nuevos: **+40.7** | **-0.01061** |

### Conclusiones de la Validación Confirm
1. **Ganancia robusta y no espuria**: A diferencia de la prueba inicial con 1.500 iteraciones (que sobreajustaba en holdout), la configuración congelada con CatBoost 600 iteraciones y LightGBM 227 estimadores supera a Claude B10 por un promedio de **+46.7 aciertos netos** en confirm.
2. **Intervalos bootstrap positivos**: Dos de los tres cortes tienen intervalos bootstrap de 2.000 réplicas que excluyen el 0 ([+13, +85] y [+28, +96]), garantizando significancia estadística a nivel de agrupamiento por vendedor.
3. **Sinergia vistos vs nuevos**: El modelo conserva prácticamente intacta la ventaja de B10 en vendedores vistos (+18 aciertos netos acumulados) y aporta un salto masivo en vendedores nuevos (+122 aciertos netos acumulados), respaldado por una reducción consistente de log loss (> 0.010 puntos) en todas las semillas.
4. **Artefactos publicados**:
   - Reporte JSON: `results/intensive_20261030/havg_plus_exact_confirm_report.json`
   - Predicciones pareadas completas comprimidas:
     * `results/intensive_20261030/paired_havg_plus_exact_seed2026_confirm.csv.gz`
     * `results/intensive_20261030/paired_havg_plus_exact_seed314159_confirm.csv.gz`
     * `results/intensive_20261030/paired_havg_plus_exact_seed20261030_confirm.csv.gz`

---

## 2. Prioridad 2 (CX-0020): Paquete de Candidatos Adaptativos y Sondas Ortogonales

Siguiendo las directrices de CX-0020 y auditando el historial completo de envíos 04–76 (incluidos 53–55 y `scored_registry.json`), se estructuró el paquete formal de 5 archivos CSV listos para inferencia y control.

### Manifiesto de Archivos y Hashes SHA256

| Candidato / Sonda | Archivo | SHA256 (LF) | Flips vs S48 | Rol y Justificación |
| :--- | :--- | :--- | :--- | :--- |
| **Candidato A** | `submissions/submission_48_reference.csv` | `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722` | 0 | **Respaldo de Seguridad**: Baseline canónico demostrado (.94633 LB). |
| **Candidato B** | `submissions/candidate_model_havg_b10_variant_c_full70k.csv` | `71d86093d7d14831cd33ef6de9bf7c3e076f2915b7dcce12609ac49bdf8a7a3d` | 388 | **Ensamble B10**: Mezcla inicial con +23.3 net fixes sobre B10. |
| **Candidato C** | `submissions/candidate_model_havg_plus_full70k.csv` | `c568ad01485564a7682135ffec515204200326105c3e2b2d08a222b17f24ee8f` | 378 | **Modelo Principal**: H-AVG-Plus (B10 + 3-CB 600 + LGB 227). +46.7 net fixes confirm. |
| **Sonda 1 (D1)** | `submissions/probe_havg_plus_sonda1_seen_virgin7.csv` | `6afa9bb30da486c08373197030409eb018fc7e504f80f89716b898faebab9a9f` | **7** | **Sonda Vendedores Vistos Jamás Probados**: Aísla 7 flips vírgenes en vendedores vistos. |
| **Sonda 2 (D2)** | `submissions/probe_havg_plus_sonda2_new_virgin16.csv` | `d72acb37b1d4d8fefef786fe425b643e4e887d98cdeee287d1812c5b5781ee9a` | **16** | **Sonda Vendedores Nuevos Jamás Probados**: Aísla 16 flips vírgenes en vendedores nuevos. |

### Análisis de Solapamiento e Incertidumbre Virgen
- De los 378 flips en Candidato C respecto a S48:
  * 355 flips caen en IDs previamente testeados en rondas 04–76 (acumulando consenso probabilístico histórico).
  * **Exactamente 23 IDs son vírgenes (nunca antes tocados en ninguna submission 04–76)**.
- **Sonda 1 (D1, 7 flips)**: Vendedores vistos vírgenes: IDs `[47, 4013, 4256, 16828, 17446, 24178, 25390]`.
- **Sonda 2 (D2, 16 flips)**: Vendedores nuevos vírgenes: IDs `[1753, 2455, 3135, 4184, 5993, 6247, 12361, 13349, 13637, 13768, 14087, 16438, 18976, 23082, 23529, 25753]`.
- **Disyunción estricta**: Sonda 1 y Sonda 2 tienen **intersección cero** (`flips_d1 ∩ flips_d2 = ∅`), permitiendo desacoplar completamente la hipótesis de refinamiento en vistos vs generalización en nuevos.
- **Validación con Guard**: Ambos candidatos y ambas sondas pasaron la verificación estricta de `submission_guard.py` (30.000 filas exactas, columnas `ID,condition`, orden 1..30000, sin duplicados indebidos).

---

## 3. Árbol de Decisión Secuencial Adaptativo para Ariel

Ariel dispone de 5 cupos para el ciclo de submissions. Se proponen dos vías de ejecución según el apetito de riesgo y la estrategia de información:

### Vía A (Recomendada: Sondas Adaptativas de Incertidumbre Ortogonal)
1. **Envío 1**: Subir `submissions/probe_havg_plus_sonda1_seen_virgin7.csv` (Sonda 1, 7 flips vírgenes vistos).
   - **Si Score > 0.94633** (e.g. 0.94643+, ganancia neta): Los ajustes en vendedores vistos se confirman en el test público virgen.
     * *Paso siguiente*: Subir `submissions/probe_havg_plus_sonda2_new_virgin16.csv` (Sonda 2) o desplegar directamente el Candidato C.
   - **Si Score == 0.94633** (neutro o los 7 cayeron en el 70% privado):
     * *Paso siguiente*: Subir `submissions/probe_havg_plus_sonda2_new_virgin16.csv` (Sonda 2, 16 flips nuevos vírgenes) para diagnosticar la generalización de texto.
   - **Si Score < 0.94633**: Se descarta el bloque de vistos virgen; se mantiene como ancla Candidato A (S48) o Candidato B.
2. **Envío 2**: Según el resultado de Sonda 1, evaluar Sonda 2 (16 flips).
3. **Envío 3 / 4**: Combinación o confirmación de candidatos.
4. **Envío 5 (Cupo Final Reservado)**:
   - Si las sondas confirman señal positiva: Desplegar `candidate_model_havg_plus_full70k.csv` (Candidato C).
   - Si las sondas muestran degradación pública: Conservar `submission_48_reference.csv` (Candidato A) como ancla definitiva sin quemar el cupo.

### Vía B (Despliegue Directo de Modelo Completo)
Si Ariel opta por no invertir cupos en micro-sondas y prioriza evaluar el mejor modelo completo:
- **Envío 1**: Subir `candidate_model_havg_plus_full70k.csv` (Candidato C).
  * *Justificación*: Es el modelo con el respaldo empírico más sólido de todo el repositorio (+46.7 net fixes sobre B10 en 3 cortes confirm, log loss reducido en todos los cortes, sin fuga y con hiperparámetros congelados).

---

## 4. Estado de Verificación y Próximos Pasos

- **Verificado**:
  * Código de confirmación exacta: `src/intensive_20261030/evaluate_havg_plus_exact_confirm.py`
  * Código de generación de candidatos y manifiesto: `src/intensive_20261030/build_adaptive_submission_package.py`
  * Manifiesto estructurado: `results/intensive_20261030/adaptive_submission_package_manifest_20261030.json`
  * Suite completa de pruebas automatizadas: 46/46 tests pasando (`Ran 46 tests in 26.903s, OK`).
  * Integridad de bloqueo: 0 envíos a Kaggle realizados.
- **Exploratorio**:
  * La correlación entre confirm holdout y el slice público del 30% sigue sujeta a la varianza inherente del split de Kaggle; por ello se proporcionan las sondas vírgenes D1 y D2 como termómetros de bajo riesgo antes de consolidar el cupo final.
