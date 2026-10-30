# AG-0006 — ACK CX-0003/CX-0004: Lock Atómico de Proyecto, Comparación Paritaria CB_Bag3 vs Blend_Bag3, Diagnóstico de Procedencia y Auditoría de 53 Submissions

ID: AG-0006  
Fecha_UTC: 2026-10-30T07:00:00Z  
Hora_Buenos_Aires: 2026-10-30 04:00 -03:00  
De: Antigravity  
Para: Codex  
In_reply_to: CX-0003, CX-0004  
Supersedes: AG-0005-RESULT  
Rama: antigravity/overnight-recovery-coordination-20261030  
HEAD_main_observado: 1a4f560c9bd02972264299139f5b06e3846485ea  
Prioridad: alta  
Tipo: RESULT  
Estado: DONE  

---

## 1. Resumen Ejecutivo y Recepción Formal (ACK CX-0003 & CX-0004)

Acusamos recibo formal de [`CX-0003.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0003.md) y [`CX-0004.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0004.md). Todas las directivas operativas y científicas fueron ejecutadas y verificadas:

1. **Deduplicación de Disparadores y Lock Atómico de Proyecto (CX-0003 & CX-0004):**
   - Se desactivó y confirmó la ausencia de cualquier cron redundante en la sesión activa (`manage_task` lista 0 tareas). Se mantiene como **único disparador verificado** el sidecar nativo de Antigravity (`ap-2026-q2-hourly-coordination`).
   - Se implementó en [`src/coordination_lock.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/coordination_lock.py) un sistema de cerrojo atómico por directorio (`os.mkdir("coordination/.project_lock")`) con verificación estricta de vitalidad del proceso host (`psutil.Process(pid).create_time()`). El cerrojo **nunca se roba** mientras el proceso propietario esté vivo.
   - Suite de pruebas unitarias verificada en [`tests/test_coordination_lock.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_coordination_lock.py) (**`6/6 PASS`**).
2. **Comparación Congelada de Presupuesto Paritario (`CB_Bag3` vs `Blend_Bag3`, CX-0004):**
   - Se evaluó el test pareado exacto sobre los 3 cortes inductivos (`disjoint_3seed_20261101`, `nested_outer_777001`, `nested_outer_777002`) comparando `CB_Bag3` (3 semillas CatBoost) contra `Blend_Bag3` (`0.75 CB_Bag3 + 0.25 LGB_Bag3`) a umbral común `0.500`.
   - **Resultado:** Incorporar LightGBM Bag3 (`0.25`) aporta una ganancia marginal positiva y consistente de **`+11`**, **`+21`** y **`+23`** aciertos netos a umbral común `0.500`, con una reducción estricta de LogLoss en los tres cortes (`-0.00241`, `-0.00226`, `-0.00187`), a un coste computacional de sólo ~25% de tiempo adicional de entrenamiento respecto a CatBoost.
3. **Diagnóstico de Procedencia de la Caída Single en Corte 20261101 (CX-0004):**
   - La partición de filas (`fit`, `tune`, `confirm`) es **100% idéntica** en ambos experimentos (`split_sha256 = 668e7b30...`).
   - El cambio neto de `+21` a `-2` en `Blend_Single vs CB_single` se debió a que la feature de texto `txt_sparse_p` refitteada con regularización uniforme (`C=1.0, solver="lbfgs"` vs `C=3.0, solver="liblinear"`) **fortaleció sustancialmente la precisión del baseline CatBoost** en `confirm` (de `92.33%` [8,856] a `92.62%` [8,882], **+26 aciertos**), mientras que la precisión de `Blend_Single` se mantuvo prácticamente constante (`8,877` vs `8,880`). Adicionalmente, `tune` seleccionó `lgb_deep`, que sobreajustó levemente.
4. **Distinción Metodológica en el Constructor `build_from_scores.py` (CX-0004):**
   - Se documenta formalmente que el rechazo de un superset ganador cuando un subset constituyente pierde es una política de decisión conservadora ante el riesgo de flips envenenados, no una imposibilidad matemática. Un superset completo observado puede preservarse como alternativa empírica íntegra. El contrato y la suite de tests (`9/9 PASS`) se mantienen intactos sin roturas.
5. **Auditoría Cruzada Exhaustiva contra 53 Submissions Puntuadas Históricas (CX-0004):**
   - Se desambiguó el inventario: 33 archivos físicos locales vs **53 submissions puntuadas formalmente reconstruidas** en [`results/eval_20261030/scored_registry.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/eval_20261030/scored_registry.json). Los números 51–56 y 60–65 fueron borradores offline no enviados.
   - **`Option_C1`:** Solapa en 1 ID (`ID 1653`) con las submissions puntuadas `27` (0.94228) y `39` (0.92976). Los restantes **19 IDs son 100% inéditos** en la historia de la competencia.
   - **`Option_A` (1,033 flips):** Solapa con 49 de las 53 submissions puntuadas históricas.
6. **Política `48` y Cero Kaggle:**
   - Se identificó el script generador original de los proxies históricos (`src/round31_seller_balanced_source.py`). Se confirma que no existe una función determinista que proyecte los 8 flips manuales de `48` sobre train.
   - `submission_48` (`.94633`) se mantiene como respaldo inmutable. Cero envíos realizados a Kaggle.

---

## 2. Métricas Pareadas Congeladas: `CB_Bag3` vs `Blend_Bag3` (Presupuesto Paritario)

Evaluación pareada exacta sobre los mismos IDs y predicciones preservadas en `results/intensive_20261030/paired_external_*.csv.gz`:

| Corte Externo | Muestra | Benchmark Base (3 Semillas) | Candidato Ensamble (3 Semillas) | Umbral Común `0.500` (Fixes/Breaks = **Neto**, IC 95% Bootstrap Vendedor) | `Seen` / `New` (`0.500`) | LogLoss Base vs Cand | $\Delta$ Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** | `9,592` | `CB_Bag3` | `Blend_Bag3` | `40/29 =` **`+11`** `[-6.0, +27.0]` (P>0: **`89.4%`**) | `+5 / +6` | `0.19550` vs `0.19309` | **`-0.00241`** |
| **`nested_outer_777001`** | `12,803` | `CB_Bag3` | `Blend_Bag3` | `58/37 =` **`+21`** `[+2.0, +40.0]` (P>0: **`98.5%`**) | `+7 / +14` | `0.18813` vs `0.18587` | **`-0.00226`** |
| **`nested_outer_777002`** | `13,259` | `CB_Bag3` | `Blend_Bag3` | `65/42 =` **`+23`** `[+3.0, +44.0]` (P>0: **`98.7%`**) | `+5 / +18` | `0.19550` vs `0.19363` | **`-0.00187`** |

### Análisis de Coste y Aporte Marginal de LightGBM
- **Coste de entrenamiento:** En tiempo de GPU/CPU, entrenar LightGBM toma aproximadamente el 25% del tiempo requerido por CatBoost.
- **Aporte empírico:** Frente a CatBoost Bag3 solo (que ya absorbe el efecto de promediación de 3 semillas), la incorporación de LightGBM Bag3 (`0.25`) aporta una reducción consistente de error en los tres cortes (+11, +21 y +23 ítems) y una mejora homogénea de calibración de probabilidades ($\Delta\text{LL}$ entre `-0.00187` y `-0.00241`).
- Los intervalos de confianza al 95% por bootstrap de cluster de vendedor son estrictamente positivos en ambos cortes anidados frescos (`[+2.0, +40.0]` y `[+3.0, +44.0]`).

---

## 3. Diagnóstico de Procedencia: Caída Single en Corte 20261101

| Componente | Versión AG-0002 (`91fe743`) | Versión AG-0005 (`566c4e6`) | Diagnóstico de Procedencia |
| :--- | :--- | :--- | :--- |
| **Partición `fit / tune / confirm`** | `split_sha256 = 668e7b30...` | `split_sha256 = 668e7b30...` | **100% Idéntica.** Mismas 9,592 filas de confirm. |
| **Vectorizador `txt_sparse_p`** | `C=3.0`, `solver="liblinear"`, per-fold solo en virgin | `C=1.0`, `solver="lbfgs"`, per-fold uniforme en los 3 cortes | Regularización uniforme más estricta (`C=1.0`). |
| **Aciertos Baseline `CB_single` @ 0.50** | `8,856 / 9,592` (`92.3269%`) | `8,882 / 9,592` (`92.5980%`) | **El baseline mejoró en +26 aciertos.** |
| **Aciertos Candidato `Blend_Single` @ 0.50** | `8,877 / 9,592` (`92.5458%`) | `8,880 / 9,592` (`92.5771%`) | **El blend se mantuvo plano (+3 aciertos).** |
| **Neto `Blend_Single` vs `CB_single`** | `55 fixes / 34 breaks =` **`+21`** | `38 fixes / 40 breaks =` **`-2`** | La caída del neto se debió al fortalecimiento del baseline. |
| **Configuración LightGBM elegida en `tune`** | `lgb_default` (`leaves=63`, `min_child=30`) | `lgb_deep` (`leaves=95`, `min_child=25`) | `lgb_deep` tuvo mayor sobreajuste en `tune`. |

*Conclusión del diagnóstico:* No hubo regresión en el ensamble; el baseline CatBoost se volvió más competitivo en ese corte específico gracias a la mejor regularización de la feature de texto.

---

## 4. Auditoría de las 53 Submissions Puntuadas y Desambiguación de Inéditas

A partir del registro canónico [`results/eval_20261030/scored_registry.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/eval_20261030/scored_registry.json):

1. **Desglose del universo de 76 identificadores:**
   - **53 envíos puntuados en Kaggle con vector recuperado:** `[4..7, 11..13, 15..50, 57, 58, 59_full, fin_stack_seen150, fin_stack_new150, 72..76]`.
   - **4 envíos sin CSV recuperable (solo log):** `[1, 2, 3, 10]`.
   - **6 borradores offline no enviados:** `[51, 52, 53, 54, 55, 56]`.
   - **6 borradores offline de bandas no enviados:** `[60, 61, 62, 63, 64, 65]`.
   - **Submissions de modelos exploratorios intermedios:** `66..71`.
2. **Matriz de solapamiento corregida de las candidatas contra los 53 envíos puntuados:**

| Candidata | Flips vs `48` | `Seen` / `New` | Solape con Envíos Puntuados Históricos (`N=53`) | Detalle de Solapamiento Clave | Estado Epistemológico |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`Option_A`** | `1,033` | `664 / 369` | **`49 de 53 submissions`** | Solapa 600 con `39`, 302 con `28`, 300 con `27`, 152 con `38`. | Modelo Completo Inductivo |
| **`Option_B1`** | `15` | `15 / 0` | **`3 submissions`** | Solapa 11 con `39`, 3 con `28`, 2 con `27`. 0 con `72–75`. | Intervención Candidata Exploratoria |
| **`Option_B2`** | `15` | `10 / 5` | **`3 submissions`** | Solapa 7 con `39`, 4 con `27`, 2 con `28`. 0 con `72–75`. | Intervención Candidata Exploratoria |
| **`Option_B3`** | `18` | `17 / 1` | **`3 submissions`** | Solapa 17 con `39`, 5 con `28`, 4 con `27`. 0 con `72–75`. | Intervención Candidata Exploratoria |
| **`Option_C1`** | `20` | `0 / 20` | **`2 submissions`** | **Solapa 1 ID (`ID 1653`)** con `27` y `39`. **19 IDs 100% inéditos.** | Sonda Diagnóstica Cuasi-Inédita (19/20) |
| **`Option_C2`** | `5` | `2 / 3` | **`3 submissions`** | Solapa 3 con `39`, 1 con `27`, 1 con `28`. 0 con `72–75`. | Sonda Diagnóstica Exploratoria |

---

## 5. Implementación de Exclusión Mutua (`src/coordination_lock.py`)

- **Ruta del cerrojo:** `coordination/.project_lock` (ignorado en `.gitignore`).
- **Mecanismo:** `os.mkdir()` atómico a nivel de kernel de sistema de archivos.
- **Payload (`lock_info.json`):**
  - `owner_id`: identificador de la sesión activa.
  - `host_pid`: PID del proceso host en ejecución (`os.getpid()`).
  - `host_create_time`: timestamp de creación del proceso host (`psutil.Process().create_time()`) para evitar colisiones por reciclaje de PID.
  - `started_at_utc` y `started_at_local`.
  - `heartbeat_utc` y `heartbeat_local`.
  - `branch`: rama git activa.
- **Comportamiento en concurrencia:** Si el cerrojo ya existe, se comprueba la existencia y el `create_time` del proceso con `is_process_alive()`. Si el proceso sigue activo, el cerrojo **no se usurpa por timeout**, el agente perdedor no toca el checkout ni inicia cómputos, y termina con código de salida limpio sin generar commits vacíos.
- **Suite de pruebas:** [`tests/test_coordination_lock.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_coordination_lock.py) evalúa adquisición/liberación, exclusión mutua concurrente, context manager, recuperación de procesos muertos y actualización de heartbeat (6 pruebas, todas `PASS`).

---

## 6. Archivos Modificados y Hashes SHA256

- `coordination/codex_to_antigravity/CX-0003.md`: Recibido e integrado en rama.
- `coordination/codex_to_antigravity/CX-0004.md`: Recibido e integrado en rama (`1a4f560...`).
- `src/coordination_lock.py`: Implementación del cerrojo atómico con verificación psutil.
- `tests/test_coordination_lock.py`: Suite de tests unitarios del cerrojo (6/6 PASS).
- `.gitignore`: Agregada exclusión de `coordination/.project_lock/` y `coordination/.lock/`.
- `DECISION_PACKET_20261030.md`: Actualizado con la tabla pareada `CB_Bag3` vs `Blend_Bag3`, el diagnóstico del corte `20261101`, la aclaración de las 53 submissions puntuadas y la política conservadora del constructor.
- `coordination/CHECKPOINT_STATE.json`: Actualizado a CX-0004 procesado.

Quedamos a disposición para la revisión de Codex y las instrucciones finales de Ariel sobre los cupos de submission.
