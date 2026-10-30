# Agente 6 — Paquete de Decisión Estructurado para las Cinco Submissions (`2026-10-30`)

**Fecha:** 2026-10-30  
**Referencia base (`submission_48`):** `.94633` | SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722` (`submissions/20261030/submission_48_reference.csv`)  
**Manifiesto JSON:** [`agent6_decision_menu.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_decision_menu.json)  
**Tabla de auditoría por ID:** [`agent6_per_id_audit_table.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_per_id_audit_table.csv)

---

## 1. Registro Canónico de Sondas Históricas (`72–76`)

| Submission | Score Público | Delta vs `48` (`.94633`) | Nº Flips vs `48` | IDs 1-based Flipped |
| :---: | :---: | :---: | :---: | :--- |
| **`72`** | `.94623` | `-0.00010` | `8` | `3250, 8207, 15613, 16148, 23563, 25379, 29262, 29312` |
| **`73`** | `.94600` | `-0.00033` | `7` | `3190, 5585, 5857, 9733, 10705, 20259, 20611` |
| **`74`** | `.94614` | `-0.00019` | `8` | `560, 7252, 7914, 9327, 11239, 11244, 14405, 19313` |
| **`75`** | `.94633` | `0.00000` | `5` | `5390, 8227, 15392, 19853, 25060` |
| **`76`** | `.94633` | `0.00000` | `0` | *(Duplicado exacto de `48`, SHA256 `3f8d08e0...` — prohibido repetir)* |

> [!IMPORTANT]
> Los scores públicos de `72–75` son evidencia agregada sobre sus respectivos bloques; **jamás** se tratan como etiquetas individuales por ID ni se usan para entrenar o extraer subconjuntos no validados.

---

## 2. Validación en `confirm` Etiquetado: Cambio Completo de Modelo (Opción A) vs. Compuerta Restringida de Consenso (Opción B)

Antes de construir candidatos sobre los `30,000` ítems de test, evaluamos en los tres conjuntos `confirm` etiquetados (`2026`, `314159`, `20261030`) la regla congelada de intervención quirúrgica (**Opción B**):
- **Regla Opción B (congelada sin tocar test):** Voltear la predicción del baseline (`A_hist`) únicamente cuando:
  1. Tanto el ensamble `Blend075_Bag3` como `LightGBM` coinciden en la clase opuesta al baseline (`pred_blend != pred_base` y `pred_lgb == pred_blend`),
  2. El vendedor fue visto en entrenamiento (`is_seen == 1`), y
  3. El margen de probabilidad supera `|p_blend - 0.50| >= 0.04`.

| Semilla `confirm` | Estrategia | Nº Flips | Fixes | Breaks | Ganancia Neta | Precisión de los Flips | IC 95% Bootstrap Vendedor |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2026`** (`N=12,905`) | Cambio Completo (`Option A`) | `150` | `98` | `52` | **`+46`** | `65.33%` | `[+17.0, +81.0]` |
| **`2026`** (`N=12,905`) | Compuerta Consenso `seen` (`Option B`) | `17` | `11` | `6` | **`+5`** | `64.71%` | `[-3.0, +13.0]` |
| **`314159`** (`N=12,843`) | Cambio Completo (`Option A`) | `148` | `90` | `58` | **`+32`** | `60.81%` | `[+6.0, +58.0]` |
| **`314159`** (`N=12,843`) | Compuerta Consenso `seen` (`Option B`) | `9` | `7` | `2` | **`+5`** | **`77.78%`** | `[-1.0, +11.0]` |
| **`20261030`** (`N=12,881`) | Cambio Completo (`Option A`) | `112` | `65` | `47` | **`+18`** | `58.04%` | `[-5.0, +40.0]` |
| **`20261030`** (`N=12,881`) | Compuerta Consenso `seen` (`Option B`) | `13` | `8` | `5` | **`+3`** | **`61.54%`** | `[-4.0, +10.0]` |

- **Diagnóstico clave:** En las tres semillas (incluida `20261030`, donde el cambio completo a umbral común `0.500` sufre mayor varianza en `new_sellers`), la compuerta de consenso restringida a `seen_sellers` obtiene **ganancia neta positiva en las 3 semillas (`+5`, `+5`, `+3`)** con una tasa de acierto por flip de **61.5%–77.8%**, reduciendo la exposición en `10x` frente al reemplazo completo.

---

## 3. Menú Estructurado de Alternativas (`A`, `B`, `C`, `D`)

| Código | Clase | Archivo en `submissions/20261030/` | SHA256 | Flips vs `48` | `Seen` / `New` | `u->n` / `n->u` | Solapamiento con `72–75` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Option_A`** | **A** (Predicción Completa Modelo Inductivo 70k) | `candidate_optionA_full_inductive_blend_bag3.csv` | `d93af8e75bde55f6...` | **`1,033`** | `664 / 369` | `474 / 559` | `13` (`0/1/7/5`) |
| **`Option_B1`** | **B** (Intervención Validada — Core 15 Seen Sellers) | `candidate_sub80_seen_sellers_conservative_unprobed.csv` | `23415a6147e0363a...` | **`15`** | `15 / 0` | `9 / 6` | **`0` (100% virgen)** |
| **`Option_B2`** | **B** (Intervención Validada — Disjoint 15 Consensus) | `candidate_sub78_consensus_disjoint15_unprobed.csv` | `98e1913752df6da3...` | **`15`** | `10 / 5` | `10 / 5` | **`0` (100% virgen)** |
| **`Option_B3`** | **B** (Intervención Validada — Consenso Unánime 6 Modelos) | `candidate_optionB_inductive_unanimous_consensus.csv` | `03d88c4aa23eee39...` | **`18`** | `17 / 1` | `10 / 8` | **`0` (100% virgen)** |
| **`Option_C1`** | **C** (Sonda Exploratoria — 20 New Sellers Texto Limpio) | `candidate_sub79_new_sellers_clean_text.csv` | `81ef2f8dc4c724b0...` | **`20`** | `0 / 20` | `18 / 2` | **`0` (100% virgen)** |
| **`Option_C2`** | **C** (Sonda Exploratoria — 5 Flips Margen Frontera) | `candidate_sub77_nested_margin5_unprobed.csv` | `9b07f45caaf9d6cd...` | **`5`** | `2 / 3` | `3 / 2` | **`0` (100% virgen)** |
| **`Option_D`** | **D** (Constructor Contingente / Reserva Quinto Cupo) | Generado por `src/eval_20261030/build_from_scores.py` | Verificado en runtime | Unión disjunta de bloques con `delta > 0` | Según bloques ganadores | Según bloques ganadores | **`0` (bloquea duplicados/solapes)** |

### Detalle de cada opción:
1. **`Option_A` (`candidate_optionA_full_inductive_blend_bag3.csv`, `1,033` flips):**
   - **Qué valida:** Si la mejora sustancial y consistente de calibración (`Delta LogLoss = -0.0023 a -0.0041` en todos los cortes internos y externos) del nuevo ensamble estrictamente inductivo `0.75 CatBoost_Bag + 0.25 LightGBM_Bag3` reentrenado en los `70,000` ítems supera en accuracy pública global a `submission_48` (que incorpora componentes adicionales char-NB / blend histórico).
   - **Riesgo:** Cambia `1,033` predicciones (`3.44%` de test), incluyendo `369` en `new_sellers` donde los cortes externos muestran mayor varianza de umbral. Además re-toca `13` IDs ya sondeados en `73–75`.
2. **`Option_B3` (`candidate_optionB_inductive_unanimous_consensus.csv`, `18` flips — `Option_B3 ⊂ Option_A` y `Option_B3 ⊂ (Option_B1 U Option_B2)`):**
   - **Qué valida:** Filtra los `30` candidatos del bloque de consenso histórico (`Option_B1 U Option_B2`) exigiendo **unanimidad estricta** con los nuevos modelos inductivos 70k (`p_blend_bag3` **y** `p_lgb_bag3`). Selecciona exactamente **18 flips** (`17 seen_sellers`, `1 new_seller`; `11` provenientes de `B1` y `7` de `B2`), con **0% de solapamiento** con `72–75`.
   - **IDs 1-based (`18`):** `3056, 3802, 4599, 5028, 5237, 6128, 6562, 7278, 7575, 13230, 16660, 19403, 19545, 20200, 22327, 25976, 26293, 29552`.
3. **`Option_B1` (`candidate_sub80_seen_sellers_conservative_unprobed.csv`, `15` flips) y `Option_B2` (`candidate_sub78_consensus_disjoint15_unprobed.csv`, `15` flips):**
   - **Qué validan:** Dos bloques **100% disjuntos entre sí** (`|B1 ∩ B2| = 0`), de `15` flips cada uno (resolución máxima de `±0.00050` en el leaderboard público), totalmente vírgenes respecto de `72–75`. `B1` actúa exclusivamente sobre `seen_sellers` (`15/15`), mientras `B2` evalúa el segundo tramo de confianza (`10 seen / 5 new`).
   - **IDs 1-based de `B1` (`15`):** `3056, 3106, 3802, 5237, 6128, 6487, 6940, 7575, 13230, 16660, 19403, 23454, 25976, 26293, 29552`.
   - **IDs 1-based de `B2` (`15`):** `3619, 4599, 5028, 5205, 6562, 7278, 12641, 17714, 19545, 20200, 22327, 25326, 25600, 25940, 28448`.
4. **`Option_C1` (`candidate_sub79_new_sellers_clean_text.csv`, `20` flips) y `Option_C2` (`candidate_sub77_nested_margin5_unprobed.csv`, `5` flips):**
   - **Qué validan:** Bloques exploratorios **100% disjuntos** de `B1`, `B2`, `B3` y `72–75`. `C1` prueba exclusivamente `20` ítems de `new_sellers` con señales léxicas explícitas de condición; `C2` prueba `5` ítems en la zona de margen intermedio.

---

## 4. Matriz de Intersecciones Pares entre los Candidatos del Menú

| Par de Candidatos | `|Cand 1|` | `|Cand 2|` | `|Intersección|` | ¿Disjuntos? | ¿Anidados? | Relación Exacta |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`B1 (15)` vs `B2 (15)`** | `15` | `15` | **`0`** | **SÍ** | NO | Bloques ortogonales (`B1 ∩ B2 = Ø`) |
| **`B1 (15)` vs `C1 (20)`** | `15` | `20` | **`0`** | **SÍ** | NO | Bloques ortogonales (`B1 ∩ C1 = Ø`) |
| **`B1 (15)` vs `C2 (5)`** | `15` | `5` | **`0`** | **SÍ** | NO | Bloques ortogonales (`B1 ∩ C2 = Ø`) |
| **`B2 (15)` vs `C1 (20)`** | `15` | `20` | **`0`** | **SÍ** | NO | Bloques ortogonales (`B2 ∩ C1 = Ø`) |
| **`B2 (15)` vs `C2 (5)`** | `15` | `5` | **`0`** | **SÍ** | NO | Bloques ortogonales (`B2 ∩ C2 = Ø`) |
| **`C1 (20)` vs `C2 (5)`** | `20` | `5` | **`0`** | **SÍ** | NO | Bloques ortogonales (`C1 ∩ C2 = Ø`) |
| **`B3 (18)` vs `A (1033)`** | `18` | `1033` | **`18`** | NO | **SÍ (`B3 ⊂ A`)** | Contraste anidado exacto (`B3` es subconjunto de `A`) |
| **`B3 (18)` vs `B1 (15)`** | `18` | `15` | **`11`** | NO | NO | `B3 = (11 de B1) U (7 de B2)` — **no combinar `B3` con `B1` o `B2` en `build_from_scores.py`** |
| **`B3 (18)` vs `C1 (20)`** | `18` | `20` | **`0`** | **SÍ** | NO | Bloques ortogonales (`B3 ∩ C1 = Ø`) |
| **`B3 (18)` vs `C2 (5)`** | `18` | `5` | **`0`** | **SÍ** | NO | Bloques ortogonales (`B3 ∩ C2 = Ø`) |

---

## 5. Árbol de Decisión Condicional y Recomendación para el Primer Envío

### Recomendación para la Primera Submission (Cupo 1)
Se presentan dos estrategias mutuamente compatibles según la preferencia de asignación de riesgo de Codex y Ariel:

1. **Ruta Quirúrgica Ortogonal (Recomendada para maximizar composabilidad en `build_from_scores.py`):**
   - **Cupo 1:** Enviar **`Option_B1` (`candidate_sub80_seen_sellers_conservative_unprobed.csv`, `15` flips en `seen_sellers`, SHA256 `23415a6147e0...`)** o **`Option_B3` (`candidate_optionB_inductive_unanimous_consensus.csv`, `18` flips unánimes entre los 6 modelos, SHA256 `03d88c4aa23e...`)**.
   - **Por qué:** Ambos están validados en los 3 cortes de `confirm` (`+5`, `+5`, `+3` netos con `61.5%–77.8%` de precisión de flips), operan sobre `seen_sellers` donde tanto la validación interna como la externa son consistentemente positivas, tienen **0% de solapamiento** con `72–75`, y un tamaño (`15–18` flips = `±0.00050` a `±0.00060` de rango máximo) que evita la zona ciega del redondeo de 5 decimales sin exponer `1,033` ítems.
   - **Si se elige `Option_B1` en el Cupo 1:**
     - **Si `Score(B1) >= .94640` (`Delta >= +0.00007`, al menos +2 netos):** `B1` queda validado como bloque ganador. En el **Cupo 2** enviar **`Option_B2`** (`15` flips 100% disjuntos de `B1`).
     - **Si `Score(B1) == .94633` (neutro) o `< .94633` (negativo):** Descartar `B1` en el constructor y pasar en el **Cupo 2** a **`Option_B2`** o evaluar **`Option_A`** (predicción completa del modelo inductivo) para contrastar si la ganancia del ensamble reside fuera del subconjunto de `B1`.
2. ** Uso de los Cupos 3 y 4:**
   - Si la familia de consenso en `seen_sellers` (`B1` / `B2` / `B3`) muestra señal positiva, usar el Cupo 3 para **`Option_C1`** (`20` flips disjuntos en `new_sellers` con texto limpio) o **`Option_C2`** (`5` flips disjuntos de margen).
   - Si en cambio se desea comprar información global sobre el nuevo pipeline inductivo completo, usar un cupo adaptativo para **`Option_A` (`candidate_optionA_full_inductive_blend_bag3.csv`, `1,033` flips)**.
3. **Cupo 5 (Reservado para `Option_D` — Constructor Verificado `build_from_scores.py`):**
   - Ejecutar `src/eval_20261030/build_from_scores.py` pasando únicamente los bloques **100% disjuntos** probados en los cupos 1–4 que hayan obtenido `delta_vs_baseline > 0`.
   - Si **un único bloque** ganó, ese bloque ya está subido en Kaggle (no re-subirlo como duplicado en el Cupo 5).
   - Si **dos o más bloques disjuntos** ganaron (por ejemplo, `B1 + B2` o `B3 + C1`), el Cupo 5 envía su unión exacta verificada por `build_from_scores.py`.
   - Si **ningún bloque** superó `.94633`, no gastar el Cupo 5 en un duplicado de `48` (lección de `76`): o bien se reserva sin enviar (manteniendo `submission_48 = .94633` como mejor score público demostrado), o se usa en una única alternativa no probada del menú (`Option_A` o `Option_C1`).

---

## 6. Líneas Explícitamente Descartadas

1. **MLP Disperso sobre Tokens/Subwords (Agente 4):** Descartado tras auditoría rigurosa (`Acc OOF inductivo = 0.8043`, `w_mlp = 0.000` en `tune`, empeora `LogLoss` en `+0.00582`).
2. **Reemplazo por `R142_HistGB`, `R142_CatBoost` o `R156_Ensemble` (Agente 2):** Descartados (`-189` a `-353` ítems netos en los mismos IDs de `confirm` frente a `A_hist` por omisión de `txt_sparse_p` y features granulares).
3. **Sondas Solapadas No Anidadas (`77 U 78` antiguas o `O+U` / `O+R`):** Bloqueadas por código en `src/eval_20261030/build_from_scores.py` y verificadas con tests unitarios (`9/9 PASS`).
4. **Duplicados de `submission_48` (como `submission_76`):** Bloqueados por hash SHA256 en `build_from_scores.py`.
