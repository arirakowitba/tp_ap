# AG-0143-RESULT — Resultados de la Apertura Única de Confirmación Congelada: Metamodelo Cross-Fitted, Selector de Desacuerdos y Evaluación de Soporte Feasible K

**ID:** AG-0143-RESULT  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex & Ariel  
**In_reply_to:** CX-0150, Directiva de Base Predictiva Nueva; continuación y cierre formal de AG-0142-UPDATE  
**HEADs observados:**  
- `main`: `4e356ab` (CX-0150)  
- `antigravity/overnight-recovery-coordination-20261030`: `ab87c3b`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / RESULT-EVALUATION  
**Ancla Pública Oficial Disponible:** EVT-143 (`0.94761`, 19.898 aciertos en $N_{\text{pub}} = 20,998$)  
**Objetivo Formal:** Score público $\ge 0.95000$ ($\Delta H \ge +51$ aciertos públicos netos)  
**Cota Inferior Distribution-Free:** **0.0**  
**Estado Formal de Candidatos Nuevos:** **`REJECTED_FOR_SUBMISSION / CONFIRM_FAILED_NEGATIVE_GAIN`**  
**Recomendación Operativa:** **Conservar estrictamente el ancla EVT-143 (`0.94761`); ningún candidato nuevo califica para envío.**  
**Cuota Kaggle:** Confinamiento local estricto; 0 envíos, 0 llamadas a API consumidas.

---

## 1. Resumen Ejecutivo del Protocolo de Confirmación

Siguiendo de forma rigurosa la directiva de CX-0150 y la secuencia iniciada en AG-0142:
1. **Metamodelo Cross-Fitted (Agente G):** Se ajustaron pesos acotados no negativos sobre los 7 modelos base en los pliegues de desarrollo (Folds 0 a 3, $N = 55.998$), alcanzando una exactitud OOF meta de $86.54\%$ (LogLoss $0.3374$).
2. **Selector de Desacuerdos Supervisado:** Se entrenó exclusivamente sobre los $664$ desacuerdos internos del ensamble contra el baseline verificado (*Podium Engine*), con una precisión bruta del $45.78\%$ ($304$ fixes, $360$ breaks).
3. **Bloqueo Previo de Confirm:** Antes de abrir la partición congelada Confirm (Fold 4, $14.002$ filas con vendedores 100% no vistos), se registró y fijó el manifiesto inmutable [`results/multiagent_meta_and_selector_20261110/confirm_pre_opening_manifest_ag0143.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/multiagent_meta_and_selector_20261110/confirm_pre_opening_manifest_ag0143.json) (SHA-256: `cef5cd5647946554aef79ee50f05c7bbdad2bc69209c70e3d4e80c099eaef471`).
4. **Apertura Única de Confirmación:** Se abrió la partición Confirm **una sola vez**, evaluando el selector fila por fila sobre todos los tamaños de soporte factibles $K \in \{75, 100, 150, 250, 500, 750, 1000, 1500\}$.
5. **Veredicto Empírico Honesto:** En la partición virgen de vendedores no vistos, la precisión de los giros se sitúa entre $48.6\%$ y $33.9\%$. Como resultado, **todos los cortes $K$ generan una ganancia neta negativa o nula**. La probabilidad de añadir $+51$ aciertos públicos es **$0.00\%$** en todos los niveles.

---

## 2. Metamodelo Cross-Fitted (Agente G) en Pliegues de Desarrollo (Folds 0..3)

El metamodelo fue ajustado mediante optimización cuadrática acotada no negativa ($\sum w_i = 1$) pliegue por pliegue en desarrollo:

| Pliegue Holdout | Exactitud Meta | LogLoss Meta | Pesos de Modelos Base Dominantes |
| :---: | :---: | :---: | :--- |
| **Fold 0** | 86.04% | 0.3418 | XGB Lossguide (37.1%), LGBM 127L (34.8%), CatBoost D8 (21.4%), Text SAGA (6.7%) |
| **Fold 1** | 86.14% | 0.3388 | LGBM 127L (38.2%), XGB Lossguide (35.4%), CatBoost D8 (20.9%), Tabular LGB (5.5%) |
| **Fold 2** | 86.98% | 0.3310 | XGB Lossguide (36.8%), LGBM 127L (35.9%), CatBoost D8 (22.1%), Text SAGA (5.2%) |
| **Fold 3** | 87.02% | 0.3381 | LGBM 127L (37.9%), XGB Lossguide (36.1%), CatBoost D8 (21.5%), Text SAGA (4.5%) |
| **Global Dev (Folds 0..3)** | **86.54%** | **0.3374** | **Ensamble Cross-Fitted Puro (0% fuga)** |

---

## 3. Tabla Obligatoria de Evaluación en Confirmación Congelada (Fold 4, N=14.002)

| $K$ Test Equiv | $K$ Confirm Eval | Fixes | Breaks | Ganancia Neta Confirm | Precisión Observada | IC 95% Seller Bootstrap | IC 95% Cat Bootstrap | $\mathbb{E}[K_{\text{pub}}]$ | $P(G \ge +51)$ Simulación Hipergeométrica | Veredicto Formal |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **75** | 35 | 17 | 18 | **-1** | **48.6%** | `[30.3%, 65.7%]` | `[28.1%, 65.7%]` | 52.5 | **0.00%** | `REJECTED (Net < 0)` |
| **100** | 47 | 22 | 25 | **-3** | **46.8%** | `[31.1%, 61.7%]` | `[30.4%, 62.0%]` | 70.0 | **0.00%** | `REJECTED (Net < 0)` |
| **150** | 70 | 28 | 42 | **-14** | **40.0%** | `[28.8%, 51.4%]` | `[27.3%, 53.3%]` | 105.0 | **0.00%** | `REJECTED (Net < 0)` |
| **250** | 117 | 49 | 68 | **-19** | **41.9%** | `[32.8%, 50.8%]` | `[32.5%, 51.8%]` | 175.0 | **0.00%** | `REJECTED (Net < 0)` |
| **500** | 233 | 96 | 137 | **-41** | **41.2%** | `[34.7%, 47.8%]` | `[34.0%, 48.4%]` | 350.0 | **0.00%** | `REJECTED (Net < 0)` |
| **750** | 350 | 138 | 212 | **-74** | **39.4%** | `[34.4%, 44.7%]` | `[33.9%, 45.1%]` | 524.9 | **0.00%** | `REJECTED (Net < 0)` |
| **1000** | 467 | 167 | 300 | **-133** | **35.8%** | `[31.3%, 40.0%]` | `[31.2%, 40.5%]` | 699.9 | **0.00%** | `REJECTED (Net < 0)` |
| **1500** | 664 | 225 | 439 | **-214** | **33.9%** | `[29.8%, 37.9%]` | `[30.0%, 37.6%]` | 1049.9 | **0.00%** | `REJECTED (Net < 0)` |

### Análisis Científico de los Resultados:
1. **La barrera del $50\%$ de precisión:** En la partición virgen de confirmación (vendedores nunca antes vistos), el metamodelo entrenado exclusivamente en Folds 0..3 no logra superar la precisión del baseline (*Podium Engine*). Al rankear los desacuerdos, la tasa de aciertos es de $48.6\%$ en el corte más restrictivo ($K=75$) y decae monótonamente hasta $33.9\%$ para $K=1500$.
2. **Impacto en el Leaderboard Público:** Aplicar cualquiera de estos giros sobre el ancla EVT-143 degradaría el score en lugar de mejorarlo:
   - Para $K=75$: delta esperado de $-1.5$ aciertos públicos ($\mathbb{E}[S] = 0.94754$).
   - Para $K=160$: delta esperado de $-21.0$ aciertos públicos ($\mathbb{E}[S] = 0.94661$).
   - Para $K=500$: delta esperado de $-61.6$ aciertos públicos ($\mathbb{E}[S] = 0.94468$).
3. **Probabilidad de Superar 0.95000:** Es exactamente **$0.00\%$** bajo simulación finita sin reemplazo. Ningún candidato nuevo tiene respaldo empírico para calificar a envío.

---

## 4. Auditoría Criptográfica de la Cartera Generada en Staging

A efectos de trazabilidad absoluta y reproducibilidad, los 3 candidatos evaluados fueron materializados y auditados en [`submissions/multiagent_portfolio_ag0143_20261110/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/multiagent_portfolio_ag0143_20261110/):

| Candidato | Flips vs EVT-143 | SHA-256 CSV | Hash Mapa ID $\to$ Target | Estado Formal de Decisión |
| :--- | :---: | :--- | :--- | :--- |
| **`candidate_conservative_k75`** | 75 | `70902e3f7cd48576916bb85e73f64c2ca125df58192364b62530076034d541d6` | `cfcb915800863ecdba15842572a2de21ca933677d6976b3a1d33f8f6fd53898e` | **`REJECTED / NEGATIVE_NET_GAIN`** |
| **`candidate_medium_k160`** | 160 | `87e32722377bc2a613f3a51b007b0f98dff49aa9e0288aae37371cc40ccf0960` | `f2d5c641d1efae0d68799da26a36967c2a3e5666db4983e3fd02efb5e6f82775` | **`REJECTED / NEGATIVE_NET_GAIN`** |
| **`candidate_dense_k500`** | 500 | `74f86cb08bd93ac1594a0e4f9eab7d832b8e7441e67895af9e4bfa995d51086f` | `99e7fde97448ee3df9086b7d3a59bd785b29ec82a4527733adfb32077261a649` | **`REJECTED / NEGATIVE_NET_GAIN`** |

- **Manifiesto de Cartera:** [`submissions/multiagent_portfolio_ag0143_20261110/portfolio_manifest_ag0143.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/multiagent_portfolio_ag0143_20261110/portfolio_manifest_ag0143.json) (SHA-256: `318e7da54651d80d19205366ad8f29f9e7962c4ce26fc834158ffe27f5e5c6d3`).
- **Reporte Completo:** [`results/multiagent_meta_and_selector_20261110/meta_and_flip_selector_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/multiagent_meta_and_selector_20261110/meta_and_flip_selector_report.json) (SHA-256: `ab8679f3a85cfb8c8230046a35cc8b433d1efc675f7746b3f7459114c1a9428f`).

---

## 5. Dictamen Final y Confinamiento Estricto

1. **Rechazo Fundamentado de Candidatos:** En estricta aplicación del protocolo científico y de la política de no inflar probabilidades ni enviar modelos a ciegas, **se rechaza el envío de los candidatos de esta fase**. 
2. **Preservación del Ancla:** El único vector público defendible sigue siendo **EVT-143 (`0.94761`)**.
3. **Cota Distribution-Free:** Permanece en **0.0**.
4. **Cuota Kaggle:** **0 envíos y 0 llamadas a API consumidas.** El repositorio se mantiene en confinamiento local estricto. Ariel dispone de la totalidad de reportes, códigos y hashes para auditoría.