# AG-0173-RESULT: Reproducción y Auditoría Forense del Baseline Fuerte Podium Engine (0.94230), Diagnóstico Pareado y Mapa de Clusters de Error

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent) & Ariel (Competidor / Usuario)
- **Fecha:** 2026-11-11T05:55:00-03:00
- **Responde a:** Mandato Vinculante de Auditoría de Baseline Fuerte, CX-0160 y origin/main (`ea94f67`)
- **Rama Activa:** `antigravity/overnight-recovery-coordination-20261030`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. No hay cuota restante hoy.
- **Récord Confirmado Inmune:** **`0.94766`** ($19.899 / 20.998$ aciertos, registrado en S4 = `EVT-154`).
- **Estado Vinculante de Modelos:**
  - `S4`: **`QUALIFIED_CONFIRMED_RECORD`** (`0.94766`).
  - `T09 K250`: **`REJECTED_PUBLIC`** (score oficial `0.94295`, delta agregado aprox $-99$ aciertos).
  - `Bases AG-0168 (A1, A2, B1, B2, D)`: **`REJECTED_WEAK_BASELINES`**.
  - `Candidatas C1-C4 AG-0168`: **`BLOCKED_DO_NOT_UPLOAD`**.

---

## 1. Verificación e Integridad Criptográfica del OOF de Podium Engine

Se auditó de forma determinística el archivo [`results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv) mediante [`scripts/audit_and_reproduce_podium_engine.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/audit_and_reproduce_podium_engine.py):

- **SHA-256 (LF):** `2ae8bd20764675b95a37fa90e8c20e2e2b60acfe355c46fa0cc8128a95ebc7b6`
- **Filas:** $70.000$ exactas | **Nulos:** $0$.
- **Polaridad Confirmada:** `y_true = 1` representa condición `used` (tasa: $0.4630$, $32.410$ usados, $37.590$ nuevos). Umbral canónico de corte: $p \ge 0.50 \implies \text{used}$.
- **Métricas Globales Confirmadas:**
  - **Accuracy OOF:** **`0.94230`** (exactamente **$65.961 / 70.000$ aciertos**).
  - **LogLoss OOF:** **`0.14887`**.
  - **ROC-AUC OOF:** **`0.98596`**.
  - **Intervalo de Confianza Bootstrap Jerárquico [Q05, Q95]:** `[0.94006, 0.94464]`, Mediana `0.94232`.

---

## 2. Deconstrucción de Componentes y Contribución Marginal (Ablación)

Evaluación de los componentes individuales de Podium Engine ([`results/podium_engine_forensic_reproduction/podium_components_ablation.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/podium_components_ablation.csv)):

| Componente | Accuracy OOF | Hits (70k) | LogLoss OOF | ROC-AUC | Ganancia vs Text | Delta vs Ensamble |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Text Specialist Linear (SAGA 100k TF-IDF)** | `0.90556` | $63.389$ | $0.27650$ | $0.96722$ | — | $-2.572$ |
| **2. CatBoost Depth 8 Native (Text + Categóricas)** | `0.93979` | $65.785$ | $0.15620$ | $0.98456$ | $+2.396$ | $-176$ |
| **3. LightGBM F130+ Bagged (3 Seeds, 900 Árboles)** | `0.94139` | $65.897$ | $0.15219$ | $0.98583$ | $+2.508$ | $-64$ |
| **4. Dual Blend (LightGBM + CatBoost)** | `0.94241` | $65.969$ | $0.14763$ | $0.98617$ | $+2.580$ | $+8$ |
| **5. Podium Engine Optimal Convex Stacking** | **`0.94230`** | **$65.961$** | **$0.14887$** | **$0.98596$** | **$+2.572$** | **$0$** |

- **Pesos Óptimos de Stacking:** $w_{\text{LGB}} = 0.6321$, $w_{\text{CB}} = 0.3608$, $w_{\text{Text}} = 0.0070$.
- La combinación balanceada entre LightGBM y CatBoost explica el $99.9\%$ del rendimiento, con LightGBM aportando la mayor calibración probabilística.

---

## 3. Desglose por Fold y Métricas Seen / New Local al Fold

Podium Engine utilizó `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Al clasificar a cada vendedor de validación estrictamente según si apareció o no en el split de entrenamiento de ese fold ([`results/podium_engine_forensic_reproduction/podium_fold_metrics.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/podium_fold_metrics.csv)):

| Fold | Filas Val | Accuracy Fold | Hits Fold | LogLoss Fold | N Vistos en Fold | Acc Vistos | N Nuevos en Fold | Acc Nuevos |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | $14.000$ | `0.94571` | $13.240$ | $0.14267$ | $9.770$ | `0.96100` | $4.230$ | `0.91040` |
| **2** | $14.000$ | `0.94050` | $13.167$ | $0.14929$ | $9.575$ | `0.96031` | $4.425$ | `0.89763` |
| **3** | $14.000$ | `0.94200` | $13.188$ | $0.15241$ | $9.703$ | `0.95981` | $4.297$ | `0.90179` |
| **4** | $14.000$ | `0.94164` | $13.183$ | $0.14983$ | $9.770$ | `0.96039` | $4.230$ | `0.89835` |
| **5** | $14.000$ | `0.94164` | $13.183$ | $0.15018$ | $9.669$ | `0.96359` | $4.331$ | `0.89263` |
| **Total** | **$70.000$** | **`0.94230`** | **$65.961$** | **$0.14887$** | **$48.487$ ($69.3\%$)** | **`0.96102`** | **$21.513$ ($30.7\%$)** | **`0.90011`** |

- **Vendedores Vistos en el Fold:** Exactitud de **`96.10%`** ($46.597 / 48.487$ aciertos).
- **Vendedores Nuevos en el Fold:** Exactitud de **`90.01%`** ($19.364 / 21.513$ aciertos).
- Esta medición corrige la estimación previa y demuestra que Podium Engine generaliza por encima del $90\%$ incluso en vendedores que jamás vio durante el entrenamiento de su fold.

---

## 4. Diagnóstico Forense: Por Qué las Bases de AG-0168 Quedaron en 0.799–0.828

El déficit neto de **$-7.990$ aciertos** del SuperLearner de AG-0168 respecto de Podium Engine se debe a 4 causas objetivas:

1. **Protocolo de Validación Dispar:**
   - AG-0168 evaluó bajo un fold exterior de grupos de vendedores donde el $100\%$ de los casos eran vendedores completamente nuevos.
   - Podium Engine evaluó bajo StratifiedKFold ($69.3\%$ vistos), que reproduce la distribución real del test set (~71% de los ítems de test pertenecen a vendedores presentes en train).
2. **Omisión de Variables de Dominio Decisivas en AG-0168:**
   - `initial_quantity`: En MercadoLibre, listings con cantidades iniciales elevadas (>10 unidades) son casi invariablemente productos nuevos de tiendas oficiales/distribuidores. Presente en Podium, ausente en AG-0168.
   - `price_ratio`: Relación entre el precio del ítem y la mediana de su categoría específica. Permite detectar de inmediato usados por descuento relativo. Presente en Podium, ausente en AG-0168.
   - Indicadores de Palabras Clave Estructurados: Diccionarios de regex para marcas de uso (`"antiguo"`, `"impecable"`, `"como nuevo"`, `"despiece"`) y novedad (`"0km"`, `"en caja"`, `"sellado"`). Ausentes en AG-0168.
3. **Escala y Capacidad de Modelado:**
   - Podium Engine utilizó un vocabulario textual de $100.000$ términos (vs $70.000$), sublinear TF, TruncatedSVD de $64$ componentes y pasó la probabilidad textual `p_text` a los árboles.
   - LightGBM fue entrenado con $900$ árboles y ensamble de $3$ semillas ($42, 2026, 777$). CatBoost utilizó $1.400$ iteraciones a profundidad 8 con tokenización de texto y categorías nativas (vs profundidad 7 y solo $500$ iteraciones en AG-0168).

---

## 5. Comparación Pareada de Errores por ID (Matriz 2x2)

En [`results/podium_engine_forensic_reproduction/paired_error_comparison_podium_vs_ag0168.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/paired_error_comparison_podium_vs_ag0168.csv):

| Condición Pareada | Cantidad de Ítems | Porcentaje | Interpretación Metodológica |
|---|:---:|:---:|---|
| **Ambos Correctos (Consenso)** | $57.065$ | $81.52\%$ | Núcleo de alta certeza compartido por ambos modelos. |
| **Ambos Incorrectos (Huecos Duros)** | $3.133$ | $4.48\%$ | Casos intrínsecamente difíciles (ambigüedades, etiquetas ruidosas). |
| **Podium Correcto, AG-0168 Incorrecto** | **$8.896$** | **$12.71\%$** | Ventaja masiva de Podium por sus variables de dominio y capacidad. |
| **AG-0168 Correcto, Podium Incorrecto** | **$906$** | **$1.29\%$** | **Señal complementaria pura** no capturada por Podium Engine. |
| **Ventaja Neta de Podium Engine** | **$+7.990$** | **$+11.41\%$** | Dominancia categórica del baseline de referencia. |

> [!IMPORTANT]
> Se han aislado exactamente **$906$ ítems** donde el modelo de AG-0168 acierta pero Podium Engine falla. Esto constituye la materia prima legítima para futuros ensambles o correcciones incrementales, sin destruir la base de $65.961$ aciertos de Podium.

---

## 6. Mapa de Clusters de Error de Podium Engine (4.039 Errores)

Análisis en [`results/podium_engine_forensic_reproduction/podium_error_clusters.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/podium_error_clusters.json):

- **Total de Errores OOF:** $4.039$ ($5.77\%$ de tasa de error).
  - **Falsos Positivos (Predicho `used`, Real `new`):** $2.180$ ($54.0\%$).
  - **Falsos Negativos (Predicho `new`, Real `used`):** $1.859$ ($46.0\%$).
- **Tasa de Error por Volumen del Vendedor:**
  - Vendedores frecuentes ($\ge 10$ publicaciones): **`3.02%`** de error ($96.98\%$ exactitud).
  - Vendedores medianos ($2$ a $9$ publicaciones): **`5.18%`** de error ($94.82\%$ exactitud).
  - Vendedores singleton ($1$ sola publicación): **`10.23%`** de error ($89.77\%$ exactitud).  
    *Los vendedores singleton concentran el $49.3\%$ de todos los errores de Podium Engine.*
- **Top Categorías con Mayor Concentración de Fallas:**
  1. `MLA1055` (Celulares y Smartphones): $189$ errores.
  2. `MLA1227` (Indumentaria y Calzado): $174$ errores.
  3. `MLA1648` (Computación): $142$ errores.
  4. `MLA31838` / `MLA31763` (Monedas y Billetes de Colección): $128$ errores.
  5. `MLA1963` (Revistas y Comics): $98$ errores.

### Taxonomía de los Cuatro Huecos Predictivos Prioritarios:
1. **Coleccionables y Antigüedades sin Uso:** Ítems con palabras como `"antiguo"`, `"vintage"` o `"retro"` que en realidad corresponden a stock nuevo de fábrica o piezas de colección selladas.
2. **Productos Usados Descritos con Jerga de Marketing:** Publicaciones con `"impecable estado"`, `"igual a nuevo"`, `"en caja original"` o `"con garantía"` que confunden al modelo hacia la clase `new`.
3. **Autopartes y Repuestos Nuevos:** Piezas de repuesto vendidas por tiendas oficiales con descripciones de compatibilidad o despiece que el modelo clasifica erróneamente como `used`.
4. **Singletons con Títulos Ambiguos o Breves:** Publicaciones de vendedores únicos sin historial previo y con títulos de menos de 4 palabras donde el modelo carece de contexto suficiente.

---

## 7. Resultados de Protocolos de Validación Corregidos

En [`results/podium_engine_forensic_reproduction/corrected_protocol_audit.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/corrected_protocol_audit.csv):

1. **Validación Temporal Forward Real (Corte en `2015-10-15`):**
   - Entrenamiento en pasado ($< 2015-10-15$, $69.689$ filas), evaluación estricta en futuro ($\ge 2015-10-15$, $311$ filas).
   - Exactitud del modelo textual lineal puro: **`0.79743`** ($248 / 311$ aciertos), LogLoss: **`0.41859`**.
2. **Leave-Category-Family-Out Real con Refit (Exclusión de `MLA1227`):**
   - Reentrenamiento completo excluyendo los $3.180$ ítems de `MLA1227` (entrenamiento en $66.820$ filas).
   - Evaluación fuera de dominio en `MLA1227`: Exactitud: **`0.73302`** ($2.331 / 3.180$ aciertos), evidenciando la especificidad del vocabulario de indumentaria.
3. **Bootstrap Jerárquico por Vendedor y Categoría (1.000 réplicas):**
   - Percentil 5: **`0.94006`** | Mediana: **`0.94232`** | Percentil 95: **`0.94464`**.
   - Confirma que Podium Engine mantiene $>94.0\%$ de exactitud con $95\%$ de confianza estadística.

---

## 8. Recomendación Estratégica y Hoja de Ruta

- **Cierre de Brecha Metodológica:** Quedan completamente archivadas y bloqueadas las candidatas de AG-0168.
- **Base Central Revalidada:** **Podium Engine** ($0.94230$) es la única arquitectura base sólida demostrada sobre los datos de entrenamiento.
- **Dirección Futura para Sprints Posteriores:**
  1. No intentar reemplazar Podium Engine desde cero con modelos simples que ignoran variables de dominio.
  2. Implementar extensiones incrementales que mantengan la estructura de Podium Engine (F130+ features, `initial_quantity`, `price_ratio`, texto nativo CatBoost, LightGBM bagged) y agreguen señales orientadas específicamente a los $906$ aciertos complementarios y los clusters de vendedores singleton.
  3. Toda nueva candidata deberá demostrar ventaja pareada positiva neta ($\Delta > 0$) respecto de los $65.961$ aciertos de Podium Engine antes de cualquier consideración para submissions.

Todos los datos, scripts y reportes quedan archivados de forma reproducible en [`results/podium_engine_forensic_reproduction/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/) y verificables con `python scripts/audit_and_reproduce_podium_engine.py`.