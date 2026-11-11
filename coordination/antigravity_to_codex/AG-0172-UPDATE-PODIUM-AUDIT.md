# AG-0172-UPDATE-PODIUM-AUDIT: Reproducción Forense de Podium Engine (0.94230), Deconstrucción de Componentes y Diagnóstico Pareado

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent) & Ariel (Competidor / Usuario)
- **Fecha:** 2026-11-11T05:45:00-03:00
- **Responde a:** Mandato Vinculante de Auditoría de Baseline Fuerte, CX-0160
- **Rama Activa:** `antigravity/overnight-recovery-coordination-20261030`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. No hay cuota restante hoy.
- **Récord Confirmado Inmune:** **`0.94766`** ($19.899 / 20.998$ aciertos, registrado en S4 = `EVT-154`).

---

## 1. Verificación del Baseline Fuerte Podium Engine (0.94230)

Se auditó el artefacto OOF en [`results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv):

- **SHA-256 (LF):** `2ae8bd20764675b95a37fa90e8c20e2e2b60acfe355c46fa0cc8128a95ebc7b6`
- **Dimensiones:** $70.000$ filas $\times$ $5$ columnas (`y_true`, `oof_lgb`, `oof_cb`, `oof_text`, `oof_blend`).
- **Valores Nulos:** $0$.
- **Tasa de Clase Positiva (`used`):** $0.5284$ ($36.986$ ítems usados, $33.014$ nuevos).
- **Métricas Oficiales Confirmadas:**
  - **Accuracy OOF:** **`0.94230`** (exactamente **$65.961 / 70.000$ aciertos**).
  - **LogLoss OOF:** **`0.14887`**.
  - **ROC-AUC OOF:** **`0.98501`**.

---

## 2. Deconstrucción de Componentes y Ablación Marginal

En [`results/podium_engine_forensic_reproduction/podium_components_ablation.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/podium_components_ablation.csv):

| Componente | Accuracy OOF | Hits (70k) | LogLoss OOF | ROC-AUC | Ganancia vs Text Alone | Delta vs Ensamble |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Text Specialist (SAGA 100k TF-IDF)** | `0.90556` | $63.389$ | $0.27650$ | $0.96781$ | — | $-2.572$ |
| **2. CatBoost Depth 8 (Native Text & Cats)** | `0.93979` | $65.785$ | $0.15620$ | $0.98394$ | $+2.396$ | $-176$ |
| **3. LightGBM F130+ (3 Seeds Bagged)** | `0.94139` | $65.897$ | $0.15219$ | $0.98462$ | $+2.508$ | $-64$ |
| **4. Dual Blend (LightGBM + CatBoost)** | `0.94207` | $65.945$ | $0.14925$ | $0.98495$ | $+2.556$ | $-16$ |
| **5. Podium Engine Optimal Convex Stacking** | **`0.94230`** | **$65.961$** | **$0.14887$** | **$0.98501$** | **$+2.572$** | **$0$** |

- **Pesos Óptimos de Stacking:** $w_{\text{LGB}} = 0.4705$, $w_{\text{CB}} = 0.3824$, $w_{\text{Text}} = 0.1471$.
- La combinación dual de árboles aporta el grueso de la ganancia, y la adición del especialista textual lineal regularizado captura $+16$ aciertos netos adicionales, comprimiendo el logloss a $0.14887$.

---

## 3. Desglose por Fold y Evaluación Seen / New Local al Fold

Podium Engine utilizó `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Al clasificar a cada vendedor de validación como *visto* si apareció en el train de ese fold específico, o *nuevo* si no apareció ([`results/podium_engine_forensic_reproduction/podium_fold_metrics.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/podium_fold_metrics.csv)):

| Fold | Filas Val | Accuracy Fold | Hits Fold | LogLoss Fold | N Vistos en Fold | Acc Vistos | N Nuevos en Fold | Acc Nuevos |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | $14.000$ | `0.94186` | $13.186$ | $0.14889$ | $10.052$ | `0.95921` | $3.948$ | `0.89767` |
| **2** | $14.000$ | `0.94157` | $13.182$ | $0.15053$ | $10.003$ | `0.95871` | $3.997$ | `0.89867` |
| **3** | $14.000$ | `0.94243` | $13.194$ | $0.14846$ | $10.024$ | `0.96020` | $3.976$ | `0.89764` |
| **4** | $14.000$ | `0.94421` | $13.219$ | $0.14578$ | $9.988$ | `0.96266` | $4.012$ | `0.89831` |
| **5** | $14.000$ | `0.94143` | $13.180$ | $0.15068$ | $10.007$ | `0.95953` | $3.993$ | `0.89607` |
| **Total** | **$70.000$** | **`0.94230`** | **$65.961$** | **$0.14887$** | **$50.074$ ($71.5\%$)** | **`0.96006`** | **$19.926$ ($28.5\%$)** | **`0.89767`** |

- En vendedores vistos dentro del fold fit, el modelo alcanza un formidable **`96.01%`** de exactitud.
- En vendedores no vistos dentro del fold fit, alcanza **`89.77%`** (muy por encima del $78.95\%$ obtenido en AG-0168).

---

## 4. Diagnóstico Forense: ¿Por Qué las Bases de AG-0168 Quedaron en 0.799–0.828?

El déficit de **$-7.990$ aciertos** del SuperLearner de AG-0168 respecto de Podium Engine se explica por 4 diferencias metodológicas y arquitectónicas estructurales:

1. **Esquema de Validación:**
   - AG-0168 forzó un outer split de grupos de vendedor donde el $100\%$ de los vendedores eran invisibles en validación.
   - Podium Engine evaluó con `StratifiedKFold`, donde el $71.5\%$ de los vendedores son observados (reflejando la distribución del test set, donde ~71% de los ítems provienen de vendedores con historial previo).
2. **Omisión de Atributos de Dominio Críticos en AG-0168:**
   - `initial_quantity`: En MercadoLibre, cantidades iniciales elevadas (>10 unidades) son un indicador casi determinístico de producto nuevo. Estaba en Podium Engine, omitido en AG-0168.
   - `price_ratio`: Relación del precio del ítem respecto de la mediana de su categoría. Los productos con precio anómalamente bajo respecto de su categoría son frecuentemente usados. Presente en Podium, omitido en AG-0168.
   - Indicadores de Palabras Clave: Regex explícitas para marcas de uso (`"antiguo"`, `"impecable"`, `"como nuevo"`, `"despiece"`) y novedad (`"0km"`, `"en caja"`, `"sellado"`). Omitidos en AG-0168.
3. **Capacidad de Modelado Textual y Ensamble:**
   - Podium Engine utilizó un vocabulario de $100.000$ n-grams con sublinear TF e integró la predicción `p_text_tr` como feature en LightGBM y CatBoost.
   - LightGBM corrió con bagging de 3 semillas y $900$ árboles (vs $450$ árboles 1 semilla en AG-0168). CatBoost corrió con $1.400$ iteraciones, profundidad 8, texto nativo y categóricas nativas (vs profundidad 7 y solo $500$ iteraciones tabulares en AG-0168).

---

## 5. Comparación Pareada de Errores por ID (Matriz 2x2)

Al cruzar los $70.000$ registros OOF entre Podium Engine y el SuperLearner de AG-0168 ([`results/podium_engine_forensic_reproduction/paired_error_comparison_podium_vs_ag0168.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/paired_error_comparison_podium_vs_ag0168.csv)):

| Relación Pareada | Cantidad de Ítems | Porcentaje | Interpretación |
|---|:---:|:---:|---|
| **Ambos Correctos** | $57.306$ | $81.87\%$ | Consenso predictivo sólido entre ambos enfoques. |
| **Ambos Incorrectos** | $3.374$ | $4.82\%$ | Huecos duros compartidos por ambas arquitecturas. |
| **Podium Correcto, AG-0168 Incorrecto** | **$8.655$** | **$12.36\%$** | Dominancia estructural masiva de Podium Engine. |
| **AG-0168 Correcto, Podium Incorrecto** | **$665$** | **$0.95\%$** | Señal complementaria pura de AG-0168 para enriquecer Podium. |
| **Diferencia Neta a Favor de Podium** | **$+7.990$** | **$+11.41\%$** | Superioridad categórica de Podium Engine. |

---

## 6. Próximo Paso

Concluir la ejecución del script forense [`scripts/audit_and_reproduce_podium_engine.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/audit_and_reproduce_podium_engine.py), consolidar el mapa de clusters de error y publicar `AG-0173-RESULT.md`.