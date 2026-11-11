# AG-0174-ACK: Compromiso de Sprint Multi-Agente contra el Baseline Podium Engine (0.94230) bajo Successive Halving

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent) & Ariel (Competidor / Usuario)
- **Fecha:** 2026-11-11T06:05:00-03:00
- **Responde a:** Mandato de Optimización Incremental sobre Baseline Fuerte Podium Engine (0.94230)
- **Rama Activa:** `antigravity/overnight-recovery-coordination-20261030`
- **HEADs Observados:** `origin/main` (`ea94f67`), `local_head` (`24e8cb8`).
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. No hay cuota restante hoy.
- **Récord Confirmado Inmune:** **`0.94766`** ($19.899 / 20.998$ aciertos, registrado en S4 = `EVT-154`).
- **Línea de Base Referencial Vinculante:** **Podium Engine OOF** ([`results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv), SHA-256 LF `2ae8bd20...`, **`0.94230` accuracy**, **$65.961 / 70.000$ aciertos**, **$0.14887$ logloss**).

---

## 1. Asignación de Agentes y Roles

| Agente | Especialidad | Exploraciones Clave | Criterio de Pase |
|---|---|---|---|
| **Agente A** | Texto de Alta Resolución | Normalización específica de negaciones, "sin uso", "0 km", "reacondicionado", "para reparar", "exhibición", TF-IDF estricto por fold, LinearSVC calibrado, NB-SVM y HashingVectorizer. | Reducción de logloss y ganancia pareada sobre `oof_text` de Podium ($63.389$ hits). |
| **Agente B** | Atributos y Categorías | Atributos originales completos (longitud, presencia de campos clave), `price_ratio` local, `initial_quantity`, categoría-específico target encoding cross-fitted y reglas de fallback. | Ganancia pareada neta ($\Delta > 0$) sobre LightGBM F130+ ($65.897$ hits). |
| **Agente C** | Seller Jerárquico | LOO seller y seller×category en train, shrinkage bayesiano según soporte $m$, modelado específico de singletons ($49.3\%$ de errores de Podium). | Reducción de tasa de error en singletons sin degradar vendedores vistos. |
| **Agente D** | Modelos No Lineales | CatBoost con texto nativo, LightGBM con F130+ enriquecido y XGBoost hist/depth 7. Early stopping interno y multi-seed. | $\Delta > 0$ hits vs Podium Dual Blend ($65.945$ hits). |
| **Agente E** | Modelos Complementarios | Explotación de los **$906$ ítems** donde AG-0168 superó a Podium, k-NN textuales y enrutamiento por régimen seen/new. | Captura de errores de Podium sin romper aciertos existentes. |
| **Agente F** | Auditor en Tiempo Real | Verificación de fugas, integridad OOF/test, paridad de joins, ausencia de inferencias hardcodeadas y cálculo riguroso de intervalos bootstrap. | Veto inmediato ante fuga o inestabilidad. |

---

## 2. Protocolo de Successive Halving y Presupuesto

- **Etapa 1 (Screening Rápido):** Evaluación rápida en Fold 1 (14.000 filas). Descarte inmediato de variantes con $\Delta \le 0$ respecto del fold base ($13.240$ hits).
- **Etapa 2 (Poda a 3 Folds):** Descarte de modelos que exhiban colapso en el peor fold o varianza alta.
- **Etapa 3 (Finalistas a 5 Folds):** Evaluación OOF completa de los 3-5 mejores modelos sobre las 70.000 filas.
- **Etapa 4 (Multi-Semilla y Estabilidad):** 3 semillas (`42, 2026, 777`) para verificar $\sigma_{\text{seed}} < 0.001$.
- **Etapa 5 (Ensamble y Congelamiento):** Stacking anidado óptimo y generación de probabilidades OOF y Test finales.

No se generará ningún CSV de submission para Kaggle durante esta fase. Se publicarán reportes reproducibles con hashes y código verificable.