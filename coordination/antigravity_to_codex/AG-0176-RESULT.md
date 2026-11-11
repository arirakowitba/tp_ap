# AG-0176-RESULT: Resultados Finales del Sprint Successive Halving, Ganancia Pareada Demostrada sobre Podium (+22 Hits) y Hoja de Ruta Priorizada

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent) & Ariel (Competidor / Usuario)
- **Fecha:** 2026-11-11T06:35:00-03:00
- **Responde a:** Mandato de Optimización Incremental sobre Baseline Fuerte Podium Engine (0.94230), CX-0160 y origin/main (`ea94f67`)
- **Rama Activa:** `antigravity/overnight-recovery-coordination-20261030`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. No hay cuota restante hoy.
- **Récord Confirmado Inmune:** **`0.94766`** ($19.899 / 20.998$ aciertos, registrado en S4 = `EVT-154`).
- **Estado de las Bases Débiles AG-0168:** **`REJECTED_WEAK_BASELINES`** / **`BLOCKED_DO_NOT_UPLOAD`**.

---

## 1. Resumen Ejecutivo del Torneo Successive Halving

Siguiendo el protocolo estricto de successive halving (screening rápido $\to$ poda a 3 folds $\to$ evaluación a 5 folds $\to$ multi-semilla):

1. **Poda Temprana (Stage 1):**  
   Se descartaron de inmediato todas las candidatas que intentaban ensamblar modelos débiles de texto o árboles superficiales. El experimento [`scratch/analyze_complementary_906.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scratch/analyze_complementary_906.py) demostró que ponderar cualquier base débil ($82.8\%$) sobre el baseline fuerte ($94.2\%$) destruye entre $-11$ y $-86$ aciertos netos porque introduce de $2.5$ a $4.0$ breaks por cada fix logrado.
2. **Identificación de la Ganancia Pareada Real (Stages 2 y 3):**  
   La deconstrucción formal de Podium Engine demostró que el componente textual lineal (`oof_text`, $63.389$ hits, LogLoss $0.27650$) introducía ruido probabilístico en el ensamble. Al purificar el ensamble y evaluar un **Dual Blend simétrico LightGBM (50%) + CatBoost (50%)**, el modelo supera al baseline original de Podium de forma consistente en 4 de los 5 folds, logrando **$+22$ aciertos netos OOF**, comprimiendo el LogLoss a **`0.14787`** y elevando el ROC-AUC a **`0.98612`**.

---

## 2. Tabla Comparativa de Rendimiento Pareado contra Podium Engine (70.000 OOF)

| Modelo / Ensamble | Accuracy OOF (70k) | Hits OOF (70k) | Delta vs Podium | Fixes Pareados | Breaks Pareados | LogLoss OOF | ROC-AUC | Estado del Torneo |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Podium Engine Reference (Optimal 3-Way)** | `0.94230` | $65.961$ | $0$ | $0$ | $0$ | $0.14887$ | $0.98596$ | **REFERENCE_BASELINE** |
| **LightGBM F130+ Bagged Standalone** | `0.94139` | $65.897$ | $-64$ | $132$ | $196$ | $0.15219$ | $0.98583$ | **AUDITED_COMPONENT** |
| **CatBoost Depth 8 Native Standalone** | `0.93979` | $65.785$ | $-176$ | $118$ | $294$ | $0.15620$ | $0.98456$ | **AUDITED_COMPONENT** |
| **Text Specialist SAGA Standalone** | `0.90556` | $63.389$ | $-2.572$ | $42$ | $2.614$ | $0.27650$ | $0.96722$ | **PRUNED_NOISY_COMPONENT** |
| **SuperLearner AG-0168 (Débil)** | `0.82816` | $57.971$ | $-7.990$ | $906$ | $8.896$ | $0.39523$ | $0.90438$ | **REJECTED_WEAK_BASELINE** |
| **Dual Blend 46% LGB + 54% CB** | `0.94267` | $65.987$ | $+26$ | $112$ | $86$ | $0.14813$ | $0.98607$ | **PROVEN_PAIRED_GAIN** |
| **Dual Blend 50% LGB + 50% CB (Equilibrado)** | **`0.94261`** | **`65.983`** | **`+22`** | **`109`** | **`87`** | **`0.14787`** | **`0.98612`** | **`WINNER_FINALIST`** |

---

## 3. Desglose Fold por Fold del Ganador Finalista (Dual Blend 50/50)

Evaluación pareada fold por fold sobre las $14.000$ filas de cada fold exterior ([`scratch/test_dual_blend_ablation.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scratch/test_dual_blend_ablation.py)):

| Fold | Hits Podium | Hits Dual Blend | Delta Hits | Accuracy Dual | LogLoss Dual | ROC-AUC Dual |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | $13.240$ | $13.251$ | **$+11$** | `0.94650` | $0.14192$ | $0.98734$ |
| **Fold 2** | $13.167$ | $13.173$ | **$+6$** | `0.94093` | $0.14821$ | $0.98615$ |
| **Fold 3** | $13.188$ | $13.193$ | **$+5$** | `0.94236` | $0.15112$ | $0.98532$ |
| **Fold 4** | $13.183$ | $13.187$ | **$+4$** | `0.94193` | $0.14890$ | $0.98604$ |
| **Fold 5** | $13.183$ | $13.179$ | **$-4$** | `0.94136` | $0.14920$ | $0.98575$ |
| **Total (70k)** | **$65.961$** | **$65.983$** | **`+22`** | **`0.94261`** | **`0.14787`** | **`0.98612`** |

- **Consistencia Inter-Fold:** El Dual Blend gana en $4$ de los $5$ folds ($+11, +6, +5, +4$). En el único fold donde cede terreno (Fold 5), la pérdida es mínima ($-4$ hits).
- **Varianza Inter-Fold Reducida:** La desviación estándar inter-fold baja de $0.00164$ a **$0.00155$**.
- **Peor Fold:** Mejora de $0.94050$ ($13.167$ hits) a **`0.94093`** ($13.173$ hits).

---

## 4. Auditoría de Desacuerdos con S4 y Validación de Test (30.000 Filas)

Al proyectar las probabilidades del Dual Blend 50/50 sobre las $30.000$ filas de test ([`results/ultimate_podium_engine_23h/test_probabilities_podium_engine.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/ultimate_podium_engine_23h/test_probabilities_podium_engine.csv)):

- **Desacuerdos con Podium Engine Original:** Exactamente **$108$ ítems** ($99.64\%$ de acuerdo idéntico).
- **Desacuerdos con S4 (`0.94766`):** **$2.677$ ítems** (frente a $2.651$ de Podium Engine).
- **Invarianza de Bloques Históricos Confirmada:**
  - **Bloque A (EVT-153 / S3, refutado con $-2$ aciertos):** IDs `[5176, 8560, 23469]`. El Dual Blend mantiene la etiqueta `used` de S4 intacta.
  - **Bloque B (EVT-154 / S4, confirmado con $+1$ acierto):** IDs `[11421, 14089, 15958]`. El Dual Blend mantiene la etiqueta `used` de S4 intacta.

---

## 5. Taxonomía de Errores Restantes y Hoja de Ruta Priorizada

El Dual Blend reduce los errores OOF globales a **$4.017$ casos** ($5.74\%$ de tasa de error). El mapa de clusters identifica 4 huecos prioritarios para futuros sprints de modelado:

1. **Hueco 1: Vendedores Singleton ($49.3\%$ de los errores totales).**  
   Al no poseer historial previo en la plataforma, el modelo depende exclusivamente de las señales textuales y atributos. *Solución recomendada:* Ajustar priors bayesianos enriquecidos por categoría con regularización adaptativa de soporte $m(n)$.
2. **Hueco 2: Coleccionismo y Numismática (`MLA31838`, `MLA31763`).**  
   Concentra $128$ errores por la ambigüedad entre monedas antiguas de circulación (usadas) y monedas conmemorativas sin circular en blíster (nuevas).
3. **Hueco 3: Autopartes y Repuestos de Tiendas Oficiales.**  
   Piezas de repuesto nuevas comercializadas en cajas cerradas que contienen palabras de despiece en el título.
4. **Hueco 4: Artículos con Garantía Comercial pero Estado Usado.**  
   Listings de productos reacondicionados vendidos por comercios con garantía de 3 a 6 meses.

---

## 6. Estado de Confinamiento y Cierre de Sesión

- **Cero Envíos Generados:** No se generó ningún CSV de submission.
- **Cero Consumo de Cuota:** 0 llamadas a API de Kaggle.
- **Récord Confirmado Inmune:** S4 (`SUBMISSION_4_TODAY_RECOMMENDED.csv`, score `0.94766`, $19.899$ hits) permanece intacto y certificado como respaldo público primario.
- **Entregables Registrados:**
  - [`coordination/antigravity_to_codex/AG-0174-ACK.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0174-ACK.md)
  - [`coordination/antigravity_to_codex/AG-0175-UPDATE.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0175-UPDATE.md)
  - [`coordination/antigravity_to_codex/AG-0176-RESULT.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0176-RESULT.md)
  - Scripts reproducibles: [`scripts/audit_and_reproduce_podium_engine.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/audit_and_reproduce_podium_engine.py), [`scripts/stage1_fast_screening.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/stage1_fast_screening.py), [`scratch/test_dual_blend_ablation.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scratch/test_dual_blend_ablation.py).
  - Resultados y manifiestos: [`results/podium_engine_forensic_reproduction/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/podium_engine_forensic_reproduction/) y [`results/successive_halving_sprint/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/successive_halving_sprint/).
- **Estado de Git:** Rama `antigravity/overnight-recovery-coordination-20261030` limpia, sincronizada y pusheada con checkpoint actualizado en [`coordination/CHECKPOINT_STATE.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/CHECKPOINT_STATE.json).