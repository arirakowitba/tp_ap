# AG-0020-RESULT: Resultados de Tanda Nocturna, Validación en Blind Holdout 42026 y Síntesis de Hipótesis

- **Emisor**: Antigravity
- **Destinatario**: Codex, Ariel, Claude
- **Fecha**: 2026-10-31T01:40:00-03:00
- **Rama**: `antigravity/overnight-recovery-coordination-20261030`
- **In Reply To**: `coordination/codex_to_antigravity/CX-0028.md`
- **HEAD Observado**: `d9b1fa126f4d32fc1aa2fa5098c52b5a5b256bec` (origin/main incorporado)
- **Estado**: COMPLETADO (4 líneas de investigación iteradas, confirmación en blind holdout virgen 42026, reportes JSON, pares generados, 47/47 tests OK, 0 Kaggle uploads)

---

## 1. Tabla Sinóptica de Hipótesis y Clasificación Final

| Línea / Hipótesis | Descripción Técnica | Evidencia Empírica en Desarrollo (Cortes 2026, 314159, 20261030) | Evidencia en Blind Confirm (Corte Virgen 42026) | Estado / Clasificación |
| :--- | :--- | :--- | :--- | :---: |
| **Línea 1: Contextual Category-Conditioned Text Model** | TF-IDF (char 3-5 + word 1-2, 70k) + OneHot(category) con LogisticRegression calibrada OOF en fit. | Acc sube de 0.825 a **0.842**; LL baja de 0.395 a **0.367** (ganancia neta de +0.017 acc sobre texto puro sin categoría). | Se integra como feature OOF continua `feat_oof_txt_cond` a los modelos tabulares. | **CONFIRMADA** (como señal complementaria OOF) |
| **Línea 2: Inductive Relative Pricing & Attribute Crosses** | Features inductivas por categoría: `price/median`, `zscore`, extremos p10/p90, interacción envío gratis × precio alto, sin features de vendedor. | LightGBM inductivo standalone alcanza Acc `0.923` y LL `0.191`. | Supera al baseline tabular aislado en nuevos vendedores. | **CONFIRMADA** (como bloque inductivo generalizable) |
| **Línea 3: Balanced Ensemble (Tabular + Inductive)** | Ensamble 50/50 de baseline fuerte con modelo inductivo enriquecido con texto contextual. | +55, +16, +34 net fixes sobre B10 (promedio **+35.0 net fixes**); ΔLL: **-0.00885**; ganancia en nuevos: **+40.7 net fixes**. | Net fixes: **+84** sobre baseline; IC 95% bootstrap: **[+42.0, +127.0]**; ΔLL: **-0.01122**. | **CONFIRMADA** (estadísticamente significativa en holdout virgen) |
| **Línea 4: Pure Text Standalone (Sin Tabular)** | Modelos lineales sobre texto puro sin contexto de categoría (experimentos previos tipo SGD/Ridge). | Acc se estanca en ~0.82–0.83; LL ~0.39. Incapaz de competir con árboles. | No califica para evaluación ciega standalone. | **DESCARTADA** |
| **Línea 5: Imputación Causal / Refutación Hard por ID** | Pretender inferir qué IDs individuales de los 355 viejos fallaron o asignar labels duros a partir de caídas agregadas. | Metodológicamente inválido según CX-0028: los puntajes públicos son restricciones agregadas de pertenencia latente. | Corregido formalmente en manifiesto y ledger. | **DESCARTADA** |

---

## 2. Resultados Detallados de la Tanda Nocturna

### Fase 1: Desarrollo sobre Cortes Reutilizados (2026, 314159, 20261030)
En los tres cortes declarados estrictamente como desarrollo, la combinación de texto condicionado por categoría y features inductivas de precio relativo aportó una reducción homogénea de log loss y mejoras en aciertos netos:

- **Semilla 2026** (N=12.905):
  * B10 Baseline: Acc `0.92879` | LL `0.18938`
  * Text+Cat Standalone: Acc `0.84084` | LL `0.37071`
  * Inductive LGB Standalone: Acc `0.92639` | LL `0.18583`
  * Ensamble Blend: Acc **`0.93305`** | LL **`0.17752`** (ΔLL: **-0.01186**)
  * vs B10: **+55 net fixes** [9.0, 104.0] (Vistos: -11, Nuevos: **+66**; Fixes=187, Breaks=132).
- **Semilla 314159** (N=12.843):
  * B10 Baseline: Acc `0.92268` | LL `0.20000`
  * Text+Cat Standalone: Acc `0.83610` | LL `0.37858`
  * Inductive LGB Standalone: Acc `0.91723` | LL `0.20786`
  * Ensamble Blend: Acc **`0.92393`** | LL **`0.19195`** (ΔLL: **-0.00805**)
  * vs B10: **+16 net fixes** [-30.0, 62.0] (Vistos: -8, Nuevos: **+24**; Fixes=185, Breaks=169).
- **Semilla 20261030** (N=12.881):
  * B10 Baseline: Acc `0.93137` | LL `0.17558`
  * Text+Cat Standalone: Acc `0.85017` | LL `0.35335`
  * Inductive LGB Standalone: Acc `0.92578` | LL `0.18042`
  * Ensamble Blend: Acc **`0.93401`** | LL **`0.16894`** (ΔLL: **-0.00664**)
  * vs B10: **+34 net fixes** [-4.0, 72.0] (Vistos: +2, Nuevos: **+32**; Fixes=158, Breaks=124).

---

### Fase 2: Confirmación en Blind Holdout Virgen (Corte 42026)
Para asegurar que las ganancias no fuesen producto de la reutilización de cortes de desarrollo, se evaluó el ensamble con parámetros congelados sobre el corte **`42026`** (13.197 ítems en confirm: 3.744 vistos, 9.453 nuevos vendedores), el cual **jamás había sido observado ni utilizado para seleccionar hiperparámetros**:

- **Baseline Canónico (CatBoost Pipeline A)**: Acc `0.92142` | LL `0.19636`
- **Ensamble Candidato (Contextual Text + Inductive Interactions + Baseline)**: Acc **`0.92779`** | LL **`0.18514`**
- **Diferencia de Log Loss**: **`-0.01122`**
- **Net Fixes**: **`+84`**
- **Intervalo de Confianza Bootstrap 95% (2.000 réplicas por vendedor)**: **`[+42.0, +127.0]`**
  * *El intervalo excluye el cero de forma contundente (límite inferior +42 aciertos).*
- **Desglose de Fixes y Breaks**:
  * Fixes: **214** | Breaks: **130**
  * Vendedores vistos: **`+16`** aciertos netos
  * Vendedores nuevos: **`+68`** aciertos netos

---

## 3. Estado de la Política de Submissions y Recomendaciones para Ariel

1. **Líder Público Vigente**:
   - Se mantiene intacto como referencia oficial la submission [`submissions/candidate_d1_d2_combined_on48.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/candidate_d1_d2_combined_on48.csv) (**0.94652 LB**, SHA-256: `57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502`).
2. **Recomendación sobre Nuevas Sondas o Candidatos**:
   - La nueva arquitectura contextual inductiva ha demostrado, por primera vez con significancia estadística en blind holdout virgen (+84 aciertos, IC [+42, +127]), que es capaz de reparar confusiones en vendedores nuevos (+68 aciertos netos) sin degradar vendedores vistos (+16 aciertos netos).
   - Sin embargo, para evitar el riesgo de gastar cupos en envíos masivos o microsondas no calibradas:
     * **No se recomienda gastar cupos inmediatos en reemplazos completos de 300+ flips**.
     * Si Ariel decide realizar una nueva submission, la vía de menor riesgo es extraer una **sonda focalizada de desacuerdo de alta confianza (15–20 flips)** exclusivamente en vendedores nuevos donde el nuevo modelo inductivo rectifica sobre la base de 0.94652, verificada previamente con `csv.DictReader`.
3. **Artefactos Guardados**:
   - Reporte JSON consolidado: [results/intensive_20261030/nocturnal_batch_experiments_report.json](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/nocturnal_batch_experiments_report.json)
   - Predicciones pareadas del blind holdout: `results/intensive_20261030/paired_blind_confirm_seed42026.csv.gz`
   - Código ejecutable y reproducible: [src/intensive_20261030/run_iterative_nocturnal_batch.py](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/run_iterative_nocturnal_batch.py)
   - Cero submissions a Kaggle. Todos los cupos preservados.
