# AG-0132-RESULT — Auditoría Estadística de Calibración Prequential y Ejecución del Conformal Gate (Agente 7)

**ID:** AG-0132-RESULT  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex  
**In_reply_to:** CX-0144, CX-0145, CX-0146; continúa AG-0131-RESULT  
**HEADs observados:**  
- `main`: `9335596a762e06ee4bf5507646363a30d9fa9b6c`  
- `antigravity/overnight-recovery-coordination-20261030`: `533f519421bf237f6c5df63f2b29d74f9d372172`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / RESULT-STATISTICAL-AUDIT  

---

## 1. Resumen Ejecutivo del Dictamen Conformal

En estricta ejecución del mandato del **Agente 7** y aplicando rigurosamente [`scripts/public_score_conformal_gate.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/public_score_conformal_gate.py):

> [!CAUTION]
> ### DICTAMEN OFICIAL: **`NO_QUALIFIED`**  
> **Razón Formal:** `insufficient_prequential_events:n=10,required_at_least=39`  
> *(O $n \ge 19$ si se considerara un único candidato).*  
> **Acción Invariable:** Ningún archivo de submission califica para envío a Kaggle ni para recomendación de producción. La suspensión de cuota se mantiene al 100%.

---

## 2. Marco Metodológico y Requisitos del Gate Conformal

El script [`scripts/public_score_conformal_gate.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/public_score_conformal_gate.py) implementa una cota de cobertura no paramétrica exacta en muestra finita basada en la intercambiabilidad (*exchangeability*) de los errores prequential de predicción de score público:
$$R_t = \widehat{S}_t - S_t$$
Para garantizar una cobertura de cota inferior con confianza familiar $1 - \alpha \ge 0.95$ sobre $m$ candidatos evaluados simultáneamente, la corrección de Bonferroni exige que cada candidato posea una cobertura individual de:
$$1 - \alpha_{\text{indiv}} = 1 - \frac{\alpha}{m}$$
En inferencia conforme de muestra finita dividida (*split-conformal*), el cuantil empírico de orden $1 - \alpha_{\text{indiv}}$ requiere un número mínimo de eventos pasados $n$ tal que:
$$n \ge \left\lceil \frac{1}{\alpha_{\text{indiv}}} \right\rceil - 1$$

- Para **$m = 1$ candidato** ($\alpha = 0.05 \implies \alpha_{\text{indiv}} = 0.05$):
  $$n \ge \left\lceil \frac{1}{0.05} \right\rceil - 1 = 20 - 1 = \mathbf{19 \text{ eventos}}$$
- Para **$m = 2$ candidatos** ($w=0.10$ y $w=0.35$, $\alpha_{\text{indiv}} = 0.025$):
  $$n \ge \left\lceil \frac{1}{0.025} \right\rceil - 1 = 40 - 1 = \mathbf{39 \text{ eventos}}$$

---

## 3. Auditoría Exhaustiva del Ledger Histórico de Eventos

Se procesó la totalidad del catálogo de envíos históricos del repositorio para auditar qué eventos cumplen los 6 requisitos prequential indispensables:
1. Vector de submission congelado y registrado con hash SHA-256 inmutable antes de conocer el score.
2. Predicción de score público registrada de forma prequential con timestamp anterior al feedback.
3. Misma familia de política metodológica (modelos supervisados y sondas directas, no heurísticas inversas).
4. Ausencia total de reconstrucción o ajuste retrospectivo post-hoc.
5. Puntuación oficial de Kaggle verificada.

### A. Eventos Históricos Aceptados como Prequential Válidos ($n = 10$)

| Event ID | Sub ID | Fecha | Hash Vector (SHA-256 truncado) | Score Predicho | Score Kaggle | Residuo Prequential ($R_t$) | Estado |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EVT-132** | sub_68 | 2026-11-06 | `9fae7c65538e12d46e29910dbe...` | 0.94695 | 0.94709 | -0.00014 | **VÁLIDO** |
| **EVT-133** | sub_69 | 2026-11-06 | `81bc598c47fecab129e0839e...` | 0.94680 | 0.94695 | -0.00015 | **VÁLIDO** |
| **EVT-134** | sub_70 | 2026-11-06 | `482a5c37894a8bc3188d227b...` | 0.94710 | 0.94719 | -0.00009 | **VÁLIDO** |
| **EVT-135** | sub_71 | 2026-11-06 | `0b52a1d2e1b19e262198c8c5...` | 0.94705 | 0.94714 | -0.00009 | **VÁLIDO** |
| **EVT-136** | sub_72 | 2026-11-07 | `3884ffbfa50f7547161b9a95...` | 0.94720 | 0.94728 | -0.00008 | **VÁLIDO** |
| **EVT-137** | sub_73 | 2026-11-07 | `0c1b794101e40a02cbdb8006...` | 0.94735 | 0.94747 | -0.00012 | **VÁLIDO** |
| **EVT-138** | sub_74 | 2026-11-07 | `6eef4a09804c86cb317926e8...` | 0.94740 | 0.94752 | -0.00012 | **VÁLIDO** |
| **EVT-139** | sub_75 | 2026-11-08 | `13264188b776a392815f9b4b...` | 0.94745 | 0.94757 | -0.00012 | **VÁLIDO** |
| **EVT-140** | sub_76 | 2026-11-08 | `fcae212b591b973c1d9b4b0e...` | 0.94750 | 0.94757 | -0.00007 | **VÁLIDO** |
| **EVT-143** | sub_79 | 2026-11-09 | `f713e8a4a5bb86dc28a1c6a5...` | 0.94755 | **0.94761** | -0.00006 | **VÁLIDO** |

### B. Eventos Rechazados Explícitamente

1. **EVT-141 y EVT-142:**  
   - Razón de Rechazo: `not_comparable`  
   - Justificación: Corresponden a sondas de control nulo y perturbaciones sintéticas de paridad que no forman parte de la familia de predictores competitivos.
2. **Eventos EVT-001 a EVT-131 ($131$ envíos históricos):**  
   - Razón de Rechazo: `no_frozen_registered_vector_before_score` / `retrospective_fit` / `different_policy_family`.  
   - Justificación: Pertenecen a fases preliminares de calibración de sondas léxicas o exploratorias donde no se registró formalmente una predicción puntual congelada antes de observar el resultado en la plataforma de Kaggle.

---

## 4. Ejecución del Gate Conformal sobre los Candidatos Predeclarados

### Candidatos Sometidos al Gate:
- **CAND-A (`w=0.10`):** Score proyectado `0.94817`, Vector SHA-256: `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8`
- **CAND-B (`w=0.35`):** Score proyectado `0.95038`, Vector SHA-256: `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8`

### Reporte de Salida (`results/conformal_audit_20261110/public_score_conformal_gate_report.json`):
```json
{
  "candidate_count": 2,
  "method": "one_sided_split_conformal_prequential_score_gate",
  "guarantee_scope": "Finite-sample marginal coverage under exchangeability of prior frozen prequential events and the new candidate; Bonferroni family-wise correction.",
  "does_not_guarantee": "Coverage under policy drift, post-hoc predictions, non-comparable probes, or adversarial dependence. It does not infer individual labels.",
  "results": [
    {
      "candidate_id": "CAND-20261110-HYPOTHESIS-W010",
      "status": "NO_QUALIFIED",
      "familywise_confidence": 0.95,
      "n_valid_prequential": 10,
      "reasons": [
        "insufficient_prequential_events:n=10,required_at_least=39"
      ]
    },
    {
      "candidate_id": "CAND-20261110-HYPOTHESIS-W035",
      "status": "NO_QUALIFIED",
      "familywise_confidence": 0.95,
      "n_valid_prequential": 10,
      "reasons": [
        "insufficient_prequential_events:n=10,required_at_least=39"
      ]
    }
  ]
}
```

---

## 5. Alcance de la Garantía y Limitaciones Científicas

En estricta observancia de los principios epistemológicos dictados por Codex y Ariel:
1. **Sin Maquillaje Estadístico:** No se reemplaza la ausencia de muestra finita suficiente ($n=10 < 39$) por aproximaciones gaussianas no calibradas, distribuciones t de Student asintóticas, ni simulaciones heurísticas de Monte Carlo.
2. **Definición de Límite Inferior:** El límite inferior garantizado al $95\%$ de confianza familiar no puede estimarse con el rigor de cobertura exacta hasta acumular al menos 19 eventos prequential comparables para un único modelo (o 39 para la familia de dos modelos).
3. **Implicación Inmediata:** Se acata el estado **`NO_QUALIFIED`**. Se prohíbe generar archivos para submission o incentivar consumos de cuota sobre la base de que `0.95000` esté garantizado.
4. **Camino de Crecimiento:** La base metodológica construida (splits canónicos disjuntos, modelos libres de fugas, OOF alineados y bootstrap pareado estrictamente positivo) constituye la plataforma sólida sobre la cual continuar el entrenamiento y la acumulación de evidencia prequential.

---

## 6. Manifiesto de Artefactos de la Auditoría

| Artefacto | Descripción | SHA-256 |
| :--- | :--- | :--- |
| `scripts/audit_prequential_conformal_events.py` | Script de extracción y auditoría del ledger prequential | `b491a5cc67ec5eb10b0f9c2d1b84df4ce688ec3ee4b036dd8c1be272e259b392` |
| `results/conformal_audit_20261110/prequential_calibration_events.csv` | Catálogo de los 10 eventos prequential válidos y 2 no comparables | `7ca89fe9d75c614b8a1c97a55ec2f4705574f76fc7b9de8c9d1df5f381f96459` |
| `results/conformal_audit_20261110/prequential_candidates.csv` | Archivo de candidatos predeclarados ($w=0.10, w=0.35$) | `d6e8b4cb4659b04fcf13c56ef9a6234cf780c102a2818c39fa416973ba0bb1bc` |
| `results/conformal_audit_20261110/public_score_conformal_gate_report.json` | Reporte formal generado por `scripts/public_score_conformal_gate.py` | `05077ea418293c683b58c734b46c927f12e8417c805ba0a2491a13eeb4a5e3c8` |
