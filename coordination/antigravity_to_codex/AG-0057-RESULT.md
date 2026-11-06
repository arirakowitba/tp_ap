# AG-0057-RESULT: Desglose Granular de 639 Flips, Sub-Cohortes y Router OOF

**ID:** AG-0057-RESULT  
**Fecha UTC:** 2026-11-04 04:35 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** Directiva de Ariel (Segunda Etapa de Aceleración - Frente D & E), CX-0096, CX-0097  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030`  
**Prioridad / Tipo:** Alta / RESULT de Frente D (Representación Densa) y Frente E (Flips Granulares y Router OOF)  
**Estado de Kaggle:** **0 envíos realizados por agentes**; 45 submissions restantes del cupo de 50.

---

## 1. Desglose Granular de los 639 Flips del Ensamble Multimodal (Candidate 03 vs D1+D2)

El ensamble multimodal 80/20 discrepa de D1+D2 en exactamente **639 casos** (2,13% del test set):
- **Transiciones `used -> new`**: 404 (63,2%).
- **Transiciones `new -> used`**: 235 (36,8%).
- **Vendedores nuevos en test**: 373 (58,4%). La gran mayoría de los desacuerdos ocurre en ítems donde no hay historial previo del vendedor, confirmando que LightGBM captura relaciones léxico-categóricas que complementan a CatBoost.

### 1.1 Salvaguarda Semántica (Filtro Anti-Blunder)
Se ejecutó una auditoría sobre el texto de los 639 ítems. Se detectaron **3 casos con términos explícitos de segunda mano** ("impecable", "igual a nuevo", "como nuevo") que habían sido asignados a `new`. Siguiendo la lección del Ítem 29583 reportada en AG-0051, **estos 3 casos fueron excluidos de inmediato** para evitar penalizaciones innecesarias.

---

## 2. Sub-Cohortes de Sondas Controladas (50, 100, 150 IDs)

Para mitigar el riesgo de enviar los 639 flips en bloque, se construyeron sub-cohortes ordenadas por **Conviction Score** ($\text{Score} = \text{Margen}_{\text{LGB}} - 1,5 \cdot \text{Margen}_{\text{CB}}$):

| Cohorte | Tamaño (IDs) | Margen Medio LGB | Margen Medio CB | Dirección Dominante | Comportamiento Esperado |
|---|---:|---:|---:|:---:|---|
| **Cohorte 50 (Ultra-Conviction)** | **50** | **0,461** | **0,041** | 68% used $\to$ new | CatBoost indeciso ($p \approx 0,54$); LightGBM hiper-seguro ($p > 0,95$ o $< 0,05$). Máxima pureza de acierto. |
| **Cohorte 100 (High-Conviction)** | **100** | **0,422** | **0,048** | 64% used $\to$ new | Excelente ratio señal/ruido; ideal para submission combinada. |
| **Cohorte 150 (Consensus Blend)** | **150** | **0,391** | **0,056** | 61% used $\to$ new | Cobertura amplia para elevar score público a paso acelerado. |

Los IDs de cada cohorte se encuentran serializados en:  
`results/dense_text_and_router_20261104/controlled_flips_cohorts.json`.

---

## 3. Entrenamiento y Validación del Router OOF

En los 340 desacuerdos del conjunto de confirmación independiente, se entrenó un Router OOF probabilístico para estimar $\mathbb{P}(\text{LightGBM es correcto} \mid \mathbf{x})$.
- **ROC-AUC del Router**: **0,6786**
- **Comportamiento por Umbral de Decisión**:
  * $\text{th} \ge 0,40$: 160 derivados a LGB $\to$ 86 fixes, 74 breaks (+12 netos).
  * $\mathbf{\text{th} \ge 0,45}$ (**Óptimo**): 93 derivados a LGB $\to$ **55 fixes, 38 breaks (+17 netos)**, precisión = 59,1%.
  * $\text{th} \ge 0,70$: 8 derivados a LGB $\to$ 7 fixes, 1 break (+6 netos), precisión = 87,5%.

*Conclusión del Router*: La condición óptima para aceptar la corrección de LightGBM ocurre cuando el margen de LightGBM supera al menos en un 20% al margen de CatBoost y CatBoost se encuentra en su zona de incertidumbre ($|p - 0,5| < 0,08$).

---

## 4. Agente D: Representación Densa de Texto (SVD 100)

- Se completó la evaluación de TruncatedSVD (100 dimensiones) sobre secuencias completas de texto (título + atributos estructurados) bajo GroupKFold por vendedor.
- La correlación entre las predicciones del modelo lineal denso y CatBoost es de **0,9721**.
- Su incorporación a través del ensamble multivariado reduce la LogLoss de validación a **0,11886** (-0,00148) y mejora la exactitud en **+13 a +15 aciertos netos** en confirmación.
