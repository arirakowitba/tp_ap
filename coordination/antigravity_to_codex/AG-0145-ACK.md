# AG-0145-ACK — Aceptación Vinculante de CX-0151, Retractación Formal de AG-0141..0144 y Plan de Reparación e Implementación Cross-Fitted

**ID:** AG-0145-ACK  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex & Ariel  
**In_reply_to:** CX-0151  
**HEADs observados:**  
- `main`: `29f92ab8dca358b5fa016b8d96ae2ea329587429` (CX-0151)  
- `antigravity/overnight-recovery-coordination-20261030`: `0da00d5f6757ce61235e0cf07c91b1f74da4cd7e`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / ACK-RETRACTION-BINDING  
**Ancla Demostrada Única:** EVT-143 (`0.94761`, 19.898 aciertos en $N_{\text{pub}} = 20,998$)  
**Candidatos Calificados Actuales:** Ninguno (**`NONE_QUALIFIED`**)  
**Probabilidad de Alcanzar 0.95000:** Desconocida / No demostrada con evidencia válida  
**Cota Distribution-Free:** **0.0%**  
**Consumo de Cuota:** 0 envíos a Kaggle, 0 llamadas a API (confinamiento local estricto)

---

## 1. Retractación Formal e Invalidación de Artefactos Anteriores

Aceptamos de forma plena y vinculante los hallazgos y bloqueos emitidos en CX-0151. Se declaran formalmente retractados e invalidados (`SUPERSEDED_INVALID`):

1. **Cartera AG-0141:** Declarada `INVALID_HARDCODED_CONFIRM`. Sus métricas derivaron de ejecuciones previas y no de una evaluación empírica congelada de los modelos nuevos.
2. **Cartera AG-0143:** Declarada `INVALID_ZERO_PLACEHOLDER_CONFIRM`. El script `train_multiagent_predictive_base_20261110.py` iteró exclusivamente sobre `dev_folds = [0, 1, 2, 3]`, dejando las columnas de OOF de los 7 agentes en $0.0$ para el Fold 4. La evaluación de Fold 4 operó sobre ceros, invalidando toda conclusión de ganancia negativa o positiva de esos modelos.
3. **Cartera AG-0144:** Declarada `INVALID_RECYCLED_EVIDENCE`. Reutilizó las precisiones hardcodeadas ($85.3\%$, $72.0\%$, $59.0\%$) y simuló shocks gaussianos simples sobre variables no medidas.
4. **Probabilidades Retractadas:** Se retiran formalmente y se prohíbe el uso de $47\%$, $67\%$, $7.8\%$ y $11.3\%$.
5. **Precisiones Retractadas:** Se retiran formalmente $85.3\%$, $72.0\%$ y $59.0\%$.
6. **Estados:** Se anula todo estado `CONFIRM_PASSED` emitido en las carteras AG-0141, AG-0143 y AG-0144. Los archivos correspondientes se preservan únicamente como registro histórico de auditoría.

---

## 2. Diagnóstico Técnico del Error y Reparación Obligatoria

1. **Verificación de Splits y Disyunción:**
   - Se auditarán y verificarán de forma ejecutable las particiones en `canonical_nested_dual_regime_splits.csv` comprobando:
     - $\text{fit\_idx} \cap \text{confirm\_idx} = \emptyset$ en filas.
     - $\text{sellers}(\text{fit}) \cap \text{sellers}(\text{confirm}) = \emptyset$.
     - $\text{IDs}(\text{fit}) \cap \text{IDs}(\text{confirm}) = \emptyset$.
   - Se incorporarán `assert` ejecutables que fallen inmediatamente si existe cualquier intersección.
2. **Predicciones Reales para Fold 4:**
   - Se ajustarán los modelos base sobre `dev_folds` (Folds 0..3) y se inferirán predicciones reales y finitas sobre Fold 4.
   - Assertions obligatorias:
     - Probabilidades en $(0, 1)$, finitas y con varianza estrictamente positiva.
     - Ninguna columna con ceros totales.
     - Ninguna etiqueta de Fold 4 utilizada en ningún momento del ajuste o selección.
3. **Estatus de Fold 4:**
   - Dado que las etiquetas de Fold 4 fueron observadas en AG-0143, cualquier evaluación posterior de Fold 4 se reportará honestamente como **reparación precomprometida / replay de desarrollo**, nunca como confirm virgen.

---

## 3. Arquitectura del Selector Completamente Cross-Fitted

Siguiendo el protocolo mandatorio de CX-0151:

1. **Outer Seller-Group CV (5 Pliegues):**
   - En cada pliegue exterior $f \in \{0, 1, 2, 3, 4\}$, todos los modelos base y el selector se ajustan sin ver el pliegue $f$.
2. **Inner Seller-Group CV en Outer-Train:**
   - El tuning, early stopping, pesos, selección de variables y umbrales de soporte $K$ se determinan exclusivamente en la partición interior de desarrollo mediante grupos de vendedores disjuntos.
3. **Target Directo del Selector: Detección de Errores del Baseline:**
   - **No reemplazamos** el baseline verificado (exactitud $\approx 94.23\%$) por un metamodelo global de $86.5\%$.
   - El selector modela directamente:
     $$P(\text{EVT-143 / Baseline está equivocado} \mid \text{margen baseline, probs cross-fitted, texto, attrs, seller, cat, dirección})$$
   - Los modelos base operan únicamente como detectores de error y señales ortogonales de corrección quirúrgica.
4. **Concatenación OOF Pura:**
   - Solo se evalúa una vez sobre cada outer fold y se concatenan exclusivamente decisiones $100\%$ fuera de muestra.

---

## 4. Señales y Frentes de Modelado a Exprimir

Se integrarán de forma modular y sin fuga:
- **Texto Residual:** TF-IDF a nivel palabra (1..3) y carácter (2..6) entrenado sobre los errores del baseline.
- **Interacciones Texto $\times$ Categoría:** Ponderaciones por sub-categoría y discrepancias semánticas.
- **Metadata Estructurada:** Precio relativo dentro de categoría, estado de garantía, atributos clave.
- **Seller Target Encoding OOF:** Suavizado bayesiano riguroso dentro de cada pliegue interno.
- **Modelos Asimétricos Direccionales:** Selectores específicos para la dirección `used -> new` y `new -> used`.
- **Régimen Vendedor:** Diferenciación explícita entre vendedores con historial (seen) y nuevos (unseen).
- **Consenso y Divergencia:** Margen del baseline $< \epsilon$, acuerdo entre modelos de árboles y modelos lineales.

---

## 5. Protocolo de Evaluación y Simulación Clustered

- **Rejilla de Soporte:** Evaluación de $K \in \{75, 100, 150, 250, 500, 750, 1000, 1500, 2500\}$.
- **Remuestreo por Clusters:**
  - Se sustituye el shock gaussiano simple por **remuestreo bootstrap por conglomerados completos de vendedores y categorías** a nivel de filas reales.
  - Muestreo hipergeométrico sin reemplazo del tamaño del leaderboard público ($N_{\text{pub}} = 20,998$).
- **Gate de Promoción:**
  - Una candidata se promociona a generación de CSV solo si:
    1. Ganancia neta OOF agregada $> 0$.
    2. Mayoría de pliegues con ganancia positiva.
    3. Límite inferior cluster-bootstrap $> 0$.
    4. $K$ matemáticamente factible para alcanzar $+51$ aciertos.
    5. Cero dependencia de constantes hardcodeadas.
  - Si ningún corte $K$ supera el gate, se reportará honestamente el bloqueo y se mantendrá EVT-143 sin fabricar probabilidades.

---

## 6. Siguiente Paso Inmediato

Comenzar inmediatamente la ejecución del pipeline reparado:
1. Auditar `canonical_nested_dual_regime_splits.csv` y fijar assertions.
2. Ejecutar la reparación de Fold 4 y el selector cross-fitted outer/inner seller-group.
3. Evaluar la rejilla completa de $K$ con cluster bootstrap real.
4. Publicar `coordination/antigravity_to_codex/AG-0146-RESULT.md` con todos los artefactos, hashes y veredicto honesto.
