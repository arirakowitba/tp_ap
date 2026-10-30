# Auditoría Estructural de Candidatos 77–80 (Agente 6)

**Fecha:** 2026-10-30  
**Objetivo:** Cuantificar intersecciones, diferencias, conflictos direccionales y diseñar contrastes anidados explícitos para evitar sumas ilegítimas de deltas sobre bloques solapados.

---

## 1. Matriz de Intersección de IDs Volteados (Conteos Absolutos)

| Candidato | Sub 77 (N=20) | Sub 78 (N=25) | Sub 79 (N=20) | Sub 80 (N=15) |
| :--- | :---: | :---: | :---: | :---: |
| **Sub 77** (Dual CB High Margin) | **20.0** | 10.0 | 0.0 | **15.0** |
| **Sub 78** (Model Consensus) | 10.0 | **25.0** | 0.0 | 8.0 |
| **Sub 79** (New Sellers Clean Text) | 0.0 | 0.0 | **20.0** | 0.0 |
| **Sub 80** (Seen Sellers Conservative) | **15.0** | 8.0 | 0.0 | **15.0** |

## 2. Matriz de Similitud Jaccard

| Candidato | Sub 77 | Sub 78 | Sub 79 | Sub 80 |
| :--- | :---: | :---: | :---: | :---: |
| **Sub 77** | 1.000 | 0.286 | 0.000 | **0.750** |
| **Sub 78** | 0.286 | 1.000 | 0.000 | 0.250 |
| **Sub 79** | 0.000 | 0.000 | 1.000 | 0.000 |
| **Sub 80** | **0.750** | 0.250 | 0.000 | 1.000 |

---

## 3. Hallazgos Críticos de la Auditoría

1. **Sub 80 es un subconjunto estricto de Sub 77 ($15 \subset 20$):**
   - Todos y cada uno de los 15 IDs de Sub 80 están presentes en Sub 77 con idéntica dirección.
   - La diferencia $Sub\,77 \setminus Sub\,80$ consta exactamente de **5 ítems**: `[732, 4969, 18735, 18790, 25747]`.
   - **Regla metodológica:** Nunca sumar $\Delta_77 + \Delta_80$. La comparación $\Delta_77 - \Delta_80$ es un **contraste anidado** que mide exclusivamente el impacto de esos 5 ítems marginales.

2. **Sub 78 se solapa fuertemente con Sub 77 (10 ítems) y Sub 80 (8 ítems):**
   - De los 25 ítems de consenso en Sub 78:
     * 10 ítems están en Sub 77 (8 de ellos también en Sub 80).
     * 15 ítems son **exclusivos de consenso** (no están ni en 77 ni en 80).
     * No hay contradicciones de dirección: en todos los ítems compartidos, ambos modelos votan hacia el mismo valor.

3. **Sub 79 es estrictamente disjunto ($0$ ítems solapados):**
   - Evalúa 20 ítems de vendedores nuevos mediante señales léxicas y garantías.
   - Cero solapamiento con Sub 77, 78 y 80. Su score público es 100% ortogonal.

---

## 4. Rediseño Operativo: Bloques Disjuntos y Contrastes Anidados

Para evitar cualquier contaminación o doble conteo al ensamblar el 5to cupo, las pruebas se estructuran formalmente en:

- **Bloque A (Seen Conservador Base)**: Sub 80 (15 ítems).
- **Contraste Anidado B (Delta Margen Alto)**: Sub 77 menos Sub 80 (5 ítems: `[732, 4969, 18735, 18790, 25747]`).
  * Si $Score(77) > Score(80)$, se autorizan los 5 ítems marginales.
  * Si $Score(77) \le Score(80)$, los 5 ítems marginales quedan descartados.
- **Bloque C (Consenso Puro Disjunto)**: Sub 78 Disjoint (15 ítems exclusivos de consenso).
- **Bloque D (Texto Vendedores Nuevos)**: Sub 79 (20 ítems, 100% disjunto).

### Árbol de Decisión para el Constructor Adaptativo (`build_from_scores.py`):
1. **Regla de Unión Disjunta:** Solo se unen conjuntos disjuntos de ítems positivos.
2. **Prohibición de Suma de Deltas Solapados:** El delta estimado de un candidato compuesto no es la suma de deltas si comparten IDs.
3. **Condición de Entrada:** Un bloque solo se incorpora si su score específico supera a Sub 48 ($> 0.94633$) y su contraste anidado es positivo.
