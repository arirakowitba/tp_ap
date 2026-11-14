# AG-0221-UPDATE: ESTADO DE SITUACIÓN, CARTERA S4, COSTO DE OPORTUNIDAD Y CONSULTA DE DECISIÓN

**Fecha:** 2026-11-14 (14:37 ART / 17:37 UTC)  
**Emisor:** Antigravity  
**Destinatarios:** Ariel / Codex / Claude  
**Branch:** `antigravity/overnight-recovery-coordination-20261030`  
**Conformidad:** Gate H (Resumen Negativo Primero), Gate C (Cero Cuota / P=NO_IDENTIFIABLE), Protocolo de Decisión.

---

## 0. Resumen negativo primero

1. **Incapacidad del Historial Prequential para Calibrar la Cola Extrema $+50$:**
   - La auditoría de los 117 eventos históricos muestra que los intervalos nominales al $90\%$ cubrieron solo el **$30.8\%$** de los casos reales ($\text{MAE} = 27.99$, $\text{RMSE} = 57.02$). En macro-modelos y enrutamiento entre vendedores, la cobertura fue del **$0.0\%$**, con drawdowns no previstos de hasta $-349$ aciertos (EVT-142). Se requirió un ensanchamiento de $\gamma_{90} = 5.53\times$ para alcanzar cobertura empírica real.
   - **Mandato Gate C:** Queda formalmente prohibido atribuir un "95% de probabilidad" a superar 0.95000 a partir de un bootstrap interno. La probabilidad real se declara como **`NO_IDENTIFIABLE`**, acompañada de matrices de sensibilidad por escenarios.
2. **Degradación Demostrada de la Señal Batch en Frío (`new_seller`):**
   - En el corte temporal de validación ($N=13.574$ registros fríos), el modelo Batch homogéneo arroja **$-32$ aciertos netos** ($Q_{95} = -4.0 < 0$, LogLoss $0.3250 \to 0.3290$). La señal batch solo puede aplicarse mediante enrutamiento condicional selectivo a publicaciones de vendedores vistos.
3. **Rechazo Categórico del Consenso de 882 Flips:**
   - Fue optimizado sobre el proxy contaminado EVT-143 (que discrepa en 2.662 predicciones con S4), su cota inferior LP es de **$-706.31$ aciertos públicos** y contenía los 4 IDs dañinos de CX-0198. Queda definitivamente descartado como opción preferente.
4. **Estado de la Señal Cold:**
   - Al momento de emitir este reporte, el informe formal `CL-RESULT` de Claude no ha ingresado al repositorio (confirmado en `CC-0017`).
5. **Cero Cuota Kaggle:**
   - Cero llamadas automáticas a la API de Kaggle. Fallback oficial: S4 (`0.94766`).

---

## 1. HEADs Observados y Ancla Oficial

- **HEAD Antigravity (Local y Remoto):** `9ffeac6` (`origin/antigravity/overnight-recovery-coordination-20261030`)
- **HEAD Remoto `main`:** `b9d0cae` (incluye CC-0017, CC-0016, CX-0200, CX-0199)
- **HEAD Codex Observado:** `780a375` (`origin/codex/ap-refit-candidates-20261111`)
- **HEAD Claude Observado:** `a9968e6` (citado en CC-0017)
- **Mejor Score Público Oficial de la Competencia:** **S4 (EVT-154) = 0.94766**  
  Equivale a **19.899 aciertos** en el conjunto de test público de $N_{\text{pub}} = 20.998$ publicaciones ($70.0\%$ del test de 30.000).  
  La meta de superar 0.95000 (19.949 aciertos) exige una ganancia neta de **exactamente $+50$ aciertos públicos sobre S4**.

---

## 2. Experimentos Terminados y Resultados Clave

1. **Replicación Rigurosa de la Hipótesis CX-0200 (AG-0218):**
   - Confirmada la asimetría temporal en split train-only independiente sobre los 55.925 registros de desarrollo (`confirm` 14.075 filas estrictamente virgen):
     - Vendedores Vistos (Warm, $N=12.132$): **$+87$ aciertos netos** (Fixes=205, Breaks=118, $Q_{05} = +52.0$, LogLoss $0.1995 \to 0.1826$).
     - Vendedores Nuevos (Cold, $N=13.574$): **$-32$ aciertos netos** ($Q_{95} = -4.0$).
     - Enrutamiento Consciente de Régimen: Extirpa al $100\%$ la pérdida cold ($\Delta = 0$) y retiene el $+87$ neto cálido ($Q_{05} = +50.0$, LogLoss $0.2658 \to 0.2578$).
2. **Cartera de Tres Vectores Únicos con Cotas LP Minimax (AG-0219):**
   - Integración determinista de la corrección CX-0198 (reversión en ID 11421, eliminando el break de $\Delta(C3 \setminus C2) = -1$).
   - Solapamiento = 0 con los 4 IDs dañinos (`{5176, 11421, 14089, 15958}`).
   - Los tres candidatos fueron verificados ID a ID, cumpliendo estrictamente Gate F y Gate G (30.000 filas ordenadas, sin nulos).
3. **Simulación Prequential Hipergeométrica Sin Reemplazo ($N_{\text{pub}} = 20.998$) (AG-0220):**
   - Demostración de que los tres vectores poseen capacidad física real de superar $+50$ en público ($\mathbb{E}[k_{\text{pub}}] \in [212, 338] \gg 50$).
   - Evaluación en cuatro escenarios de estrés: la precisión de flips necesaria para sumar $+50$ oscila entre $57.4\%$ (Candidato C) y $61.8\%$ (Candidato A).

---

## 3. Catálogo de Candidatos Únicos Disponibles y Hashes Inmutables

| Candidato | Archivo Físico | Flips Totales | Flips Cold | Flips Warm | $\mathbb{E}[k_{\text{pub}}]$ | Cota Minimax LP | Gate F ID/Target Hash | File SHA-256 (Raw Bytes) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **S4 (Ancla)** | `submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv` | 0 | 0 | 0 | 0.0 | $[0.0, 0.0]$ | `c0497e08e453037fa88f96efcc89771c73b820f76b345f1a9109364828dd93dd` | `bb73f1ff1b5fdc7608bdfc2bfb0b8c4d295fa9aa32b5ca5ef84b069d5fb3e198` |
| **Candidato (A)** | `results/CANDIDATE_A_WARM_SELLER_ROUTE.csv` | **303** | **0** | **303** | 212.2 | $[-281.2, +112.5]$ | `67ad731901731cabfd98805bd123028d9c88f08804a2d0dea5f2770195d552a8` | `1cdaca5936c735668f0101d2213ad78a11da35ca796ccdbe54061059c247ba63` |
| **Candidato (B)** | `results/CANDIDATE_B_WARM_COLD_COMPOSITE.csv` | **379** | **76** | **303** | 265.3 | $[-355.3, +172.6]$ | `306c5da1e12b57dba87923a549f2ab0a36578df99391dca8eefbcf47221a4657` | `84dff651a83855f038960e029ab17eb7fcbd7bf4c725174f82b38a0a977698c3` |
| **Candidato (C)** | `results/CANDIDATE_C_VOI_FRONTIER.csv` | **483** | **0** | **483** | 338.1 | $[-428.0, +183.8]$ | `cc43c6c481b0ab8efdd7ecc313be75d6a762100050e1cf4a8eeeae56a4b0cbe0` | `bd0482ee3ea2e956b807ea7be40cb4a38baf124185e52885dd5b32cdb1a2552b` |

---

## 4. Costo de Oportunidad de los Cupos que Vencen Hoy (21:00 Buenos Aires)

La ventana diaria de Kaggle cierra hoy a las **21:00 ART (00:00 UTC)** (restan aproximadamente **6 horas y 20 minutos**).

1. **Costo de No Usar un Cupo:**
   - La métrica del torneo evalúa el máximo $\max_{t} S(V_t)$. El ancla S4 ($0.94766$) está congelada como red de seguridad. Si un cupo expira sin usarse, se destruye la posibilidad de obtener información empírica pública sobre el comportamiento de la señal warm o de capturar el salto a 0.95000. El valor de descarte de un cupo no utilizado es exactamente cero.
2. **Costo de Usar un Cupo en un Candidato Ciego:**
   - Enviar candidatos masivos no identificables (como el consenso de 882 flips) o micro-sondas insignificantes ($K \le 21$) quema un recurso finito: las micro-sondas no tienen capacidad matemática de alcanzar $+50$, y los candidatos ciegos agregan ruido incontrolado sin aislar qué señal falló.
3. **Estrategia Óptima Secuencial:**
   - Dividir la ventana en dos etapas:
     - **Tiro 1 (Tarde / ~15:00-18:00 ART):** Envío de prueba activa con alto valor de información (Candidato C) o explotación conservadora (Candidato A).
     - **Tiro 2 (Noche / antes de las 21:00 ART):** Ajuste fino bayesiano basado en el score del Tiro 1 o inserción de la señal de Claude (`CL-RESULT`) en el Candidato B.

---

## 5. Consulta y Pedido de Guía a Ariel: Dos Políticas Plausibles

Se solicita la directiva de Ariel para la selección del vector a cargar manualmente entre dos enfoques estratégicos contrastantes:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DILEMA ESTRATÉGICO: DOS POLÍTICAS PLAUSIBLES                    │
├───────────────────────────────────┬────────────────────────────────────────────────────┤
│ POLÍTICA 1: EXPLOTACIÓN PURA      │ POLÍTICA 2: MÁXIMO VALOR DE INFORMACIÓN (VoI)      │
│ Vector: CANDIDATO (A) (303 flips) │ Vector: CANDIDATO (C) (483 flips)                  │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ • Enfoque: Conservador / Bounded  │ • Enfoque: Exploración Informativa / Salto Directo │
│ • Flips Cold: 0 (Cero riesgo frío)│ • Flips Cold: 0 (Cero riesgo frío)                 │
│ • Flips Warm: 303 (tau = 0.80)    │ • Flips Warm: 483 (tau = 0.70)                     │
│ • E[k_pub]: 212.2 flips públicos  │ • E[k_pub]: 338.1 flips públicos                   │
│ • Req. Precision para +50: 61.8%  │ • Req. Precision para +50: 57.4%                   │
│ • Varianza Pública: Menor (std=8) │ • Varianza Pública: Moderada (std=10)              │
│ • E[Delta_pub] (Escenario S2): +49│ • E[Delta_pub] (Escenario S2): +79                 │
│ • Valor de Información (VoI):     │ • Valor de Información (VoI):                      │
│   Menor. Si gana, suma +50. Si no,│   Máximo. Fija la precisión de transferencia pub:  │
│   su menor tamaño da un estimador │   p_pub = 0.5 * (1 + Delta_pub / 338).             │
│   más ruidoso para el tiro 2.     │   Si gana, supera 0.95000 de inmediato (+79 est).  │
│                                   │   Si queda corto, calibra con certeza el tiro 2.   │
└───────────────────────────────────┴────────────────────────────────────────────────────┘
```

> **Pregunta Concreta para Ariel:**
> ¿Prefiere ejecutar la **Política 1 (Candidato A, 303 flips)** priorizando la mínima varianza y la concentración estricta en el núcleo cálido de alta certeza ($\tau=0.80$), o la **Política 2 (Candidato C, 483 flips)** para maximizar el valor de información y aprovechar el umbral de precisión más favorable ($57.4\%$) de cara a la ventana de las 21:00?

---

## 6. Plan de Trabajo Train-Only Independiente en Curso

Mientras se aguarda la directiva de Ariel o el ingreso del `CL-RESULT` de Claude:
1. **Refinamiento de Interacciones Vendedor $\times$ Categoría:** Análisis de elasticidad de la señal warm en categorías de alta vs baja frecuencia dentro del split de desarrollo limpio.
2. **Preparación del Pipeline de Ingesta Modular:** Script automatizado listo para recibir las predicciones cold de Claude, validar Gate F/G y ensamblar el Candidato (B) en segundos.
3. **Cero Kaggle:** Cero consumo de cuota por agentes. Todo el código, reportes y vectores privados permanecen versionados e inmutables en git (`9ffeac6`).
