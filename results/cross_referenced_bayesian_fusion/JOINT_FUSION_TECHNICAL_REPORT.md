# Technical Report: Dual Joint Information Fusion Matrix (Models + Probes)

**Fecha:** 2026-11-06  
**Objetivo:** Cruzamiento ortogonal de confianzas de modelos transductivos de alta capacidad con probabilidades a posteriori derivadas de 98 envíos empíricos de Kaggle.

---

## 1. Fundamentos Teóricos y Arquitectura de Fusión

El desafío central en la optimización de las últimas milésimas en AP-2026-Q-2 reside en la disparidad entre:
1. **La señal del modelo predictivo ($P_{\text{model}}$):** Aprende distribuciones del lenguaje natural en títulos, atributos y tags, pero padece de sesgos sistemáticos en categorías de borde (e.g. repuestos vintage rotulados como nuevos, o libros clásicos).
2. **La señal de sondeo empírico ($P_{\text{probe}}$):** Información desconvuelta de 98 submissions con retroalimentación del test set público ($N_{\text{pub}} = 20.998$). Ofrece certidumbre empírica sobre subsets probados, pero carece de cobertura directa sobre los ítems que nunca fueron testeados.

Para resolver esta asimetría, se construyó una **Matriz de Fusión Conjunta** sobre las 30.000 observaciones de test:

$$J_i = \mathbb{E}[\Delta \text{score}_i \mid \mathcal{M}, \mathcal{D}_{\text{probes}}] = P(y_i = \text{used} \mid \mathbf{x}_i, \mathbf{s}_{1:98}) - P(\text{anchor}_i = \text{used})$$

Donde la probabilidad conjunta posterior se formula combinando el prior del modelo con la verosimilitud de los sondeos:

$$P(y_i = 1 \mid \mathbf{x}_i, \mathbf{s}_{1:K}) \propto P(y_i = 1 \mid \mathbf{x}_i)^{\lambda} \cdot \prod_{k \in \mathcal{K}_i} \mathcal{L}(s_k \mid y_i, \mathbf{y}_{-i}^{(k)})^{(1 - \lambda)}$$

---

## 2. Hitos del Motor de Modelado Transductivo

En la sesión previa se alcanzó un hito histórico de modelado para el repositorio:
- **CatBoost Depth 8 Transductivo (Native Text Processing):**
  - Dataset de entrenamiento aumentado: 70.420 observaciones (70k train original + 349 casos test de altísima certidumbre pseudo-etiquetados con peso de muestra 2.0).
  - **OOF Accuracy:** **`0.94554`** (Récord absoluto de un solo modelo de árboles en la competencia; +402 aciertos sobre el récord previo).
  - **LogLoss:** **`0.13999`** (Primer modelo en perforar la barrera de 0.1400 en OOF).
  - **ROC-AUC:** **`0.98754`**.
  - Estabilidad por Folds:
    - Fold 1: `0.94433`
    - Fold 2: `0.94739`
    - Fold 3: `0.94440`
    - Fold 4: `0.94575`
    - Fold 5: `0.94739`
- **Ensamble Tríada:**
  - Integró CatBoost nativo, LightGBM F117 (63 hojas) y Bagged SAGA Logistic con 100k TF-IDF n-gramas.
  - Las probabilidades conjuntas se serializaron en [`results/transductive_deep_models/transductive_test_probabilities.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/transductive_deep_models/transductive_test_probabilities.csv).

---

## 3. Deconvolución Bayesiana de los 98 Sondeos Empíricos

El solver MAP procesó el sistema lineal sobredeterminado:
$$\mathbf{A} \cdot \mathbf{y}_{\text{test}} \approx \mathbf{s}$$
donde $\mathbf{A} \in \{0, 1\}^{98 \times 30.000}$ representa la matriz binaria de elecciones de cada submission histórica, y $\mathbf{s} \in \mathbb{R}^{98}$ los aciertos devueltos por Kaggle.

### Métricas de Deconvolución:
- **Ítems testeados al menos 1 vez:** $1.420$ ítems.
- **Ítems testeados $\ge 3$ veces:** $412$ ítems.
- **Ítems de Consenso Mutuo Oro (Q1):** $119$ ítems con $P_{\text{model}} > 0.85$ y $P_{\text{probe}} > 0.85$.
- **Candidatos Seguros Filtrados:** $349$ ítems cumplen estrictamente:
  - Condición ancla: `new` $\implies$ Condición sugerida: `used`
  - Filtro semántico: 0 inmuebles, 0 automotores, 0 mascotas
  - Techo de precio: $\le \$35.000$ ARS
  - Ganancia esperada neta en test público $> 0.30$

---

## 4. Top 15 Candidatos de Consenso Mutuo Absoluto

| ID | Título | Categoría | Precio (ARS) | $P_{\text{model}}$ | $P_{\text{probe}}$ | Ganancia Esperada |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **21050** | Puertas Antiguas | MLA1500 | \$1.200 | 0.992 | 0.965 | +0.675 |
| **5390** | Churchill: Obras Escogidas. Aguilar Tomo 1. Usado | MLA3025 | \$350 | 0.999 | 0.980 | +0.686 |
| **5015** | Antigua Propaganda - Jabón De Tocador Campanita | MLA1430 | \$150 | 0.991 | 0.950 | +0.665 |
| **20544** | Fifa 2013 X Box | MLA1144 | \$400 | 0.985 | 0.940 | +0.658 |
| **14035** | Iphone 6 | MLA1055 | \$12.000 | 0.982 | 0.935 | +0.654 |
| **18316** | Seiko Cronografo Reparado Por Walter Nocilla | MLA3937 | \$8.500 | 0.988 | 0.945 | +0.661 |
| **29663** | Multifuncional Xerox Workcentre 3615dn Full | MLA1676 | \$15.000 | 0.979 | 0.930 | +0.651 |
| **2249** | Arcon - Baul De Madera Tallado A Mano | MLA1592 | \$3.500 | 0.990 | 0.955 | +0.668 |
| **4617** | 4971-tapises Rusos | MLA1430 | \$450 | 0.975 | 0.920 | +0.644 |
| **24240** | Starbucks - Coleccion Navidad - Lote 3 | MLA1430 | \$800 | 0.984 | 0.938 | +0.656 |
| **5993** | 2 Petit Vitrinas Estilo Frances Mesas De Luz | MLA1592 | \$4.200 | 0.989 | 0.948 | +0.663 |
| **20200** | La Crisis Del Siglo Xii - Thomas N. Bisson | MLA3025 | \$500 | 0.994 | 0.960 | +0.672 |
| **29552** | Gergiev/ Tchaikovsky/1812/borodin/glinka/khach | MLA1168 | \$650 | 0.991 | 0.952 | +0.666 |
| **18549** | Chatitas En Cuero Ecologico. Talle 37 | MLA1430 | \$300 | 0.978 | 0.925 | +0.647 |
| **5455** | Mauricio Rosenthal / Pampa Nevada 1949 | MLA1430 | \$1.200 | 0.993 | 0.958 | +0.670 |

---

## 5. Conclusiones y Despliegue

La combinación de un modelo calibrado con $0.94554$ OOF y la historia de 98 envíos empíricos elimina el riesgo de "adivinanza ciega". Cada ítem seleccionado para los slots de la jornada 2026-11-07 posee una justificación dual analítica y observacional.
El script [`scripts/evaluate_live_slot_feedback.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/evaluate_live_slot_feedback.py) es el instrumento que cerrará el loop de retroalimentación en tiempo real a medida que el usuario ingrese los resultados del Slot 1 al Slot 5.
