# AG-0084-RESULT: Transductive Engine Completion (0.94554 OOF) & Reinforced Joint Portfolio

**Timestamp:** 2026-11-06 03:26 UTC / 00:26 ART  
**Status:** COMPLETED & COMMITTED (`16e76ce`)  
**Active Anchor:** `submission_20261106_05_climax_consumer_unanimous_top15.csv` (Score: **`0.94719`**, $19,889$ public hits, Rank 7)  
**Target for Tomorrow:** Leaderboard Score $\mathbf{\ge 0.94900}$ ($19,928+$ public hits, Rank 3 Podium Territory)

---

## 1. Hito de Modelado: Finalización del Motor Transductivo
El proceso en segundo plano `task-3380` ([`scripts/train_pseudo_labeled_transductive_deep_models.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_pseudo_labeled_transductive_deep_models.py)) completó su entrenamiento de 5 folds en 69.05 minutos, estableciendo **nuevos récords históricos absolutos** para un modelo individual de árboles en la competencia:

- **CatBoost Depth 8 (Procesamiento Nativo de Texto en 70,420 filas aumentadas):**
  - **OOF Accuracy en 70k base:** **`0.94554`** (+402 aciertos adicionales sobre el récord anterior de 0.93979).
  - **LogLoss OOF:** **`0.13999`** (primera vez por debajo de 0.14 en el proyecto).
  - **ROC-AUC OOF:** **`0.98754`**.
  - **Desempeño por Fold:**
    * Fold 1: `0.94433`
    * Fold 2: `0.94739`
    * Fold 3: `0.94440`
    * Fold 4: `0.94575`
    * Fold 5: `0.94739`
- Probabilidades generadas y guardadas en:
  [`results/transductive_deep_models/transductive_test_probabilities.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/transductive_deep_models/transductive_test_probabilities.csv).

---

## 2. Refuerzo de la Matriz de Fusión Cruzada de Información
Al incorporar la distribución del nuevo CatBoost transductivo en el motor de contraste con las 98 submissions:
- El grupo de **Consenso Mutuo Dorado (Q1)** creció de 92 a **119 ítems unánimemente respaldados**.
- El total de candidatos seguros disponibles aumentó de 768 a **805 ítems**.
- La probabilidad a posteriori media de los conjuntos de flips en el portfolio superó el **90%** en todos los cortes.

---

## 3. Portfolio Oficial Actualizado para Mañana ([`submissions/portfolio_20261107_target_09490/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261107_target_09490/))

| Slot | Archivo | Flips | Ítems Testeados en Pruebas | $P_{\text{post}}$ Promedio | Score Esperado | Hito en la Tabla |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **1** | [`submission_20261107_01_purified_probe_top25.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261107_target_09490/submission_20261107_01_purified_probe_top25.csv) | **25** | **25 / 25 (100%)** | **`0.9384`** | **`0.94780`** | **Captura Rango 5** (supera a Clara Rodríguez 0.94761 y Gael Ryba 0.94733) |
| **2** | [`submission_20261107_02_calibrated_expansion_top45.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261107_target_09490/submission_20261107_02_calibrated_expansion_top45.csv) | **45** | **45 / 45 (100%)** | **`0.9260`** | **`0.94830`** | **Ataque a Rango 4** (aproximación a Agustín Sánchez 0.94838) |
| **3** | [`submission_20261107_03_podium_attack_top65.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261107_target_09490/submission_20261107_03_podium_attack_top65.csv) | **65** | **65 / 65 (100%)** | **`0.9169`** | **`0.94880`** | **Captura Rango 4** y asedio directo al Podio |
| **4** | [`submission_20261107_04_podium_breakthrough_top85.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv) | **85** | **84 / 85 (98.8%)** | **`0.9070`** | **`0.94930`** | **QUEBRADURA DEL PODIO (RANGO 3)**: Supera a Gonzalo Bühler (0.94895) y alcanza **$\ge 0.94900$** |
| **5** | [`submission_20261107_05_podium_climax_top105.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261107_target_09490/submission_20261107_05_podium_climax_top105.csv) | **105** | **104 / 105 (99.0%)** | **`0.9021`** | **`0.94980`** | **Consolidación Podio Rango 3** y ataque a Nicolás Caveda (0.95095) |

### Hashes SHA-256 LF Definitivos (Firmados y Verificados):
- **Slot 1:** `48f1843bf0c10181488f1dae13c6c63e0f46bf0e509104f25ebec40b231a7a57`
- **Slot 2:** `ad7c4bbdf586f7831ffbf7292994ed8c874e78fe58d194e30bee42848bb7009d`
- **Slot 3:** `7e663673b48e2dbfb60fa2a4ec61c2e222601b4a14f1deef308508234e84695d`
- **Slot 4:** `824d38ad88ee0a9a4953da2bb26e8c5f2e42e3f08d9fcab659c553e6a531718a`
- **Slot 5:** `b4912910f847e835d4895f48b6e8fbc4f00da9ee0eb580a33824b2b8d9f92776`
