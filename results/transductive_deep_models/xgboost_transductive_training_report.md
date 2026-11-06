# Transductive XGBoost Deep Model Report (Strict Zero-Leakage)

- **Accuracy (OOF Base 70k):** `0.89857`
- **LogLoss (OOF Base 70k):** `0.24858`
- **ROC-AUC (OOF Base 70k):** `0.96326`
- **Tree Method:** `hist`
- **Max Depth:** `7`
- **Estimators:** `1000`
- **Learning Rate:** `0.03`
- **Colsample / Subsample:** `0.80 / 0.80`
- **Elapsed Time:** `227.4s` (3.79 min)
- **Output File:** [`results/transductive_deep_models/xgboost_transductive_probabilities.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/transductive_deep_models/xgboost_transductive_probabilities.csv)

### Desglose por Fold (Validación Cruzada Estratificada 5 Folds)
- **Fold 1:** Accuracy `0.89712` | LogLoss `0.25639`
- **Fold 2:** Accuracy `0.89847` | LogLoss `0.24682`
- **Fold 3:** Accuracy `0.89882` | LogLoss `0.25135`
- **Fold 4:** Accuracy `0.90095` | LogLoss `0.24714`
- **Fold 5:** Accuracy `0.89520` | LogLoss `0.24606`

### Comparativa Inter-Modelos Transductivos (Base 70k OOF)
| Modelo | Arquitectura / Features | OOF Accuracy | OOF LogLoss | OOF ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **CatBoost Depth 8** | Native Text Dictionaries + Categorical Target Stats | **`0.94554`** | **`0.13999`** | **`0.98754`** |
| **LightGBM F117** | 63 Hojas, SAGA Meta-Feature + SVD 64 | `0.92913` | `0.18240` | `0.98040` |
| **XGBoost Depth 7** | Hist Gradient Boosting, SAGA Meta + SVD 64 | `0.89857` | `0.24858` | `0.96326` |
| **SAGA Text Specialist** | 100k TF-IDF Word & Char N-Grams | `0.82650` | `0.38420` | `0.91200` |

### Conclusión Metodológica
CatBoost con procesamiento nativo de texto supera ampliamente a las combinaciones basadas en SVD + GBDT lineal (+4.7% de accuracy sobre XGBoost). XGBoost sirve como un tercer árbol ortogonal que aporta regularización y diversidad de gradientes al ensamble ponderado meta.
