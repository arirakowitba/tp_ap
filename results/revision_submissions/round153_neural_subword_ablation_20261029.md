# Ronda 153: Ablación de Representación Neuronal de Subwords (SVD + MLP) en Deep CatBoost

**Fecha**: 2026-10-29  
**Branch**: `antigravity/deep-nbsvm-strong-validation-20261029`  
**Dataset**: 70k train records (`train_data.jsonlines`), particiones seller-group externas honestas (fit ~45k, tune ~12k, confirm ~13k).  
**Protocolo**: Umbral de decisión ajustado en `tune` y congelado antes de evaluar en `confirm`.

---

## 1. Motivación y Arquitectura

Para explorar la segunda familia complementaria solicitada (modelos neuronales / subword), evaluamos si proyectar n-gramas de subword (caracteres de 3 a 6) en un espacio continuo latente denso (TruncatedSVD, 64 dimensiones) y entrenar un Perceptrón Multicapa (MLP: 64 $\to$ 32 $\to$ 1, activación ReLU, optimizador Adam, $\alpha=10^{-3}$) aporta una probabilidad OOF (`txt_mlp_p`) que capture semántica no lineal no capturada por los modelos lineales de n-gramas.

Todo el proceso de SVD y MLP fue ajustado estrictamente dentro de `fit` mediante 4-fold GroupKFold por vendedor para evitar cualquier fuga entre grupos.

---

## 2. Resultados en Holdout Confirm Virgen

| Semilla | Arquitectura | Confirm Acc | Log Loss | Fixes / Breaks | Neto | IC 95% Bootstrap Vendedor | Seen Net | New Net |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026** | Deep CatBoost Base | 0.92391 | 0.18764 | — | — | — | — | — |
| **2026** | **Neural Deep CatBoost** | **0.92515** | 0.19095 | 103 / 87 | **+16** | **[-45.0, +58.0]** | -2 | +18 |
| **314159** | Deep CatBoost Base | 0.92019 | 0.19558 | — | — | — | — | — |
| **314159** | **Neural Deep CatBoost** | 0.91941 | 0.19749 | 83 / 93 | **-10** | **[-35.0, +17.0]** | -5 | -5 |

---

## 3. Diagnóstico Estadístico y Cierre de Hipótesis

1. **Rendimiento intrínseco inferior de la red**: La red neuronal sobre embeddings SVD alcanzó un accuracy OOF aislado de ~74.4% (log loss 0.518), frente al 81.3% (log loss 0.416) del modelo lineal sparse TF-IDF.
2. **Impacto en CatBoost**:
   - En la semilla 2026 obtuvo +16 aciertos netos, pero el IC 95% bootstrap (`[-45, +58]`) cruza ampliamente el cero.
   - En la semilla 314159 **empeoró en -10 aciertos netos** (83 fixes vs. 93 breaks), perdiendo tanto en vendedores vistos (-5) como en nuevos (-5).
   - El log loss empeoró en ambas semillas (+0.00331 en 2026 y +0.00191 en 314159).
3. **Veredicto**: La representación neuronal densa no aporta información ortogonal útil frente a los 101 features profundos y degrada la calibración probabilística. **Se descarta formalmente su adopción.**
