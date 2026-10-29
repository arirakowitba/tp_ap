# NEXT STEPS & RESUME PROTOCOL

**Fecha y Hora**: 2026-10-29 09:15 ART  
**Branch de trabajo**: `antigravity/seller-group-validation-20261029`  
**Estado del repositorio**: Árbol limpio, sin procesos en segundo plano activos.

---

## 1. Resumen Ejecutivo de la Sesión

1. **Ronda 142 (`seller_group_experiments_20261029.py`)**:
   - Validación externa estricta por grupo de vendedor (fit ~45k, tune ~12k, confirm ~13k) sobre dos semillas (2026 y 314159).
   - CatBoost base demostró ser el mejor modelo individual (~0.908–0.910 confirm).
   - La compuerta de historial vendedor colapsó a +10 (IC `[-10, +29]`, cruzando cero).
   - El ensamble lineal con umbral perdió frente a CatBoost puro (-16 net en 2026, -5 en 314159).
2. **Ronda 143 (`enriched_features_experiment_20261029.py`)**:
   - Prueba de 15 expresiones regulares de condición léxica ("como nuevo", "repuesto", etc.) y z-scores de precio.
   - **Resultado Negativo**: CatBoost enriquecido perdió -64 aciertos netos en semilla 2026 (IC `[-139, -2]`). Las reglas binarias rígidas crean cortes de árbol de alta varianza que degradan probabilidades de texto suaves.
3. **Ronda 144 (`nbsvm_catboost_experiment_20261029.py`)**:
   - Prueba de representación dual de texto: Sparse TF-IDF + NB-SVM (razón de log-conteos OOF en 4 folds por vendedor).
   - Mejoró frente a su baseline sin NB-SVM: +24 net en semilla 2026 (log loss 0.2311 $\to$ 0.2266) y +30 net en semilla 314159 (log loss 0.2447 $\to$ 0.2363). Concentrado en vendedores nuevos (+25 y +34).
4. **Ronda 147 (`audit_dual_text_vs_strong_baselines_20261029.py`)**:
   - Auditoría directa contra los **Modelos Fuertes** previos sobre exactamente los mismos IDs de confirm:
     - Frente a HistGB: +61 (2026) y +123 (314159). Superación categórica.
     - Frente a LightGBM: +47 (2026) y +102 (314159). Superación categórica.
     - Frente a CatBoost_r142: **+2 net** en semilla 2026 (IC 95%: `[-52.0, +48.0]`, empate estadístico) y **+46 net** en semilla 314159 (IC 95%: `[+6.0, +89.0]`).
5. **Ronda 145 (`full_train_dual_text_20261029.py`)**:
   - Reentrenamiento con las 70k etiquetas de train (5-fold GroupKFold OOF por vendedor).
   - Predicciones para los 30.000 ítems del test (`test_data.jsonlines`).
   - Contraste por ID contra Submission 48 (.94633):
     - Modelo Puro (`submission_64`): 1.774 flips vs 48 (1.188 en vistos). SHA256: `e96d204f6e88b09780437a91903d151bdca53f85448377a192e329351dc4927a`.
     - Modelo Conservador Nuevos Vendedores (`submission_65`): 141 flips vs 48 (todos en nuevos, margen $|p-0.5| \ge 0.25$, 0 en vistos). SHA256: `33e03d6ffe4d983c70e5b18159b66648b15df21108dd1a487290ecea13ead867`.

---

## 2. Recomendación para la Decisión Final de Kaggle (29-Octubre Noche)

- **Recomendación Principal**: **Mantener la Submission 48 (.94633) como la predicción definitiva seleccionada.**
- **Justificación**:
  - En la semilla 2026, el intervalo bootstrap del CatBoost Dual-Text frente al CatBoost fuerte incluye el cero por amplio margen (`[-52.0, +48.0]`).
  - Las 5 sondas públicas de `fin_stack` demostraron que los flips agresivos (especialmente en vendedores vistos, donde se perdieron 150 aciertos y el score cayó a .94257) destruyen la ventaja de la 48.
  - La Submission 48 es la referencia empírica más sólida disponible.
- **Alternativa Condicional (Sólo si Ariel decide arriesgar el último cupo en un bloque exploratorio)**:
  - Usar `submissions/20261029/submission_65_dual_catboost_new_sellers_top_margin.csv`.
  - Deja intactos los 21.487 registros de vendedores vistos (0 riesgo en vistos) y modifica únicamente 141 registros de vendedores no vistos donde NB-SVM y CatBoost tienen alta certidumbre combinada.

---

## 3. Comando Exacto para Reanudar la Sesión

Para inspeccionar o retomar la investigación:

```bash
git checkout antigravity/seller-group-validation-20261029
python src/revision_submissions/audit_dual_text_vs_strong_baselines_20261029.py
```
