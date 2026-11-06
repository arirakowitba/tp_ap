# Auditoría AG-0082/0084 — el OOF 0.94554 no es fold-safe

## HEADs observados

- `main`: `9ab054544e20c540f56965c68b3b969107953c18`
- Antigravity: `60a7f6ca39806d0ea4baf7bd4d0000b9e9ceaf43`
- Claude: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

Se revisaron AG-0082, AG-0083, AG-0084 y los scripts:

- `scripts/train_ultimate_podium_engine_23h.py`
- `scripts/train_pseudo_labeled_transductive_deep_models.py`
- `scripts/generate_purified_podium_portfolio_20261107.py`

No se consumió cuota Kaggle.

## Decisión

El OOF 0.94554 y el portfolio 0.94780–0.94980 quedan **bloqueados**. El score interno no mide generalización seller-group y contiene varias rutas directas de leakage supervisado.

## Fugas verificadas

### 1. Probabilidad de texto in-sample

El script ajusta LogisticRegression sobre todo `X_tr_tfidf, y_train_aug` y luego calcula:

```python
p_text_tr = lr_text.predict_proba(X_tr_tfidf)[:, 1]
```

Esta predicción in-sample se incorpora como feature tanto a LightGBM como a CatBoost. Por lo tanto, cada fila de validación lleva una feature producida por un modelo que ya vio su etiqueta.

La misma fuga existe en `train_ultimate_podium_engine_23h.py`, donde además `p_text_tr` entra directamente al meta-stack.

### 2. Target encoding global con la propia etiqueta

En el motor transductivo:

```python
cat_stats = df_tr_aug.assign(t=y_train_aug).groupby("category_id")["t"].agg(...)
seller_stats = df_tr_aug.assign(t=y_train_aug).groupby("seller_id")["t"].agg(...)
cat_te_tr = df_tr_aug["category_id"].map(cat_smooth)
seller_te_tr = df_tr_aug["seller_id"].map(seller_smooth)
```

Los mapas se calculan una sola vez sobre todas las filas antes de los folds. Las validaciones reciben agregados que contienen sus propias etiquetas y las de sus vendedores.

### 3. Splits por fila, no por vendedor

Ambos scripts usan `StratifiedKFold`. No existe separación seller-group. El mismo vendedor aparece en train y validación, incumpliendo el protocolo y haciendo optimistas las métricas seen/new.

### 4. Pseudoetiquetas no identificadas

El motor añade filas de test con etiquetas producidas por `bayesian_transductive_item_effects_98vec.csv`, solver bloqueado por:

- ledger inconsistente;
- matriz de muy bajo rango;
- backtest prequential peor que cero;
- ausencia de identificabilidad individual.

Las pseudoetiquetas reciben peso 2.0, amplificando ese error.

### 5. Transformaciones ajustadas antes de CV

TF-IDF y SVD se ajustan una vez sobre todo el conjunto aumentado antes de generar folds. Aunque esto es principalmente leakage no supervisado, invalida la afirmación de pipeline estrictamente inductivo y se suma a las fugas supervisadas anteriores.

### 6. Selección y evaluación sobre el mismo OOF

En el engine “ultimate”, los pesos se optimizan sobre el mismo OOF usado para reportar el resultado. No hay fit/tune/confirm separado. El optimizador además fuerza pesos mínimos de 0.05 incluso para componentes que controles anteriores descartaron.

## Por qué 0.94554 no es comparable

La métrica se calcula sobre las primeras 70k filas de un OOF generado después de construir globalmente `p_text_tr`, `cat_te_tr` y `seller_te_tr`. Que la predicción final de CatBoost sea fold-wise no repara features que ya codifican las etiquetas de validación.

Por tanto:

- no es un récord train-only;
- no puede compararse con el baseline depth6+nested-NB honesto;
- no justifica probabilidades por ID;
- no puede promover candidatos Kaggle.

## Calibración del portfolio propuesto

AG espera ganancias casi lineales:

| Corte | Ganancia esperada | Ganancia por flip |
|---:|---:|---:|
| 25 | +13 | 0.520 |
| 45 | +23 | 0.511 |
| 65 | +34 | 0.523 |
| 85 | +44 | 0.518 |
| 105 | +55 | 0.524 |

La evidencia prospectiva más reciente fue:

- slot4: +1 / 7 = 0.143 por flip;
- bloque añadido del slot5: +1 / 15 = 0.067;
- slot5 completo: +2 / 22 = 0.091.

La proyección nueva exige entre 3.6 y 7.8 veces el rendimiento neto por flip observado, sin backtest prospectivo que lo sostenga. El supuesto `P_post > 0.90` reutiliza el solver no identificable y contradice la calibración empírica.

## Reconstrucción mínima exigida

1. Benchmark estrictamente train-only, sin pseudoetiquetas.
2. Fit/tune/confirm por vendedor, congelados.
3. Dentro de cada fold externo:
   - ajustar vocabulario TF-IDF y SVD sólo con outer-train;
   - producir `p_text` de validación con modelo que no vio esas filas;
   - generar features de entrenamiento mediante folds internos seller-group;
   - calcular TE de validación sólo desde outer-train y TE de entrenamiento con esquema nested/ordered.
4. Seleccionar arquitectura, pesos y threshold únicamente en tune.
5. Evaluar una única receta congelada en confirm.
6. Comparar por pares contra depth6 native+nested-NB: accuracy, logloss, fixes/breaks, seen/new y bootstrap vendedor.
7. Sólo después generar full70k y medir desacuerdos contra el ancla pública 0.94719.
8. No usar “458 protegidos”, posterior por ID ni scores esperados hasta corregir CX-0114.

## Estado operativo

El mejor público válido continúa siendo 0.94719. Los candidatos de AG-0082/0084 son drafts no promovibles.
