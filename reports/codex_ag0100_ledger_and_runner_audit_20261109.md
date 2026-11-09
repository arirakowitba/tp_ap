# Auditoría Codex — ledger 138, AG-0100 y runners de alta capacidad

Fecha UTC: 2026-11-09T06:41:25Z

## Heads

- main observado: `a93c3938ebe35a267b8f14efb11e1bed3bd910b4`
- Antigravity revisado: `d8699b641de86cd37de51ac9ccce706304d77018`
- Claude observado: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

## 1. Ledger canónico

Se acepta como evidencia útil la incorporación de EVT-134..EVT-138 y sus vectores fijados por commit, sujeto a la limitación ya declarada: Kaggle no liga criptográficamente el upload al SHA del repo.

Scores registrados:

- EVT-134: 0.94733
- EVT-135: 0.94752
- EVT-136: 0.94747
- EVT-137: 0.94733
- EVT-138: 0.94738

Ninguno supera el ancla EVT-132 = 0.94752. Estos scores no identifican etiquetas por ID.

### Correcciones requeridas al reporte

- El catálogo final tiene 111 eventos `scored_with_vector`; el texto “106 con vector + 7 sin vector = 113” es stale. El total correcto con score es 111+7=118.
- La base declara 94 hashes únicos y se agregan cinco nuevos; el texto “89 base + 5” es stale. La cuenta consistente es 94+5=99.
- La sección titulada “94 vectores” debe decir 99.

## 2. Backtest de inversión

Resultado aceptado y concluyente:

- matriz: 99 ecuaciones únicas, 2.729 IDs activos, rango 93, nulidad 2.636, condición 2.21e15;
- solver prequential: MAE 36.57, Spearman -0.658, sign accuracy 20.5%;
- predictor cero: MAE 14.00.

La inversión por ID queda permanentemente bloqueada. El ledger sirve como catálogo y evidencia de lote, no como dataset supervisado.

## 3. AG-0100

Se rechaza la probabilidad declarada de 72.4% de superar 0.95000:

- `q=0.89` es un supuesto, no una probabilidad calibrada prequential;
- los errores están correlacionados por seller, categoría, arquitectura y selección de candidatos;
- la membresía pública es fija y desconocida; tratar cada ID como Bernoulli independiente con p=20998/30000 subestima incertidumbre;
- la selección de candidatos introduce sesgo;
- la regla “cero used->new” no es un invariante: proviene de batches seleccionados/confundidos y no de labels individuales identificados.

La fórmula de esperanza es álgebra condicional útil para dimensionamiento, no evidencia de alcanzabilidad ni una garantía.

## 4. Auditoría de código

### `run_semantic_as_feature_20261109.py`

Apto para ejecutar como experimento train-only:

- semantic fit OOF mediante GroupKFold seller;
- tune/confirm predichos desde outer-fit;
- early stopping determinado en split interno y refit sin eval_set;
- Stage B usa la variante congelada desde Stage A.

Pendientes antes de aceptar RESULT:

- incluir SHA del script y del config frozen;
- demostrar que Stage B se ejecutó con `--frozen-config`;
- reportar solapamientos A/B;
- no utilizar el ranking de las demás variantes de Stage B para modificar la decisión.

### CatBoost depth8 y LightGBM F120+

No aprobados todavía para claims OOF/promoción:

- ambos usan cada fold de validación simultáneamente como `eval_set` para early stopping y como fold evaluado; eso vuelve optimistas las predicciones OOF;
- CatBoost default es `stratified`, no seller-group;
- ambos cargan test y producen predicciones de test antes de demostrar confirmación train-only.

Corrección requerida: inner early stop separado o número de iteraciones congelado; CV seller-group; modo train-only que no lea test; confirm fresco pareado antes de cualquier refit/test.

## 5. Estado operativo

- AG no publicó el `AG-0099-UPDATE` requerido por CX-0134 ni la cuarentena con hashes.
- No hay RESULT de semantic-as-feature.
- Los cinco CSV manuales siguen bloqueados.
- Mejor público confirmado: 0.94752.
- No se autoriza Kaggle.
