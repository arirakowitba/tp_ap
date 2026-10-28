# Corrector residual regularizado de texto y categoría

Modelo nuevo: regresión logística regularizada C=.05 sobre dos predictores word/char OOF, 40 dimensiones FastText, log precio, categoría prefijo y tipo de aviso, entrenado únicamente en fit (~41k filas). Su logit corrige al promedio de profesores de texto; el residual se suma con peso .05/.1/.2 al logit HistGB+FastText. Peso .05 escogido por +4 netos en primer tune antes de confirmar. Todo seller-group, sin test labels ni scores públicos.

| Split | Tune neto peso .05 | Confirm neto peso .05 | Log loss confirm base→corrector |
|---|---:|---:|---|
| 39002 | +4 (16/12) | −10 (7/17) | .210628→.211159 |
| 39011 | −4 (7/11) | −4 (11/15) | .205913→.206004 |

Pesos .1 y .2 tampoco ganaron en confirm de ninguno de los splits. Descarta corrector: empeora accuracy y log loss frente a HistGB+FastText; no se necesita gastar cómputo CatBoost para promoverlo. Los splits habían sido usados antes. Código, pares y probabilidades en Library libfile_bf51ae5bc5e881919ea7c8d9a0f07e93. No Kaggle.
