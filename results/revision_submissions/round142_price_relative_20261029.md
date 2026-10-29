# Ronda 142: precio relativo y categorías nativas (29-10-2026)

Fuente: Sheet original `Gemini Kaggle AP full compact 100k`, export íntegro de 24.589.999 bytes. Train de 70.000 filas; partición histórica de ronda 13, fit 56.000 y valid 14.000, alineación `row_id` y `y=used` verificada contra NPZ de CatBoost. Se calculó la mediana de log precio por categoría y tipo de publicación **sólo en fit**. HistGradientBoosting (380 iteraciones, 22 hojas, tasa .05) con las 37 variables tabulares del compacto, categorías codificadas por claves sin target, y dos diferencias de precio. Sin texto; esto mide una señal tabular complementaria, no un reemplazo equiparable.

| Modelo | Accuracy | Log loss | Seen | New |
|---|---:|---:|---:|---:|
| Ronda 13 CatBoost fuerte | .944214 | .147403 | .962673 | .902603 |
| Ronda 142 tabular | .873214 | .293938 | .892555 | .829614 |

Frente a CatBoost: 250 fixes y 1.244 breaks, neto −994. La señal tabular sola no alcanza y no justifica una submission. El corte es histórico y ya explorado; no se usa como confirm virgen ni para inferir etiquetas públicas. La siguiente hipótesis tendría que sumar una señal nueva a la base fuerte, con OOF interno sin fuga y confirm seller-group distinto.

Checkpoint con código, JSON y pares por ID: Library `libfile_25f10a33f3808191bbe82306a4df44e3`, SHA256 `eb0f9666131ba81935775b5f1afa26888f606434528f2414845a85455971317d`. No se envió a Kaggle.
