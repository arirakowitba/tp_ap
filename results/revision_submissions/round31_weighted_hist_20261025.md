# Ronda 31: HistGB ponderado por frecuencia de vendedor (25 oct 2026)

**Resultado:** descartado para test/submission. Se aplicó únicamente a la segunda etapa HistGB un peso de entrenamiento proporcional a `n_avisos_del_vendedor^-0.35`, con media normalizada a 1. Los predictores word/char OOF, las variables, los folds por vendedor, los tres cortes y el umbral 0,475 quedaron iguales a ronda 30.

| Semilla | Validación | Sin peso | Ponderado | Arreglos / roturas | Neto | Log loss sin/con peso |
|---:|---:|---:|---:|---:|---:|---:|
| 883 | 14.335 | 92,878% | 92,940% | 78 / 69 | +9 | 0,18387 / 0,18350 |
| 2026 | 13.493 | 91,825% | 91,907% | 93 / 82 | +11 | 0,21335 / 0,20929 |
| 314159 | 13.898 | 91,797% | 91,668% | 83 / 101 | −18 | 0,20981 / 0,21060 |

Por clase real, el tercer corte perdió 34 aciertos en `new` y ganó 16 en `used`. Para vendedores con un solo aviso, los saldos fueron +10, 0 y 0. Los intervalos bootstrap agrupados por vendedor incluyen cero en los tres cortes. El reemplazo de HistGB por su variante ponderada dentro de la mezcla 50/50 con CatBoost fue exploratorio y dio +7, +9, −5; tampoco estable.

El checkpoint reproducible se entregó al usuario por separado. No se entrenó la variante completa para test ni se creó o subió una submission. Los cortes comparten el dataset y se han explorado antes; no hay validación pública de esta hipótesis. La submission 33 sigue siendo la mejor pública observada: 0,94614; el líder comunicado por Ariel está en 0,95404. Ningún score agregado se usó como etiqueta por ID.
