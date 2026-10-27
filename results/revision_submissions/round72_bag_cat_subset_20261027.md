# Ronda 72 — ensamble HistGB frente a CatBoost en los mismos desacuerdos

Se cruzaron los pares del ensamble de ronda66 con CatBoost de ronda50 por ID de fila, verificando identidad exacta de las probabilidades FastText y que todas las filas estaban en confirm. Se usan umbrales históricos: .475 para HistGB/FastText y .5 para CatBoost. El subconjunto es de desacuerdos seleccionado en ronda50, no un holdout representativo ni nuevo.

| Corte | Filas | Ensamble vs CatBoost neto (IC vendedor) | Ensamble vs FastText neto (IC vendedor) | Log loss ensamble / CatBoost / FastText |
|---|---:|---:|---:|---:|
| a | 254 | −8 [−31,+14] | −1 [−14,+12] | .6782 / .6780 / .6749 |
| b | 293 | −2 [−30,+24] | −7 [−25,+10] | .6735 / .7023 / .6733 |

La mejora global de log loss de ronda66 no se traduce en mejor decisión en la región de desacuerdos. Tampoco CatBoost domina de modo consistente a FastText allí (+7/−5 netos). No preparar submission. Los dos cortes se solapan y ya habían sido usados; el bootstrap agrupa vendedores pero no elimina esa dependencia. Scores públicos solo informan efectos de bloques, jamás labels por fila.

Checkpoint `round72_bag_cat_subset_checkpoint_20261027.zip`, Library ID `libfile_bb93ed3841d4819195b959616b68d021`, SHA256 `a9d95b5c28ee474606a3740c3a9b3990a5f162597fc39d01f7e05900f5819b4b`. Contiene código, pares, métricas y hashes; depende de checkpoints de rondas50 y66.
