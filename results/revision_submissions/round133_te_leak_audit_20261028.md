# Ronda 133 — impacto medible de TE global sobre fit externo

Nuevo HEAD Claude `612303c91d73645ccba5aa97504af43a13a64490`, un commit desde `07e2870`. Añade agentes A/C y ampliación B. A: `a_cat_raw_s81` .95159/.12538, mezcla 30% con `ens_ll8` .95197/.12364; A descarta madurez vendedor y CTR más complejo en proxy dos folds. B: al quitar `stk_sellerother` comunica `b_cat_noso` .9524/.1250 y stack con dos CatBoost más XGBoost .9542/.1214. C: red PLE complementaria comunica stack de dos CatBoost sin `sellerother` + red .9536/.1223, con compuertas .9538/.1208. Todos usan la misma CV aleatoria; ninguna cifra equivale aún a confirm seller-group. A y C no existían en HEAD anterior.

Experimento reproducible sobre los 70k train JSONL: para cada fold externo k y cada fila de fit que pertenece a otro fold j, comparar `te_oof` global (entrenado con todos excepto j) contra TE calculado con el mismo entrenamiento **menos k**, manteniendo fija la misma previa global para aislar la dependencia de etiquetas de k. Hay 280.000 comparaciones de fila/pareja de folds (cada fila aparece cuatro veces):

| Feature TE | Cambia al excluir etiquetas de validación externa | Fracción | Diferencia absoluta media |
|---|---:|---:|---:|
| vendedor | 148.416 | 53,0% | .01871 |
| categoría | 229.887 | 82,1% | .02495 |
| vendedor×categoría | 57.568 | 20,6% | .00928 |

Esto demuestra exposición frecuente del fit externo a etiquetas de su fold de evaluación a través de TE global. **No cuantifica la inflación en accuracy ni invalida necesariamente cada predicción individual.** El mismo `frame.py` alimenta las variantes A/B/C; quitar o corregir `stk_sellerother` no corrige este TE. Además, stack de nivel 2 reutiliza OOF de bases que pudieron entrenarse con etiquetas del fold de evaluación del stack. La selección de múltiples configuraciones y early stopping en el mismo OOF añade optimismo. Una comparación limpia debe recomputar toda la tubería por fold externo seller-group, con OOF interno y tune/confirm separados.

57 y 58 igualaron a 48 en score público .94633; se registran solo como bloques agregados. 59–62 siguen sin scores. No se infieren labels de test ni se usa dataset público etiquetado de MLA.

Checkpoint Library `libfile_ae0e71c7916481918738f17eed5b37c9`, SHA256 `d7a1c1b5ae81c672fb30fc7b03941bfeee5e4599c82dad88518b54d2d8777aa4`.
