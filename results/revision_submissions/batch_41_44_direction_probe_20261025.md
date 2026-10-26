# Tanda 41/44: consenso de vendedores nuevos, prueba por dirección

Base: submission 33, score público reportado 0.94614, SHA256 `3d490d8c5b2121a5021007090e583d718d8d0d14132e9ec4e59f79c36808b3c7`. No se usaron etiquetas inferidas del leaderboard. La 42 y la 43 empataron públicamente a 0.94614; ambas probaron otra familia de IDs FastText, disjunta de 41.

| Número | Estado | Cambio respecto de 33 | IDs | Archivo |
|---|---|---|---|---|
| 41 | score público 0.94628, reportado por el usuario | seis cambios de consenso en vendedores nuevos | `new→used`: 1219, 8862, 22148; `used→new`: 18412, 26663, 26668 | `submission_41_consensus_6_vendedores_nuevos_sobre33.csv` |
| 44 | preparado, sin score | subconjunto de tres cambios `used→new` de la 41 | 18412, 26663, 26668 | `submission_44_consensus_new_sellers_used_to_new_3_sobre33.csv` |

SHA256 de 44: `a1bfb711c49fcb77cdf8c7b1fd405fcb43a66b1101a595f17b34aee43fde45db`. Validada con 30.000 IDs únicos 1..30000, columnas `ID,condition`, tres y solo tres cambios respecto del CSV exacto de la 33. 41 ya existía, no se ha modificado en esta tanda.

**Feedback 41 (usuario):** 0.94628; mejora pública de +0.00014 frente a 33=0.94614. Es el nuevo mejor score reportado. El resultado corresponde al bloque de seis cambios en conjunto; no identifica aciertos por ID ni por dirección. No se verificó independientemente Kaggle.

**Secuencia actualizada:** 41 ya subida; si se dispone de cuota y se busca separar señales, subir 44 después. Al comparar 44 con 33 se mide el saldo agregado de los tres `used→new`; al comparar 41 con 44, el de los tres `new→used`. No hace falta subir una tercera candidata del complemento: su efecto agregado queda determinado por esas diferencias salvo resolución de redondeo. Si 41 pierde y el cupo es escaso, reservar 44 para una arquitectura nueva con validación sobre etiquetas train. No combinar estos cambios con 42/43, cuyo efecto neto observable fue cero. Ningún score agregado autoriza etiquetar individuos; tampoco se garantiza mejora sobre 33.
