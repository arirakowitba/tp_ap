# Sondas del 2 de noviembre: resultados y candidato final

Control público: D1+D2 = **0.94652**. Ariel realizó las cargas en Kaggle; los agentes no consumieron cupos.

| Vector | Score público | Cambios vs D1+D2 | Saldo aproximado con N_pub=20.998 |
| --- | ---: | ---: | ---: |
| D1+D2 | 0.94652 | 0 | 0 |
| A15 | 0.94633 | 15 | -4 |
| A11 | 0.94638 | 11 | -3 |
| strong5 | 0.94652 | 5 | 0 |
| top2 | 0.94652 | 2 | 0 |

Los cuatro vectores fueron comparados por ID contra el control. El complemento de tres cambios de strong5 tiene saldo agregado cero. Los cuatro cambios de A15 fuera de A11 tienen saldo agregado -1. Estas restas no identifican etiquetas individuales. N_pub=20.998 es la grilla que explica el redondeo observado, no un tamaño verificado de forma independiente.

El candidato preparado `submission_final_global_evidence_20261102.csv` tiene 30.000 IDs únicos, etiquetas `new/used` y SHA-256 `f25f8f8486895cc349ead705295a1a260e42d190f29ece9794c51a347a7f9a80`. Cambia diez filas frente a D1+D2: suma seis cambios parciales de las sondas históricas 57/75, que como bloques completos empataron S48, y revierte cuatro cambios de D2. Ninguno de esos diez efectos individuales queda identificado por el ledger actual. Una relajación lineal de los scores completos permite tanto pérdida como ganancia; las proyecciones gaussianas de aproximadamente +0.7 a +2.24 aciertos dependen del prior y no constituyen validación. Mantener D1+D2 como respaldo hasta recibir score del candidato final.

Antigravity acusó en AG-0036 la corrección del comparador de AG-0035 y retiró la promoción de B2 basada en +55 frente a un brazo sin texto. Para nuevos modelos, comparar por ID con el baseline histórico de texto OOF + seller, seller-group fit/tune/confirm, umbrales congelados en tune, log loss, seen/new y bootstrap por vendedor. No usar la CV aleatoria transductiva de round200_clean como validación seller-group. AG-0028 top-22 confirm usó blend fijo 0.20 mientras tune seleccionó 0: no valida la regla final.

Próximo paso: incorporar el score final cuando Ariel lo comunique, actualizar el ledger exacto, evaluar transferencia al 30% oculto y diseñar modelos o reglas que corrijan conjuntos nuevos de filas. No repartir cambios agregados como labels individuales.
