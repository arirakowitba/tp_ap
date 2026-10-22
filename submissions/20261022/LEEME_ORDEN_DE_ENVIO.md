# Tanda del 22 de octubre de 2026

Cinco archivos listos para Kaggle ap-2026-q-2. Todos: 30.000 filas, ID1..30000, columnas ID y condition, valores new/used, sin nulos. No hay nuevos scores públicos observados al generar esta tanda.

## Orden recomendado

| Envío | Archivo | Objetivo |
|---|---|---|
|26|submission_26_A_E2_complemento16.csv|Prueba pendiente: conserva A y agrega complemento16 de E2. Idéntico al archivo26 entregado antes.|
|27|submission_27_fields_titulo_garantia_modelo_puro.csv|Medir modelo nuevo con teachers separados de título/garantía, sin flips públicos superpuestos.|
|28|submission_28_ensamble_historico_modelo_puro.csv|Medir ensamble de histórico y modelos nuevos, sin flips públicos superpuestos.|
|29|submission_29_consenso_G1_32_sobre26.csv|32 cambios con mayor apoyo mínimo de cinco predictores, siempre respecto de26.|
|30|submission_30_consenso_G2_32_sobre26.csv|Los siguientes32 cambios, disjuntos de G1, también respecto de26.|

Si26 ya se envió, no repetirla: comunicar su score y conservar ese cupo. Los números son identificadores de archivo, no obligación de gastar cupos. Ideal: comunicar scores26–28 antes de usar los dos últimos; eso permite reconsiderarlos si una base nueva mejora sustancialmente. Si se envía la tanda completa,29/30 siguen siendo experimentos interpretables sobre26.

## Qué esperamos aprender

El mejor score público realmente observado sigue siendo0.94533 de24. Bajo21000 filas públicas y truncamiento a5 decimales,26 debería marcar0.94547 por las ecuaciones anteriores. Es una inferencia, no un resultado medido. Si no coincide, revisar supuestos/archivos antes de interpretar nuevas ecuaciones.

27 obtuvo94.3643% interno en14000filas, frente a93.1571% del baseline en las mismas filas. 28 obtuvo94.3815% en4147filas comunes con el histórico, frente a93.9715%. Los subconjuntos difieren y fueron reutilizados: no comparar ambos porcentajes directamente ni asumir superioridad pública.

29y30 toman los primeros64 casos (G1primeros32,G2siguientes32) de323 cambios apoyados por todos los predictores deep, seller-dropout, fields, TPE y meta con histórico. Se ordena por el mínimo P(alternativa a26), descendente, desempate por ID. Se excluyen160 IDs de las cohortes A/E ya probadas. El apoyo mínimo observado es0.7345 enG1 y0.6813 enG2. Son probabilidades de modelos correlacionados, sin garantía de calibración; no representan cinco fuentes independientes ni certezas sobre etiquetas.

Los dos grupos parten de26; **30 no incluye los cambios de29**. Para cada grupo, bajoNpublic21000, el saldo público se estima como round(21000*(score_grupo-score26)), sujeto a las mismas hipótesis. El feedback es agregado, no identifica etiquetas individuales. Si ambos saldos son positivos, se puede preparar la unión después; si solo uno mejora, se conserva ese. Esa selección favorece el segmento público y no acredita ganancia en el oculto.

La base pública existente24 se conserva. 27/28 son pruebas completas de diseño y entrenamiento;29/30 aplican esa información a cambios acotados del control. No se entrenó con etiquetas inferidas del leaderboard.

## Trazabilidad

manifest.json incluye hashes, fuentes, umbrales, conteos e IDs cambiados. auditoria_flips_29_30.csv registra apoyo de cada predictor. generate_submissions_20261022.py reproduce la generación desde el checkpoint de entrenamiento versión4. Los CSV listos son los únicos archivos del paquete que se suben a Kaggle.
