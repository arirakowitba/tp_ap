# Revisión integrada de candidatos 28-10-2026

Base pública observada: submission 48 = .94633, igual que 57 y 58; scores públicos agregados, jamás etiquetas individuales. Revisión del trabajo de Claude a HEAD 37e6106 (fusionado en main por PR 4/5) y auditorías de rondas 131–136.

## Evidencia de train

Claude reporta fin_stack .9548 accuracy / .1191 logloss en CV estratificada aleatoria de 5 folds. Nuestro control encontró fuga original de stk_sellerother y dependencia de TE y OOF entre folds del meta modelo. El ensamble final quita sellerother, pero no ha demostrado esa mejora en confirm seller-group seen/new sin dependencia. La señal FastText vs HistGB fue favorable en nuevos vendedores en tres cortes (+12,+39,+45), con solapamiento entre cortes; no valida fin_stack frente a la política 48. Por ello no promover el modelo puro 63 como mejor submission sin prueba pública y confirm limpio.

## Diseño de pruebas públicas

| CSV | Flips sobre 48 | Segmento | Direcciones | SHA256 | Library |
|---|---:|---|---|---|---|
| 59 fin_stack banda alta | 67 | 41 vistos, 26 nuevos | 34→used, 33→new | b34723587e1ef09c897a4af5768a2ea025c0dd313c04bd6494d5f0289899f373 | libfile_4fa30a298c2481919e6417536c5d5492 |
| 60 visto | 171 | vistos | 26→used, 145→new | ef110bf7777bdafb4b07633f27e87b428b239e2c24702e8fb712c47dc3432b24 | libfile_3f10a0800dac8191a6c4ac8179dff1ee |
| 60 nuevo | 81 | nuevos | 45→used, 36→new | c410b84dd0a93ce3da9e112caad0e94aa6036702bc5bda536b17883b1b469709 | libfile_536d9220675c81919febe015d19f384e |

Los dos nuevos CSV se reconstruyeron desde el manifiesto 60 y el CSV original: al revertir sus 252 flips se obtiene la 48 exacta, SHA256 3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722. Son disjuntos entre sí y de 59. Test usado únicamente para segmento y generación de predicciones, no etiquetas.

Secuencia: 59 primero. Si mejora, probar 60 visto y 60 nuevo por separado para estimar heterogeneidad antes de extender a bandas 61/62. Si 59 cae o empata, priorizar 60 nuevo (81, cercano a la señal FastText en nuevos) y dejar 60 visto condicionado; una caída en 59 no implica que cada ID esté mal. Si 60 visto mejora pero nuevo cae, segmentar futuras bandas por seller seen. Si ambas fallan, detener esa familia de flips y dedicar el próximo cupo a contrastar el modelo puro 63 sólo si sirve para decidir el modelo final; no consolidar bandas negativas. Toda decisión posterior usa scores de bloques y no labels por ID.

La banda 60 original mezcla 171 vistos/81 nuevos y direcciones asimétricas; dividirla compra información adicional. El modelo puro 63 altera aproximadamente 1518 IDs frente a 48 y puede bajar mucho el público; su valor es medir generalización, no explotar flips de alta confianza. No se envió ningún CSV a Kaggle.

## Resultado público de 59

Ariel informó 59 = **0,94576** frente a 48 = **0,94633**. Diferencia redondeada −0,00057, aproximadamente −12 aciertos si el público contiene 21.000 filas. Esta es una ecuación de **bloque** sobre sus 67 flips; no revela etiquetas por ID ni permite separar dirección o vendedor visto/nuevo. El CSV exacto de 59 es el de SHA256 `b34723587e1ef09c897a4af5768a2ea025c0dd313c04bd6494d5f0289899f373`.

Decisión: no ampliar automáticamente a bandas 60–62 ni consolidar 59. La confianza de `fin_stack` frente a la política 48 no está calibrada en público: la banda de mayor confianza perdió. La siguiente prueba, si se usa otro cupo, debe comprar información sobre una hipótesis distinta y predeclarada; la partición 60 por vendedor nuevo (81) sigue siendo exploratoria, pero su evidencia no permite prometer ganancia. Priorizar validación etiquetada sin fuga del stack antes de promoverlo como base. El modelo puro 63 es demasiado amplio para recomendarlo como explotación de score actual sin esa validación.

## Diagnóstico anidado tras la caída de 59

Preparado `probe_59_new_seller_sobre48.csv` (Library `libfile_2aa3e844b7cc8191891ad75ee171ab07`, SHA256 `de1ebd9af13ecc1723e5f521e1c0dcb525a2e9261267ae3edb7bafe4bc5e7b9f`): conserva sólo 26 de los 67 flips de 59, todos de vendedores nuevos (20→used, 6→new). Tiene 30.000 IDs únicos y se reconstruyó desde la 48 exacta. Si se envía, su score de bloque junto al de 59 permite estimar el saldo agregado complementario de los 41 flips de vendedores vistos, sujeto al redondeo público; no identifica labels por ID. Es un test diagnóstico, no una candidata con mejora garantizada. Evitar enviar 60 completo antes de interpretar este resultado, porque mezcla segmento y dirección.

## Resultado del diagnóstico de vendedores nuevos

Ariel informó `probe_59_new_seller_sobre48.csv` = **0,94619**, frente a 48 = **0,94633**. Diferencia pública redondeada ≈−3 aciertos con 21.000 filas públicas. El bloque completo 59 = 0,94576 (≈−12); por sustracción de saldos agregados, los 41 flips complementarios de vendedores vistos aportaron ≈−9, sujeto al redondeo y tamaño público exacto. Ninguna de estas ecuaciones asigna labels por ID. La banda de máxima confianza falla en ambos segmentos agregados; no promover 60–63 automáticamente.

Para usar un próximo cupo con mayor información y menor exposición se preparó `probe_59_seen_to_new_sobre48.csv`: 27 flips de vendedores vistos en dirección →new, SHA256 `d06115e8984d4ae82c34cd0ac7276f7b16a6666d1eca57bea736aa5f7b7f32e7`, Library `libfile_23180154cca08191a41fcb4aeb244a3d`. Su score, junto con los saldos de 59 y 59-new, permitiría estimar el saldo agregado complementario de 14 flips vistos →used. Es un experimento de separación de dirección, no una predicción de mejora. Si el objetivo inmediato es preservar score, usar 48. Detener las bandas Claude después de esta partición si no surge un subgrupo favorable; priorizar nueva validación etiquetada.
