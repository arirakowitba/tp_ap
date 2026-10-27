# Rondas 105–111 — suavizado débil del historial de vendedor (27-10-2026)

Tras la mejora de las rondas 90–104, comparé en cortes pareados la misma arquitectura de vendedor LOO con suavizado 1, 20 y 50. Solo se usan etiquetas reales de fit y los textos originales; cada fila de entrenamiento excluye su propia etiqueta al construir el historial del vendedor. La compuerta conserva HistGB+FastText para vendedores nuevos. Cada corte aleatorio tiene 42k fit, 14k tune y 14k confirm; el temporal usa 42k antiguos, 14k siguientes y 14k recientes.

| Corte | Suavizado 1, neto tune/confirm vistos vs base | Suavizado 20, neto tune/confirm vistos vs base | Delta confirm 1−20 |
|---|---:|---:|---:|
| 20261042/43 | +163 / +143 | +155 / +127 | +16, bootstrap vendedor [−2,+34] |
| 20261044/45 | +159 / +170 | +157 / +157 | +13, bootstrap [−6,+32] |
| Temporal `date_created` | +129 / +123 | +108 / +113 | +10, bootstrap [−8,+30] |

En el primer corte, suavizado 50 obtuvo +156/+138. El suavizado 1 quedó delante de 20 en los tres tune y los tres confirm, pero **cada intervalo de diferencia en confirm incluye cero** y los cortes se solapan. El beneficio principal, frente a HistGB+FastText, es grande y repetido; la elección precisa de suavizado aún es incierta.

Reentrené suavizado 1 con las 70k etiquetas y generé la **candidata 53** sobre el CSV exacto de 48: 411 cambios entre vendedores vistos, ninguno entre nuevos; 250 cambios hacia used. SHA-256 `a894af2ebabbe3166caf984842813b1a47d0504b181a5800198655e2b15813a8`, Library `libfile_2071a29b9e7481919a588c058125db96`. Difiere de la candidata 51 (suavizado 20) en 210 IDs y de la 52 (promedio 5/20) en 183. Estos son desacuerdos sin etiquetas ni predicción confiable de score público. El checkpoint con código, pares, comparación directa, predicción de test y hashes es `libfile_03ce1102b9348191b47000e003de53ef`, SHA-256 `636337cd38ef8d6b8b25151d528cd8e838f8928125cf27989b693fac0c39aeb1`.

**Decisión provisional:** si la cuenta y la cuota Kaggle están disponibles, usar primero la 53 para evaluar la variante de mayor neto en tune y confirm; mantener 51/52 como alternativas condicionales al score de 53. No enviar las tres juntas. No se subió ninguna desde este entorno: el navegador disponible muestra Kaggle sin sesión iniciada. El mejor público comunicado sigue 48=.94633, sin inferir labels por ID.