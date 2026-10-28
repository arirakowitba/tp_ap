# Estrategia de submissions hasta el cierre

**Referencia:** 33 = 0.94614 público. Meta intermedia exigente: 0.954; aspiración: 0.960. Si la evaluación pública usa 21.000 filas fijas y muestra cinco decimales truncados, las diferencias equivalen aproximadamente a **+165** y **+291 aciertos netos** respecto de 33. La hipótesis de 21.000 no está demostrada; estos son órdenes de magnitud, no recuentos de etiquetas conocidos.

## Diagnóstico de las 41 numeradas

El inventario y matrices de predicciones están en `inventario_01_41.md` y los TSV. Los ensayos 36–40 muestran que extender consenso y blend de modelos contra 33 perjudicó el score público. El modelo de probabilidad por fila tampoco anticipó el signo de 37–39: estimó +116,6 / +150,8 / +433,2 ganancias en test completo, mientras los scores públicos bajaron −0.00124 / −0.00191 / −0.01638. Esto invalida seleccionar cientos de cambios solo por margen o consenso. No invalida todas las familias de modelos en otros segmentos, ni identifica etiquetas individuales.

35 hizo 13 cambios con saldo público casi neutro (−0.00005): 8 de vendedores nuevos (4 hacia used y 4 hacia new) y 5 de vendedores vistos (4 hacia used y 1 hacia new). 36/40 separan un bloque de 4 y otro de 14, ambos con delta negativo agregado. 37/38 separan 146 y otros 55 vistos, también con delta negativo. La 41 aísla 6 vendedores nuevos de 37; aún no tiene score.

## Dos escalas de mejora

1. **Pruebas pequeñas que compran información:** no más de un bloque causal por submission; base 33 fija, cambios exactos y hashes antes del score. Después de 41, aislar un bloque de 35 por segmento solo si tune/confirm ofrece una hipótesis clara. Comparar diferencias anidadas o disjuntas y registrar efectos de grupos, nunca verdades por ID. No gastar cupos en variantes casi duplicadas.
2. **Nueva base para la mejora grande:** +165 o +291 netos no es plausible a partir de los pocos bloques de 4–18 flips. Diseñar modelos/familias con ganancia generalizada frente a HistGB/CatBoost y un proxy reproducible de 33. Priorizar vendedores nuevos sin sacrificar vistos; evaluar predicciones pareadas y fixes/breaks en cortes tune/confirm por vendedor, título, categoría y margen. Evitar reabrir SVD, ExtraTrees, atributos aislados, fechas de imágenes, fuzzy title, calibración simple y CatBoost/LightGBM fijo ya ensayados sin una hipótesis nueva. Una candidata grande solo se justifica si gana consistentemente en confirm no seleccionado y en segmentos relevantes; estimar incertidumbre del neto con bootstrap agrupado por vendedor. El tercer corte de ronda 13 fue reutilizado y no cuenta como confirm independiente.

## Secuencia de decisiones

- **Ahora:** esperar el score de 41; registrar score y efecto del bloque de 6. Si gana, auditar la regla contra etiquetas de train antes de ampliarla. Si pierde, descartar la ampliación por esa regla, no las seis filas como etiquetas conocidas.
- **Siguiente microprueba:** el bloque visto/nuevo de 35 solo tras definir por adelantado cuál se prueba y comprobar pares en tune/confirm. El total de 35 fue −0.00005, por lo que una partición puede revelar una mejora de grupo, sin implicar aciertos individuales.
- **Cada ciclo de modelos:** entrenar con folds separados por vendedor para el objetivo de transferencia, reservar confirm, comparar fixes/roturas con baselines comunes, congelar regla y CSV, luego usar como máximo una submission pública para corroborar transferencia. Si falla, diagnosticar shift y selección de desacuerdos; no compensar con ajustes por ID al score.
- **Al acercarse al cierre:** elegir el mejor score público observado entre candidatas que también tengan soporte de train, documentar el riesgo sobre el 30% oculto y mantener una submission final reproducible. No hay garantía de 0.954 o 0.960.

## Registro obligatorio al llegar un score

Anotar hora, archivo y SHA256, ID/dirección/segmento de cada cambio, base exacta, score observado (o pendiente), diferencia vs base, intersecciones con pruebas previas, métricas tune/confirm y decisión. Conservar los scores de todos los experimentos anteriores incluso cuando su CSV no está disponible; no rellenar los cuatro CSV faltantes ni scores 26–30 por inferencia.

**Regla de evidencia:** las probabilidades por fila provienen de etiquetas de train y validación; los scores públicos actualizan la credibilidad de una regla o bloque en conjunto. Una diferencia de leaderboard no es una etiqueta individual. La meta del 96% exige mejoras reales de generalización y no una identificación encubierta de test.
