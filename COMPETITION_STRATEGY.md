# Memoria de trabajo de la competencia — Análisis Predictivo

Preferencia explícita de Ariel, registrada el 18 de octubre de 2026 (Buenos Aires).
Vigencia: hasta el cierre de esta competencia.

## Objetivo y enfoque

Mejorar iterativamente la submission completa, tanto en su porción pública como en la oculta. Aprovechar estratégicamente todas las submissions disponibles hasta el cierre para obtener información, además de buscar mejoras inmediatas de score. Cada ronda debe integrar la experiencia y la evidencia de las anteriores.

El usuario concibe los resultados de las pruebas de flips como una fuente adicional de evidencia agregada para aprender sobre el dataset, actualizar probabilidades y descubrir relaciones que los modelos entrenados todavía no captan. Evaluar este enfoque por su diseño y sus resultados; no reducirlo automáticamente a la búsqueda del score público.

## Frentes que deben trabajar juntos

- Seguir entrenando: mejorar arquitecturas existentes, probar modelos nuevos, ajustar hiperparámetros y construir ensembles que aprovechen errores complementarios.
- Mejorar los datos: limpieza, representaciones, ingeniería de variables, encodings y señales de texto u otras modalidades pertinentes.
- Investigar técnicas nuevas cuando puedan aportar algo que todavía no se haya probado.
- Incorporar a posteriori los resultados públicos de pruebas anteriores para actualizar la confianza en grupos y casos dudosos.
- Diseñar probes que equilibren ganancia esperada e información nueva: casos nuevos, divisiones de grupos, reversiones y pruebas de hipótesis comparables.
- Usar las conclusiones de los probes para orientar tratamientos, entrenamiento y correcciones que puedan beneficiar también a filas de la parte oculta.

## Cómo convertir resultados en evidencia

Conservar la trazabilidad entre el control, los IDs modificados, las etiquetas originales y nuevas, los modelos que respaldan cada cambio y el score observado. Registrar el solapamiento entre probes y las hipótesis sobre tamaño público y redondeo; analizar conjuntamente las restricciones y su dependencia.

Un cambio de score informa sobre el resultado agregado de los casos públicos afectados. No identifica por sí solo cuáles de todas las filas cambiadas son públicas ni cuáles tienen la etiqueta correcta. Mantener separadas las etiquetas verificadas, las restricciones agregadas, las probabilidades posteriores y las hipótesis de extrapolación.

El propósito es mejorar también la parte oculta; una mejora pública no demuestra por sí sola que eso haya ocurrido. Contrastar las relaciones aprendidas mediante validación por grupos, predicciones fuera de muestra, pruebas controladas y estabilidad entre particiones cuando corresponda. Registrar la reutilización del holdout para interpretar su evidencia correctamente.

## Ciclo de trabajo

1. Recuperar modelos, predicciones, auditorías y resultados de todas las rondas anteriores.
2. Incorporar los nuevos scores sin perder las relaciones entre pruebas.
3. Actualizar el diagnóstico de errores y la confianza posterior.
4. Entrenar y comparar nuevos modelos y tratamientos guiados por ese diagnóstico.
5. Diseñar las siguientes submissions para rendimiento, confirmación o información, explicando qué permite concluir cada resultado.
6. Guardar archivos reproducibles y actualizar el historial antes de iniciar la siguiente ronda.

Mantener la mejor referencia comprobada, conservar las etiquetas verificadas y distinguir resultados medidos de estimaciones. Distribuir el presupuesto de submissions según lo aprendido y el tiempo restante; recuperar la fecha efectiva de cierre antes de planificar el calendario.


## Aclaracion del objetivo competitivo (20 de octubre de 2026)

El usuario define este trabajo como **reward hacking del score final de la competencia**: aprovechar todas las submissions disponibles para ganar dentro de las reglas aplicables. No se exige que la prediccion resultante coincida con el modelo que se desplegaria en produccion. Se permite disenar pruebas de flips por fuerza bruta con tamanos y cruces calculados, contrastar arquitecturas nuevas e investigar informacion externa del mismo dataset cuando las reglas lo permitan.

La parte publica representa el 70 % del test segun informo el usuario. Trabajar con 21.000 filas publicas y 9.000 ocultas entre 30.000 es una **hipotesis de calculo** que se revisa si contradice los resultados. Un score solo impone restricciones agregadas sobre las filas afectadas. Analizar conjuntamente cohortes que reaparecen en varias pruebas; no declarar etiquetas individuales. El beneficio oculto se extrapola con incertidumbre y se contrasta con CV agrupada y senales independientes.

El calculo iterativo reproducible vive en la seccion 13 del `notebooks/23_revision_submissions_y_flips.ipynb`, generado por `scripts/build_submission_review_notebook.py` y `scripts/iterative_notebook_section.py`; la inferencia esta en `src/ap2026q2_iterative_submissions.py`. Actualizar `results/revision_submissions/feedback_iterativo_20261020.json` con cada score. La submission 15 obtuvo 0,94523. Ya se recibieron 16:0,94500; 17:0,94447; 18:0,94404; 19:0,94409; 20:0,94285. Sus saldos de cohortes son A:+2, B:-7, C:-18, D:-17, E:-15 bajo 21.000 filas publicas fijas y truncamiento. Solo A suma, con confianza de transferencia aun limitada; 15+A infiere 0,94533 publico, no medido. E1-E4 e IDs individuales siguen sin resolver. Priorizar la validacion de A con casos nuevos y contrastes que separen E; conservar modelos complementarios, evaluando su politica de flips en el subconjunto dudoso. Guardar en el repositorio CSV exactos, hashes, auditorias, patrones, resultados y origen del control. Ver `results/revision_submissions/resultados_cruzados_16_a_20.md`.

## Generalizacion y meta hasta el cierre

Actualizacion explicita del usuario posterior a los resultados 16-20. La meta
es maximizar el resultado al cierre, con horizonte aproximado de 15 dias y
una aspiracion de **95-96 % de accuracy publica**, cuidando tambien el 30 %
oculto. No es un resultado ya alcanzado ni una garantia de que esos niveles
sean posibles en esta particion. Bajo el denominador publico de 21.000 filas,
pasar de la 15 al 95 % requiere 100 aciertos netos adicionales; al 96 %, 310.
La candidata no enviada 15+A reduce esas brechas a 98 y 308.

**No es necesario probar todos los flips ni identificar cada caso dudoso.**
Se pueden corregir casos nunca incluidos en un probe mediante relaciones
aprendidas de modelos fuertes, predicciones fuera de fold, similitud entre
casos, distribuciones y evidencia agregada de pruebas anteriores. Registrar
esas correcciones como inferencias con incertidumbre y auditar su origen;
no convertirlas en etiquetas verdaderas para entrenamiento supervisado.

Cada ronda debe combinar entrenamiento y ensambles, tratamientos de datos,
calibracion y umbrales por segmento, analisis de distribuciones y feedback
posterior de los probes. Evaluar nuevas relaciones en folds agrupados y en
los subconjuntos donde se aplicaran, incluyendo clase, categoria, vendedor,
margen y tipo de desacuerdo. Incorporar la evidencia agregada a la politica
de seleccion y a sus confianzas, y comprobar su extension a IDs distintos.
Una familia que falla como regla de flips en un segmento puede seguir
aportando a un ensamble general: medir ambos usos.

Elegir los probes por su valor para la decision final: cuanto pueden cambiar
el ranking de predicciones y la confianza transferible a casos no probados,
ademas de la mejora de score esperada. Evitar objetivos de cobertura total
o fuerza bruta por si mismos. Mantener exploracion de relaciones nuevas y
validacion de hallazgos anteriores junto con consolidaciones progresivas.

La proxima tanda prevista tiene cinco envios: cuatro pruebas adaptativas
que combinen validacion, nueva informacion y extensiones inferidas; la
quinta se decide **despues** de recibir los cuatro scores. Los tres cruces
propuestos para separar E1-E4 son una opcion, no una agenda obligatoria:
reemplazarlos si otra prueba o modelo tiene mayor valor esperado para el
cierre. Conservar el codigo, registros, auditorias y resultados en el repo.

## Objetivo actualizado: 96 % global con margen y confianza

El objetivo explicito posterior del usuario es **96 % sobre las 30.000 filas
completas**, con margen y alta confianza, usando todas las tecnicas anteriores.
La identidad es A_global = 0.7*A_publica + 0.3*A_oculta. Con 95 % publico y
un oculto hipotetico de 94,523 %, el global seria 94,8569 %. Con ese oculto
fijo se necesita aproximadamente 96,633 % publico para llegar a 96 % global.
La score 15 contiene ajustes con feedback: no mide por si sola la calidad
oculta de un modelo sin ajustes. La igualdad entre segmentos es una hipotesis.

Meta operativa propuesta: **97 % publico y 95 % oculto central = 96,4 % global**.
Exigir una distribucion predictiva conjunta validada con probabilidad al menos
95 % de superar 28.800 aciertos globales; buscar ademas un limite predictivo
inferior oculto de 94 %, que con 97 % publico da 96,1 % global. Estas son
metas y criterios, no resultados obtenidos ni confianza ya acreditada.

Auditoria importante: los 70.000 grupos usados en la CV historica son todos
distintos. Sus folds no separaban vendedores, aunque asi se describieron
antes. Las metricas OOF se conservan; corregir su interpretacion. El 71,6233 %
del test pertenece a vendedores vistos, frente a aproximadamente 67,4 % en
la validacion actual. Preparar protocolos para vendedores vistos y nuevos,
ponderarlos segun el test y simular un publico 70 % con un 30 % sellado para
validar la transferencia del procedimiento adaptativo completo.
El ensamble actual obtuvo 96,1157 % OOF en vendedores vistos y 88,3605 %
en nuevos. Priorizar representaciones, datos y politicas que mejoren este
ultimo segmento sin perjudicar el primero. Cada punto porcentual ganado
en nuevos equivaldria a unos 85 aciertos del test si transfiriera. Un
escenario objetivo de 98 % en vistos y 92,5 % en nuevos daria 96,4393 %
global con la mezcla actual; sigue siendo una meta por validar.

Priorizar mejoras de base, datos, ensambles y politicas de correccion por
segmento, actualizadas con feedback agregado y aplicables a filas no probadas.
Los probes deben descubrir y confirmar reglas utiles para muchas predicciones.
No alcanzar confianza aparente suponiendo independencia entre errores ni
convertir inferencias en etiquetas verdaderas. El detalle y las cuentas
reproducibles estan en `results/global_target_96/README.md`, el modulo
`src/ap2026q2_global_objective.py` y la seccion 13.6 del notebook de revision.
