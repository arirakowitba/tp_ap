# Revisión consolidada de submissions

Abrir [el notebook 23](../../notebooks/23_revision_submissions_y_flips.ipynb).
Está ejecutado y pensado para exponer desde cero. Incluye ejemplos didácticos
separados de los datos reales, gráficos, ecuaciones, comprobaciones y un guion
breve de presentación.

## Corte y resultados

Corte del 18 de octubre de 2026, anterior al score de la submission 10:

- 23 scores observados y 4 archivos preparados pendientes.
- Mejor público observado: 94,476%.
- Control combinado: 94,485% inferido bajo dos modelos de display; no medido.
- Holdout histórico de referencia: 94,3952%, reutilizado.
- 2.531 casos dudosos sin etiqueta verificada: 768 ya aparecen en pruebas
  puntuadas y 1.763 todavía no.
- 64 casos nuevos en la submission 10, divididos en cuatro bloques de 16.
- Plan adaptable de 75 pruebas; sólo el primer CSV está preparado.

El notebook explica la diferencia entre score de modelo, suma pública
identificada, posterior de cohorte, transferencia y proxy de máxima entropía.
Ninguna de estas últimas cantidades se presenta como una etiqueta oculta
individualmente observada.

## Archivos

- snapshot_20261018.json: snapshot ligero para ejecutar el notebook sin datos
  originales, modelos pesados ni entrenamiento. Incluye hashes y los IDs
  efectivamente cambiados por archivo.
- submission_feedback_log.csv: registro de scores y notas originales.
- snapshot_manifest.json: hash y controles del snapshot.
- execution_report.json: resultado de la ejecución completa del notebook.
- ../../src/ap2026q2_submission_review.py: cálculos de display, posterior Beta
  con máscara pública, predictivas e intervalos enteros.
- ../../scripts/build_submission_review_snapshot.py: reconstruye el snapshot
  desde la carpeta de experimentos que conserva las auditorías completas.
- ../../scripts/build_submission_review_notebook.py: fuente editable de la
  narrativa, código y estructura del notebook.
- ../../scripts/execute_submission_review.py: ejecuta desde un proceso limpio.
- ../../tests/test_submission_review.py: invariantes estadísticos y del diseño.

## Recalcular

Desde la raíz del repo:

    python -m pip install -r requirements_submission_review.txt
    python scripts/execute_submission_review.py
    python -m unittest discover -s tests -p test_submission_review.py -v

En entornos que bloquean los sockets locales de Jupyter, usar
`python scripts/execute_submission_review.py --in-process`. Este modo ejecuta
todas las celdas secuencialmente mediante IPython en un proceso nuevo y conserva
las tablas y figuras en el notebook. El informe de ejecución identifica el modo.

No se consumen submissions y no se entrenan modelos al ejecutar este informe.

Para incorporar nuevos resultados, primero actualizar el registro y las
auditorías en el entorno de experimentos. Luego regenerar el snapshot:

    python scripts/build_submission_review_snapshot.py --experiment-root RUTA_EXPERIMENTOS --repo-root .

El generador exige los CSV de cada archivo, controles, auditorías y resultados
del experimento. No inventa datos faltantes ni descarga datos privados. El
snapshot de Git es suficiente para leer y ejecutar el notebook; los archivos
masivos de entrenamiento no se duplican.

Si cambia el estado de una submission, también actualizar la fecha de corte,
las afirmaciones de estado y los controles del notebook antes de versionarlo.
Las celdas finales y los tests detectan cambios en los conteos del corte.

## Convenciones para continuar

1. Versionar el archivo exacto o su auditoría de IDs, hash, control y score.
2. Mantener separados resultados observados, inferidos y pendientes.
3. Conservar solapamientos en el sistema de restricciones.
4. Actualizar los límites y la confianza sin asignar etiquetas por un score
   agregado.
5. Replanificar las pruebas restantes y contrastar transferencia.
6. Validar una variante antes de reemplazar la referencia global.

