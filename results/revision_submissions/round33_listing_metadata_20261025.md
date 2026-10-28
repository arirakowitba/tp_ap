# Ronda 33: identidad de tags, pagos y estado del aviso (25 oct 2026)

Se evaluaron señales conjuntas de publicación presentes en el JSON original: indicadores de tags y métodos de pago no MercadoPago (vocabulario por frecuencia aprendido solo en fit), estado/subestado, presencia de tienda oficial/producto de catálogo y cantidad de promociones. HistGB, textos originales OOF, codificaciones por vendedor, folds, hiperparámetros y umbral 0,475 quedaron iguales al control de ronda 30.

| Semilla | Validación por vendedores ausentes | Control | Metadatos | Arreglos / roturas | Neto | Log loss control/nuevo |
|---:|---:|---:|---:|---:|---:|---:|
| 883 | 14.335 | 92,878% | 92,836% | 32 / 38 | −6 | 0,18387 / 0,18419 |
| 2026 | 13.493 | 91,825% | 91,729% | 36 / 49 | −13 | 0,21335 / 0,21315 |
| 314159 | 13.898 | 91,797% | 91,812% | 46 / 44 | +2 | 0,20981 / 0,21003 |

Los intervalos bootstrap por vendedor incluyen cero en los tres cortes. Tampoco aparece una mejora consistente por clase real ni por vendedores con un solo aviso. **Decisión:** descartar esta ampliación; no entrenar versión de test ni gastar submissions. El checkpoint reproducible se entregó al usuario por separado.

Los cortes reutilizan el mismo dataset, y ninguna diferencia de score público se trató como etiqueta por ID. Mejor pública observada: 33=0,94614; líder comunicado por Ariel: 0,95404. La hipótesis surgió al revisar soluciones públicas para el mismo tipo de datos, pero el ensayo local la contradice en estos cortes.
