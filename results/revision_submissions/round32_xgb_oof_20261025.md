# Ronda 32: XGBoost sobre predictores textuales OOF (25 oct 2026)

Se probó una segunda etapa XGBoost con los mismos textos originales, logits word/char fuera de muestra, codificaciones categóricas OOF agrupadas por vendedor y variables profundas del control HistGB. No se ajustaron hiperparámetros mirando los cortes: 900 árboles, profundidad 5, learning rate 0,04, min_child_weight 20, reg_lambda 15, subsample 0,85 y columnas 0,8. HistGB usa umbral 0,475 y XGB 0,5. Una mezcla 50/50 con umbral 0,4875 se congeló tras el primer corte.

| Semilla | Avisos de vendedores nuevos | HistGB | XGB | XGB neto | Mezcla neta | Log loss HistGB/XGB/mezcla |
|---:|---:|---:|---:|---:|---:|---:|
| 883 | 14.335 | 92,878% | 92,871% | −1 | +15 | 0,18387 / 0,18253 / 0,18198 |
| 2026 | 13.493 | 91,825% | 91,781% | −6 | −8 | 0,21335 / 0,21184 / 0,21131 |
| 314159 | 13.898 | 91,797% | 91,740% | −8 | +10 | 0,20981 / 0,20924 / 0,20820 |

XGB mejora consistentemente el log loss, pero no la exactitud. En la clase real `new` gana +34, +44 y +38 aciertos frente a HistGB; en `used` pierde 35, 50 y 46. La mezcla no confirma una ganancia de exactitud en el segundo corte. **Decisión:** conservar como señal probabilística complementaria para investigación; no promover a submission ni entrenar test completo por ahora.

Los tres cortes usan el mismo dataset y se han explorado antes. No se convierten scores públicos agregados en etiquetas individuales. No se generó ni envió una submission. Mejor pública observada: 33=0,94614; líder comunicado por Ariel: 0,95404. El checkpoint reproducible se entregó al usuario por separado.
