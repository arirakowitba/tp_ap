# Sensibilidad de umbral en cinco cortes de texto — 2026-11-02

Predicciones pareadas ya guardadas; no se reentrenó ni eligió un umbral en confirm. Para diagnóstico se aplicó el **mismo umbral** a CatBoost tabular y CatBoost+texto en cada corte; los netos en confirm son:

| Seed | t=.45 | t=.50 | t=.55 |
|---|---:|---:|---:|
| 20261109 | +66 | +97 | +104 |
| 20261111 | +27 | +20 | +28 |
| 20261112 | +35 | +42 | +45 |
| 20261113 | +44 | +53 | +49 |
| 20261114 | +52 | +50 | +60 |

En los quince cruces el texto mejora al tabular bajo umbral común. Esto reduce la preocupación de que la ganancia se explique sólo por seleccionar umbrales diferentes en tune; no sustituye una validación independiente ni una comparación fiel con D1+D2. Los cinco cortes comparten 16.525 IDs repetidos y se usaron durante desarrollo. Código: leer `paired_deep_<seed>.csv`, calcular `sum(((p_candidate>=t)==y_true)-((p_base>=t)==y_true))`. No seleccionar `t` con esta tabla para test.
