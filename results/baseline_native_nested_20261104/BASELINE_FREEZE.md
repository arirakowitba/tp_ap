# BASELINE_FREEZE — native-text + nested-NB 50/50

Estado: baseline interno de desarrollo, congelado para replicación  
Fecha: 2026-11-06  
Código fuente: `scripts/train_codex_catboost_native_nested_20261104.py`

## Receta exacta

```text
p_blend_native_nb =
    0.50 * p_cb_native_text
  + 0.50 * p_cb_nested_nb

label = 1[p_blend_native_nb >= 0.50]
```

No confundir con:

```text
p_blend_cb_lgb80 =
    0.80 * p_cb_nested_nb
  + 0.20 * p_lgb_nested_nb
```

## Split congelado

- seed: 20261104
- pure: false
- fit: 44.834
- tune: 12.542
- confirm: 12.624
- fit hash: `0b15e2c6f97fe810fbdd4616274f813b19986c015a49dc38ba2b253467b1025d`
- tune hash: `d0bcf08902d3517f5f7d6c0543f7bc03fb12204faf6dd644734a542fbfcc4a2c`
- confirm hash: `ced0e88f1d8c26cc21e66f948c5c1a5a45cd0a47d37e9f508fb7ea3cc8288cd2`
- tune CSV Git blob: `a1dd349f98094375f645dc8330cb11c6e6538594`
- confirm CSV Git blob: `034e5779acefd2c2c0b7b560c2d8a50505f94263`
- confirm pairs SHA-256 declarado por runner: `4968442d9434649ef5c2a381865838e661f3534f312734d39e9ec92d0db1822d`

## Métricas a umbral fijo 0.50

| Modelo | Tune hits | Tune logloss | Confirm hits | Confirm logloss |
|---|---:|---:|---:|---:|
| cb_nested_nb | 11.893 | 0.13885 | 12.010 | 0.13030 |
| cb_native_text | 11.893 | 0.14051 | 11.978 | 0.13940 |
| blend_native_nb 50/50 | **11.915** | **0.13653** | **12.024** | 0.13221 |
| lgb_nested_nb | 11.785 | 0.15876 | 11.920 | 0.14808 |
| blend_cb_lgb80 80/20 | 11.909 | 0.13776 | 12.016 | **0.12942** |

El 50/50 gana +22 hits sobre cada componente en tune y +14 sobre nested-NB en confirm. Su IC bootstrap vendedor contra nested-NB en este split fue [-5,+35], por lo que una sola confirmación no basta.

## Diagnóstico de pesos

Sweep diagnóstico con umbral fijo 0.50, seleccionado sólo por tune:

| Peso native | Tune hits | Confirm hits |
|---:|---:|---:|
| 0.45 | 11.911 | 12.025 |
| **0.50 congelado** | 11.915 | **12.024** |
| 0.55 (mejor tune) | **11.920** | 12.021 |
| 0.60 | 11.918 | 12.005 |
| 0.65 | 11.916 | 11.997 |

El óptimo de peso no transporta de tune a confirm. Por eso se congela 0.50 por predeclaración y estabilidad conceptual, no se adopta 0.45 por su resultado confirm.

## Limitaciones

- Split de desarrollo reutilizado; no es confirmación externa.
- CatBoost usa tune como eval_set para early stopping.
- No demuestra transporte al test público.
- No autoriza submission.
- La próxima evaluación debe usar receta, umbral y pesos sin cambios en splits nuevos.
