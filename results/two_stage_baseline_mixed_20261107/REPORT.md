# Reporte Two-Stage del Baseline Congelado — Seed 20261107 (Mixed Split)

- **Fecha:** 2026-11-07 01:38:41 UTC
- **Pure:** False
- **Seed:** 20261107
- **Tamaños de Corte:** fit=45684 (stage 1 train=38514, val=7170), tune=12198, confirm=12118
- **Iteraciones Congeladas (Stage 1):** {'cb_tabular': 878, 'cb_nested_nb': 817, 'cb_native_text': 896, 'lgb_nested_nb': 458}
- **Duración total:** 1211.1 segundos

## 1. Métricas en Confirm a Umbral Fijo 0.50 (Refit 100% Fit)

| Modelo | Confirm Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 11463 | 0.94595 | 0.13589 | -77 | 98 | 175 | [-179.0, -13.0] | -17 | -60 |
| **cb_nested_nb** | 11540 | 0.95230 | 0.12843 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 11481 | 0.94743 | 0.13447 | -59 | 98 | 157 | [-139.0, -6.0] | -12 | -47 |
| **blend_native_nb** | 11524 | 0.95098 | 0.12858 | -16 | 57 | 73 | [-58.0, 15.0] | +0 | -16 |
| **lgb_nested_nb** | 11437 | 0.94380 | 0.15041 | -103 | 145 | 248 | [-144.0, -61.0] | -77 | -26 |
| **blend_cb_lgb80** | 11540 | 0.95230 | 0.12846 | +0 | 27 | 27 | [-14.0, 15.0] | +2 | -2 |

## 2. Métricas en Tune a Umbral Fijo 0.50 (Refit 100% Fit)

| Modelo | Tune Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 11582 | 0.94950 | 0.13770 | -28 | 109 | 137 | [-67.0, 13.0] | -10 | -18 |
| **cb_nested_nb** | 11610 | 0.95180 | 0.13187 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 11622 | 0.95278 | 0.13458 | +12 | 124 | 112 | [-27.0, 51.0] | +7 | +5 |
| **blend_native_nb** | 11648 | 0.95491 | 0.13050 | +38 | 77 | 39 | [15.0, 62.0] | +14 | +24 |
| **lgb_nested_nb** | 11491 | 0.94204 | 0.15489 | -119 | 135 | 254 | [-165.0, -75.0] | -123 | +4 |
| **blend_cb_lgb80** | 11608 | 0.95163 | 0.13224 | -2 | 30 | 32 | [-17.0, 13.0] | -2 | +0 |

## 3. Comparación Pareada Específica de `blend_native_nb` (50/50) vs `cb_native_text`

- **Confirm:** Net vs Native Text = +43 (Fixes: 84, Breaks: 41), IC 95%: [11.0, 84.0]
- **Tune:** Net vs Native Text = +26 (Fixes: 73, Breaks: 47), IC 95%: [0.0, 52.0]

## 4. Hashes de Reproducibilidad

- `pairs.csv`: `f6a0e47c10845192794b4a632f663c9bcd7640ede3bf4aec52b707268fa539ae`
- `tune_pairs.csv`: `6ea414811f4cc6399d1b26e6788f7b8a7e990b8221d461ce0f1c1bff7845df87`
- Fit hash: `c3504ffbd99511fff0ea2fe92d194e607a9072a15a559fc1cb1fcfa70cbb1b9c`
- Stage 1 inner train hash: `d61f10b5d44a304ba38b22f7bdecfbc7d5a148d4ed89a5f4377809cdd52d9195`
- Stage 1 inner val hash: `bb4b4c1c83f94c2dd0fcc2cb08ac2d4c743dde087f5b35db976f24e5463384bc`
- Tune hash: `e524cc91c212e165c0f40768deb77fee5f6625434bbf771ce895a58799ddd3ce`
- Confirm hash: `dbefd7b0f924587f36a2fac54ac78e3cf771b00c809aead366514e0606fa1d73`
