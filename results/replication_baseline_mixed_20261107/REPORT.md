# Reporte de Replicación del Baseline Congelado — Seed 20261107 (Mixed Split)

- **Fecha:** 2026-11-07 00:50:03 UTC
- **Pure:** False
- **Seed:** 20261107
- **Tamaños de Corte:** fit=45684 (inner_train=38514, inner_val=7170), tune=12198, confirm=12118
- **Duración total:** 1451.8 segundos

## 1. Métricas en Confirm a Umbral Fijo 0.50

| Modelo | Confirm Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 11403 | 0.94100 | 0.14607 | -75 | 123 | 198 | [-171.0, -10.0] | -27 | -48 |
| **cb_nested_nb** | 11478 | 0.94719 | 0.13618 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 11456 | 0.94537 | 0.14383 | -22 | 128 | 150 | [-91.0, 29.0] | -11 | -11 |
| **blend_native_nb** | 11482 | 0.94752 | 0.13717 | +4 | 72 | 68 | [-28.0, 32.0] | -3 | +7 |
| **lgb_nested_nb** | 11391 | 0.94001 | 0.15697 | -87 | 153 | 240 | [-130.0, -45.0] | -65 | -22 |
| **blend_cb_lgb80** | 11492 | 0.94834 | 0.13611 | +14 | 40 | 26 | [-3.0, 32.0] | +11 | +3 |

## 2. Métricas en Tune a Umbral Fijo 0.50

| Modelo | Tune Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 11501 | 0.94286 | 0.15156 | -50 | 120 | 170 | [-90.0, -9.0] | -37 | -13 |
| **cb_nested_nb** | 11551 | 0.94696 | 0.14369 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 11529 | 0.94515 | 0.14702 | -22 | 128 | 150 | [-60.0, 18.0] | -13 | -9 |
| **blend_native_nb** | 11571 | 0.94860 | 0.14243 | +20 | 76 | 56 | [-7.0, 50.0] | -3 | +23 |
| **lgb_nested_nb** | 11420 | 0.93622 | 0.16605 | -131 | 139 | 270 | [-177.0, -85.0] | -129 | -2 |
| **blend_cb_lgb80** | 11555 | 0.94729 | 0.14402 | +4 | 31 | 27 | [-12.0, 20.0] | -4 | +8 |

## 3. Comparación Pareada Específica de `blend_native_nb` (50/50) vs `cb_native_text`

- **Confirm:** Net vs Native Text = +26 (Fixes: 82, Breaks: 56), IC 95%: [-8.0, 72.0]
- **Tune:** Net vs Native Text = +42 (Fixes: 94, Breaks: 52), IC 95%: [18.0, 68.0]

## 4. Hashes de Reproducibilidad

- `pairs.csv`: `92cf32006997afde268bf368030e61a7f2c3effad53fa0138000a1b10644574b`
- `tune_pairs.csv`: `4091b3c469eb35aad0164eed7cf9e3166e26378044966a8cc15cdd9c2fcd95dc`
- `inner_val_pairs.csv`: `440fa1febeff3dd1fe57d49476f0f2589a4ffdf57e24b2b35c6c45b9e87a808a`
- Fit hash: `c3504ffbd99511fff0ea2fe92d194e607a9072a15a559fc1cb1fcfa70cbb1b9c`
- Inner train hash: `d61f10b5d44a304ba38b22f7bdecfbc7d5a148d4ed89a5f4377809cdd52d9195`
- Inner val hash: `bb4b4c1c83f94c2dd0fcc2cb08ac2d4c743dde087f5b35db976f24e5463384bc`
- Tune hash: `e524cc91c212e165c0f40768deb77fee5f6625434bbf771ce895a58799ddd3ce`
- Confirm hash: `dbefd7b0f924587f36a2fac54ac78e3cf771b00c809aead366514e0606fa1d73`
