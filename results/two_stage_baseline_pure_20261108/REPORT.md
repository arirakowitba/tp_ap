# Reporte Two-Stage del Baseline Congelado — Seed 20261108 (Pure Seller-Group)

- **Fecha:** 2026-11-07 14:29:52 UTC
- **Pure:** True
- **Seed:** 20261108
- **Tamaños de Corte:** fit=48696 (stage 1 train=41647, val=7049), tune=11339, confirm=9965
- **Iteraciones Congeladas (Stage 1):** {'cb_tabular': 894, 'cb_nested_nb': 868, 'cb_native_text': 888, 'lgb_nested_nb': 622}
- **Duración total:** 46265.5 segundos

## 1. Métricas en Confirm a Umbral Fijo 0.50 (Refit 100% Fit)

| Modelo | Confirm Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 9159 | 0.91912 | 0.21847 | -28 | 182 | 210 | [-74.0, 16.0] | +0 | -28 |
| **cb_nested_nb** | 9187 | 0.92193 | 0.20018 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 9165 | 0.91972 | 0.21366 | -22 | 191 | 213 | [-69.0, 26.0] | +0 | -22 |
| **blend_native_nb** | 9220 | 0.92524 | 0.20176 | +33 | 104 | 71 | [6.0, 59.0] | +0 | +33 |
| **lgb_nested_nb** | 9160 | 0.91922 | 0.20951 | -27 | 141 | 168 | [-64.0, 9.0] | +0 | -27 |
| **blend_cb_lgb80** | 9198 | 0.92303 | 0.19921 | +11 | 35 | 24 | [-6.0, 31.0] | +0 | +11 |

## 2. Métricas en Tune a Umbral Fijo 0.50 (Refit 100% Fit)

| Modelo | Tune Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 10264 | 0.90519 | 0.23807 | -107 | 217 | 324 | [-173.0, -39.0] | +0 | -107 |
| **cb_nested_nb** | 10371 | 0.91463 | 0.22013 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 10321 | 0.91022 | 0.23240 | -50 | 237 | 287 | [-118.0, 17.0] | +0 | -50 |
| **blend_native_nb** | 10396 | 0.91684 | 0.22100 | +25 | 128 | 103 | [-15.0, 66.0] | +0 | +25 |
| **lgb_nested_nb** | 10305 | 0.90881 | 0.23088 | -66 | 154 | 220 | [-111.0, -22.0] | +0 | -66 |
| **blend_cb_lgb80** | 10367 | 0.91428 | 0.21924 | -4 | 27 | 31 | [-20.0, 12.0] | +0 | -4 |

## 3. Comparación Pareada Específica de `blend_native_nb` (50/50) vs `cb_native_text`

- **Confirm:** Net vs Native Text = +55 (Fixes: 142, Breaks: 87), IC 95%: [19.0, 92.0]
- **Tune:** Net vs Native Text = +75 (Fixes: 184, Breaks: 109), IC 95%: [35.0, 117.0]

## 4. Hashes de Reproducibilidad

- `pairs.csv`: `eed1441ba653b291e227338ce226fcbca24be432e43c5573a81ffa155ef5d2d8`
- `tune_pairs.csv`: `ec4659cfa7ad7b5f5f0c5b90e461d24653b0edef3a2d0c19591ac44f834a6058`
- Fit hash: `92b4c1fc391ed56e4604aacb34fe6dff9cb6e826045783dc0d6e3162c1918fb8`
- Stage 1 inner train hash: `1f9cb63c737c0aa804fde3f21b70c08a415f1f771698dbcdd993fb4dd032d449`
- Stage 1 inner val hash: `03113953ec0cf54f13b9967ff7efa5a994eccea5434629ffa021501463da571d`
- Tune hash: `8da17bcb87863171cf0d8048ac782dceb71c2b55bcc72d37b9c5369582fd86c9`
- Confirm hash: `cc02bfa0b72ba9d9975317e4319851282d9dab3d6bbbeab46c57558975185a76`
