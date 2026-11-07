# Reporte de Replicación del Baseline Congelado — Seed 20261108 (Pure Seller-Group)

- **Fecha:** 2026-11-07 01:14:18 UTC
- **Pure:** True
- **Seed:** 20261108
- **Tamaños de Corte:** fit=48696 (inner_train=41647, inner_val=7049), tune=11339, confirm=9965
- **Duración total:** 665.0 segundos

## 1. Métricas en Confirm a Umbral Fijo 0.50

| Modelo | Confirm Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 9144 | 0.91761 | 0.22398 | -45 | 179 | 224 | [-93.0, 2.0] | +0 | -45 |
| **cb_nested_nb** | 9189 | 0.92213 | 0.20458 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 9154 | 0.91862 | 0.21727 | -35 | 174 | 209 | [-87.0, 18.0] | +0 | -35 |
| **blend_native_nb** | 9201 | 0.92333 | 0.20537 | +12 | 95 | 83 | [-21.0, 44.0] | +0 | +12 |
| **lgb_nested_nb** | 9126 | 0.91581 | 0.21579 | -63 | 131 | 194 | [-105.0, -24.0] | +0 | -63 |
| **blend_cb_lgb80** | 9187 | 0.92193 | 0.20391 | -2 | 29 | 31 | [-18.0, 14.0] | +0 | -2 |

## 2. Métricas en Tune a Umbral Fijo 0.50

| Modelo | Tune Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **cb_tabular** | 10257 | 0.90458 | 0.24054 | -71 | 240 | 311 | [-132.0, -12.0] | +0 | -71 |
| **cb_nested_nb** | 10328 | 0.91084 | 0.22114 | +0 | 0 | 0 | [0.0, 0.0] | +0 | +0 |
| **cb_native_text** | 10281 | 0.90669 | 0.23926 | -47 | 245 | 292 | [-112.0, 17.0] | +0 | -47 |
| **blend_native_nb** | 10356 | 0.91331 | 0.22486 | +28 | 140 | 112 | [-8.0, 66.0] | +0 | +28 |
| **lgb_nested_nb** | 10298 | 0.90819 | 0.23420 | -30 | 179 | 209 | [-78.0, 19.0] | +0 | -30 |
| **blend_cb_lgb80** | 10355 | 0.91322 | 0.22054 | +27 | 57 | 30 | [8.0, 46.0] | +0 | +27 |

## 3. Comparación Pareada Específica de `blend_native_nb` (50/50) vs `cb_native_text`

- **Confirm:** Net vs Native Text = +47 (Fixes: 126, Breaks: 79), IC 95%: [12.0, 84.0]
- **Tune:** Net vs Native Text = +75 (Fixes: 180, Breaks: 105), IC 95%: [33.0, 120.0]

## 4. Hashes de Reproducibilidad

- `pairs.csv`: `47327924bc08bb4eeafc108477075acb171a9d91f6a368b46677073bd008b1ae`
- `tune_pairs.csv`: `ee092f0a5a3ad19bc47ec6a03f4c7463a0a2534a3953a805d82cc8030cd46c35`
- `inner_val_pairs.csv`: `4787b231290b38fd93e2494d54094d742569a89e7cf11834ec2afddf2078d2ff`
- Fit hash: `92b4c1fc391ed56e4604aacb34fe6dff9cb6e826045783dc0d6e3162c1918fb8`
- Inner train hash: `1f9cb63c737c0aa804fde3f21b70c08a415f1f771698dbcdd993fb4dd032d449`
- Inner val hash: `03113953ec0cf54f13b9967ff7efa5a994eccea5434629ffa021501463da571d`
- Tune hash: `8da17bcb87863171cf0d8048ac782dceb71c2b55bcc72d37b9c5369582fd86c9`
- Confirm hash: `cc02bfa0b72ba9d9975317e4319851282d9dab3d6bbbeab46c57558975185a76`
