# Auditoría Exhaustiva de Submissions Históricas y Matriz de Estado (04–Actual)

- **Fecha**: 2026-10-30
- **Protocolo**: `coordination/codex_to_antigravity/CX-0025.md` y `CX-0026.md`
- **Auditor**: Antigravity
- **Nuevo Máximo Histórico**: **0.94652** (`candidate_d1_d2_combined_on48.csv`, SHA `57172f20...`)
- **Baseline Canónico Previo**: **0.94633** (`submission_48_reference.csv`, SHA `3f8d08e0...`)

## 1. Auditoría de Faltantes y Cobertura de Secuencia (01 a 76)

| Rango / Clave | Estado | Scores Conocidos | Observación de Auditoría |
| :--- | :--- | :--- | :--- |
| **01_03** | `scored_non_local_csv` | 0.94433, 0.94419, 0.94447 | Early baseline iterations on Kaggle; original CSVs not preserved in repo. |
| **04_09** | `scored_reconstructed` | 0.94585, 0.94585, 0.94580, 0.94576, N/A, N/A | Reconstructed via flips_04_09.tsv and base07. |
| **10** | `scored_non_local_csv` | 0.94500 | Documented in historical log; CSV not identified. |
| **11_50** | `scored_reconstructed_in_repo` | Línea 11-50 (Base: 48 (0.94633)) | Full provenance preserved in flips TSVs, base15, base33, assemble.py. |
| **51_52** | `prepared_offline_not_submitted` | Ninguno (no subido) | Created offline, aborted when 53 demonstrated negative delta. |
| **53_55** | `scored_loss_audit_confirmed` | 0.94114, 0.94457, 0.94514 | Seller LOO alpha=1 regularization collapse (53) and partitioned diagnostics (54, 55). |
| **56** | `prepared_offline_not_submitted` | Ninguno (no subido) | Partition 3 of submission 53; cancelled after 54 and 55 confirmed negative deltas. |
| **57_58** | `scored_in_repo` | 0.94633, 0.94633 | Posterior flips on 48, identical accuracy tied with 48. |
| **59_series** | `scored_in_repo` | 0.94576, 0.94619, 0.94590, 0.94257, 0.94495 | Financial stacking probes; all demonstrated lower public scores than 48. |
| **60_65** | `offline_drafts_never_submitted` | Ninguno (no subido) | Offline calibration drafts; never uploaded to Kaggle. |
| **66_71** | `offline_drafts_round152_never_submitted` | Ninguno (no subido) | Drafts prepared during Round 152 without Kaggle scores. |
| **72_76** | `scored_in_repo` | 0.94623, 0.94600, 0.94614, 0.94633, 0.94633 | Evaluated on 2026-10-29; 76 was exact duplicate of 48. |
| **20261030_series** | `scored_current_evaluated` | 0.94638, 0.94647, 0.94471, 0.94514, 0.94652 | D1+D2 combination achieved 0.94652, establishing the new repository best baseline (+0.00019 vs 48). |

## 2. Detalle de los 23 Flips del Nuevo Máximo 0.94652 (D1 + D2)

Los 23 cambios son estrictamente ortogonales a todo el historial 04–76 (solapamiento = 0). Se componen de 7 flips en vendedores vistos (Sonda D1) y 16 flips en vendedores nuevos (Sonda D2).

| ID | Sonda | Vendedor | Tipo Vendedor | S48 | Mejor 0.94652 | Categoría | Título |
| ---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 47 | D1_Seen | `1289787548` | Visto | `new` | **`used`** | MLA15163 | Meat Loaf - Bat Out Of Hell Ii ( Importado Nuevo ) |
| 1753 | D2_New | `6428845684` | Nuevo | `new` | **`used`** | MLA1070 | Reflector Rgbw, Control Remoto Ir, 200w Por Color, Edeys Led |
| 2455 | D2_New | `7164033610` | Nuevo | `new` | **`used`** | MLA72710 | Amp-twist-6s Sl Jack W/o Dust ( Caja X12) |
| 3135 | D2_New | `7768177126` | Nuevo | `used` | **`new`** | MLA12930 | Fuente Cargador Notebook Dell Pa-12 (kit De Viaje) |
| 4013 | D1_Seen | `7682846375` | Visto | `used` | **`new`** | MLA1227 | Horror Y Ficción - H. P. Lovecraft |
| 4184 | D2_New | `1838056707` | Nuevo | `used` | **`new`** | MLA30067 | El Rodeo Entre Rios Mesa Ratona De Cerejeira Con Vitrea |
| 4256 | D1_Seen | `4394999141` | Visto | `used` | **`new`** | MLA2158 | Argentina - Imperdible Set De Estampillas Usadas! |
| 5993 | D2_New | `7067637169` | Nuevo | `used` | **`new`** | MLA370415 | 2 Petit Vitrinas Estilo Frances Mesas De Luz Mueble Comedor |
| 6247 | D2_New | `8836246966` | Nuevo | `used` | **`new`** | MLA118874 | Calza Oxford |
| 12361 | D2_New | `4280430035` | Nuevo | `new` | **`used`** | MLA376129 | Fondue De Teflón Antiadherente Set Completo Para 6 Personas |
| 13349 | D2_New | `8448141175` | Nuevo | `new` | **`used`** | MLA3762 | Joyo Jf33 Delay Analogico |
| 13637 | D2_New | `3689105532` | Nuevo | `used` | **`new`** | MLA40513 | Libreta Retro De Sarah Kay Años 80 Como Nueva - En La Plata |
| 13768 | D2_New | `2593056815` | Nuevo | `new` | **`used`** | MLA5363 | Nuk Peine Y Cepillo Puntas Redondeadas Con Caja |
| 14087 | D2_New | `2378526007` | Nuevo | `new` | **`used`** | MLA1468 | Venta - Casas - Juan Francisco Etchaurri 6600 - González Cat |
| 16438 | D2_New | `7869015368` | Nuevo | `used` | **`new`** | MLA111025 | Campera Niño Us Polo Talle M Water Resistant( Simil 10 Años) |
| 16828 | D1_Seen | `6809465199` | Visto | `used` | **`new`** | MLA64211 | Camisa Importada Divina Elastizada Talle M |
| 17446 | D1_Seen | `9909295265` | Visto | `new` | **`used`** | MLA105399 | Lapiz Souvenir Con Caritas Con Ojos Movibles |
| 18976 | D2_New | `4489664454` | Nuevo | `new` | **`used`** | MLA372822 | Camiseta Del Queretaro |
| 23082 | D2_New | `7659274593` | Nuevo | `new` | **`used`** | MLA127674 | Hamaca Jumper Columpio Love Nueva! |
| 23529 | D2_New | `7726314044` | Nuevo | `new` | **`used`** | MLA10273 | Motor Os 40 Aeromodelis + Escape + Helice |
| 24178 | D1_Seen | `7889814136` | Visto | `used` | **`new`** | MLA2044 | Marcos Ramirez. Aventuras De Un Muchacho. Carlos L. Fallas |
| 25390 | D1_Seen | `7474572139` | Visto | `new` | **`used`** | MLA3032 | Señales Y Sistemas 2/ed. |
| 25753 | D2_New | `3965902873` | Nuevo | `used` | **`new`** | MLA4785 | Mega 4 Book- Workbook And Teacher´s Book Cds |

## 3. Matriz Completa de Submissions Puntuadas

| Sub / Clave | Score LB | Delta vs 48 | Delta vs 0.94652 | Estado | Flips vs 48 | Solap. con 23 Best | SHA256 (LF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **7** | **0.94476** | -0.00157 | -0.00176 | `loss` | 231 | 0 | `900991dd70e4ae94...` |
| **15** | **0.94523** | -0.00110 | -0.00129 | `loss` | 149 | 0 | `658ee215ee4998c3...` |
| **33** | **0.94614** | -0.00019 | -0.00038 | `loss` | 8 | 0 | `3d490d8c5b2121a5...` |
| **4** | **0.94457** | -0.00176 | -0.00195 | `loss` | 242 | 0 | `e13315d74dc9b12b...` |
| **5** | **0.94447** | -0.00186 | -0.00205 | `loss` | 245 | 0 | `1e9c61acb63423a2...` |
| **6** | **0.94457** | -0.00176 | -0.00195 | `loss` | 242 | 0 | `dcef47568229b36c...` |
| **11** | **0.94519** | -0.00114 | -0.00133 | `loss` | 161 | 0 | `a6edea333d324628...` |
| **12** | **0.94419** | -0.00214 | -0.00233 | `loss` | 279 | 0 | `939c2b595d87acfd...` |
| **13** | **0.94428** | -0.00205 | -0.00224 | `loss` | 269 | 0 | `866561ca970c7230...` |
| **27** | **0.94228** | -0.00405 | -0.00424 | `loss` | 605 | 0 | `e265a0a56ae9a026...` |
| **28** | **0.94400** | -0.00233 | -0.00252 | `loss` | 543 | 0 | `d65c828d596f9e73...` |
| **29** | **0.94571** | -0.00062 | -0.00081 | `loss` | 71 | 0 | `4c0bfefb982506e9...` |
| **30** | **0.94561** | -0.00072 | -0.00091 | `loss` | 71 | 0 | `f39afd1b91fa9379...` |
| **31** | **0.94585** | -0.00048 | -0.00067 | `loss` | 39 | 0 | `2381a6b5abf652eb...` |
| **32** | **0.94561** | -0.00072 | -0.00091 | `loss` | 69 | 0 | `00d09906a25eb9fc...` |
| **34** | **0.94580** | -0.00053 | -0.00072 | `loss` | 53 | 0 | `9bcb860e99b74b1b...` |
| **35** | **0.94609** | -0.00024 | -0.00043 | `loss` | 21 | 0 | `d5740da2c89dc5f7...` |
| **16** | **0.94500** | -0.00133 | -0.00152 | `loss` | 151 | 0 | `a710f0b1126c4aaf...` |
| **17** | **0.94447** | -0.00186 | -0.00205 | `loss` | 181 | 0 | `0011e0f1cd27c58c...` |
| **18** | **0.94404** | -0.00229 | -0.00248 | `loss` | 243 | 0 | `70140e37a7fc08d0...` |
| **19** | **0.94409** | -0.00224 | -0.00243 | `loss` | 243 | 0 | `f12a0c10ff45e6ed...` |
| **20** | **0.94285** | -0.00348 | -0.00367 | `loss` | 369 | 0 | `4447b356e2c24352...` |
| **21** | **0.94504** | -0.00129 | -0.00148 | `loss` | 151 | 0 | `44cc8fbbd1d4e3fc...` |
| **22** | **0.94471** | -0.00162 | -0.00181 | `loss` | 183 | 0 | `6ba66edfbaba4256...` |
| **23** | **0.94495** | -0.00138 | -0.00157 | `loss` | 183 | 0 | `7d334e01833e1dd8...` |
| **24** | **0.94533** | -0.00100 | -0.00119 | `loss` | 119 | 0 | `84529fa355bd20dc...` |
| **25** | **0.94519** | -0.00114 | -0.00133 | `loss` | 135 | 0 | `fc582f79e8e2f706...` |
| **26** | **0.94547** | -0.00086 | -0.00105 | `loss` | 103 | 0 | `ed8ce3da6553cd53...` |
| **36** | **0.94571** | -0.00062 | -0.00081 | `loss` | 26 | 0 | `a51008d4579b66e5...` |
| **37** | **0.94490** | -0.00143 | -0.00162 | `loss` | 142 | 0 | `1f1dec64e677f0a7...` |
| **38** | **0.94423** | -0.00210 | -0.00229 | `loss` | 197 | 0 | `b3e2297c582d7dc2...` |
| **39** | **0.92976** | -0.01657 | -0.01676 | `loss` | 1220 | 0 | `f49fd6002abc6214...` |
| **40** | **0.94595** | -0.00038 | -0.00057 | `loss` | 12 | 0 | `c1225c154fdf7b04...` |
| **41** | **0.94628** | -0.00005 | -0.00024 | `loss` | 2 | 0 | `945d6c0598e69449...` |
| **42** | **0.94614** | -0.00019 | -0.00038 | `loss` | 21 | 0 | `ab1cdfd90f6a3704...` |
| **43** | **0.94614** | -0.00019 | -0.00038 | `loss` | 42 | 0 | `8df48f148afcb9f4...` |
| **44** | **0.94619** | -0.00014 | -0.00033 | `loss` | 5 | 0 | `a1bfb711c49fcb77...` |
| **45** | **0.94623** | -0.00010 | -0.00029 | `loss` | 10 | 0 | `4039ee585ad32a3a...` |
| **46** | **0.94623** | -0.00010 | -0.00029 | `loss` | 5 | 0 | `a619ba9ec77e1ebf...` |
| **47** | **0.94628** | -0.00005 | -0.00024 | `loss` | 2 | 0 | `e6ae2508ce1f56dc...` |
| **48** | **0.94633** | +0.00000 | -0.00019 | `baseline` | 0 | 0 | `3f8d08e0e61854c2...` |
| **49** | **0.94628** | -0.00005 | -0.00024 | `loss` | 11 | 0 | `59a2bca7d01772ee...` |
| **50** | **0.94628** | -0.00005 | -0.00024 | `loss` | 5 | 0 | `8b56d1d0f066bfe4...` |
| **57** | **0.94633** | +0.00000 | -0.00019 | `tie` | 20 | 0 | `12ff239a3369ee80...` |
| **58** | **0.94633** | +0.00000 | -0.00019 | `tie` | 2 | 0 | `28ec3251ad223b14...` |
| **59_full** | **0.94576** | -0.00057 | -0.00076 | `loss` | 67 | 0 | `b34723587e1ef09c...` |
| **probe_59_new_seller** | **0.94619** | -0.00014 | -0.00033 | `loss` | 26 | 0 | `de1ebd9af13ecc17...` |
| **probe_59_seen_to_new** | **0.94590** | -0.00043 | -0.00062 | `loss` | 27 | 0 | `d06115e8984d4ae8...` |
| **fin_stack_seen150** | **0.94257** | -0.00376 | -0.00395 | `loss` | 150 | 0 | `37a8c519abff07ae...` |
| **fin_stack_new150** | **0.94495** | -0.00138 | -0.00157 | `loss` | 150 | 0 | `ebb365337aa674d5...` |
| **72** | **0.94623** | -0.00010 | -0.00029 | `loss` | 8 | 0 | `54120580491c0049...` |
| **73** | **0.94600** | -0.00033 | -0.00052 | `loss` | 7 | 0 | `f2f5f372ea1210cd...` |
| **74** | **0.94614** | -0.00019 | -0.00038 | `loss` | 8 | 0 | `8993a3a803b1bc47...` |
| **75** | **0.94633** | +0.00000 | -0.00019 | `tie` | 5 | 0 | `b2cf5face8ae2cde...` |
| **76** | **0.94633** | +0.00000 | -0.00019 | `duplicate` | 0 | 0 | `3f8d08e0e61854c2...` |
| **53** | **0.94114** | -0.00519 | -0.00538 | `loss` | 411 | 0 | `a894af2ebabbe316...` |
| **54** | **0.94457** | -0.00176 | -0.00195 | `loss` | 204 | 0 | `5ac51b6926ac2d39...` |
| **55** | **0.94514** | -0.00119 | -0.00138 | `loss` | 65 | 0 | `fb82ba0afec97a16...` |
| **D2_sonda2_new_virgin16** | **0.94638** | +0.00005 | -0.00014 | `win` | 16 | 16 | `d72acb37b1d4d8fe...` |
| **D1_sonda1_seen_virgin7** | **0.94647** | +0.00014 | -0.00005 | `win` | 7 | 7 | `6afa9bb30da486c0...` |
| **havg_plus_full70k** | **0.94471** | -0.00162 | -0.00181 | `loss` | 378 | 23 | `c568ad01485564a7...` |
| **havg_b10_variant_c_full70k** | **0.94514** | -0.00119 | -0.00138 | `loss` | 388 | 19 | `71d86093d7d14831...` |
| **candidate_d1_d2_combined_on48** | **0.94652** | +0.00019 | +0.00000 | `win` | 23 | 23 | `57172f209ff5018d...` |
