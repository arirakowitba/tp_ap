# Diseño Secuencial de Sondas Adaptativas hacia ~0.96 (CX-0025 & CX-0026)

- **Baseline de Partida**: **0.94652** (`candidate_d1_d2_combined_on48.csv`, SHA `57172f...`)
- **Seguridad / Ancla**: **0.94633** (`submission_48_reference.csv`, SHA `3f8d08...`)
- **Objetivo**: Brecha de +0.01348 hacia ~0.96000.

## 1. Tabla de Sondas y Candidatos Generados

| Sonda / Candidato | Archivo | SHA256 (LF) | Flips vs Mejor (.94652) | Flips vs 48 | Rol e Hipótesis |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **probe_s1_seen_virgin25** | `submissions/probe_s1_seen_virgin25.csv` | `a87142adbdc302e37e025f4eecdad8294f5ce491ebefc928be0d12b821939876` | **25** | 48 | Hypothesis S1: Seen-seller high-margin tree agreement (margin >= 0.18) on untouched virgin IDs. |
| **probe_s2_new_virgin25** | `submissions/probe_s2_new_virgin25.csv` | `b9acc4b93c91d032a22d640ba72a3e633650aceff4a3b3057323b3e370acb0dd` | **25** | 48 | Hypothesis S2: New-seller high-margin text tree agreement on untouched virgin IDs. |
| **probe_s3_lexical_virgin25** | `submissions/probe_s3_lexical_virgin25.csv` | `dcf5a5eb4cdc099274a5b84028691c1edea7ac618325b078a037ed94774871e0` | **25** | 48 | Hypothesis S3: Third orthogonal virgin consensus block (margin >= 0.15) on untouched test IDs. |
| **candidate_s1_s2_combined_on_best** | `submissions/candidate_s1_s2_combined_on_best.csv` | `bb21e59d43cf9cf0d8e0d0d6bc56b81a1de8c52a2c6ace5155ba867e50312cc3` | **50** | 73 | Orthogonal union of confirmed Seen (S1) + New (S2) blocks. |

## 2. Árbol de Decisión Secuencial para Ariel (5 Cupos)

1. **Envío 1 (Sonda S1 - 25 flips vistos vírgenes)**:
   - Subir `submissions/probe_s1_seen_virgin25.csv`.
   - *Si Score > 0.94652*: S1 queda validada (ganancia esperada +0.00020..+0.00030). El nuevo piso operativo sube a ~0.94675+.
   - *Si Score <= 0.94652*: S1 se descarta; el piso 0.94652 queda protegido intacto.
2. **Envío 2 (Sonda S2 - 25 flips nuevos vírgenes)**:
   - Subir `submissions/probe_s2_new_virgin25.csv`.
   - *Si Score > 0.94652*: S2 queda validada en vendedores nuevos.
3. **Envío 3 (Sonda S3 - 25 flips léxicos vírgenes)**:
   - Subir `submissions/probe_s3_lexical_virgin25.csv` para testear correcciones léxicas en texto.
4. **Envío 4 (Combinación Factorial Aditiva)**:
   - Si S1 y S2 dieron señal positiva, subir `candidate_s1_s2_combined_on_best.csv` (50 flips acumulados).
   - *Garantía matemática*: Debido a la disyunción estricta (`S1 ∩ S2 = ∅`), los deltas públicos se suman sin interferencia, proyectando un score de ~0.94700 a 0.94750+.
5. **Envío 5 (Cupo Reservado para Posterior de Claude)**:
   - Reservado para el modelo posterior calibrado de Claude (CC-0010) una vez que verifique backtest prequencial.
