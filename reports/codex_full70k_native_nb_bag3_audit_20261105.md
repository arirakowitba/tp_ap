# Full70k bag3 native-text + nested-NB audit

Date UTC: 2026-11-05
Protocol source: CX-0112
Data SHA256: 3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886
Test labels used: false
Kaggle submitted: false

## Construction

- Exact two components validated in four seller-group replications.
- CatBoost depth6, 900 iterations, seeds 20261104/20261106/20261108.
- Nested text probability is fit-only OOF on train and full-train prediction on test.
- Per family, average three seeds; final probability = .50 nested-NB bag + .50 native-text bag.
- Frozen classification threshold .50.

Artifacts:
- probabilities SHA256 62597ecefe3600752f06d940f81524e6baae18486197d2d1c39cda1990b86811
- candidate SHA256 ef203e3896a6d9abce898644622fa753ad38d73faedaf946e28ae38a89ada10a
- checkpoint ZIP SHA256 269c8605609cba680efc6dec10265e96cf9555346af33296966957861e909357

## Stability

Threshold-.50 Hamming distances on 30k:

| pair | differences |
|---|---:|
| seed04 vs seed06 | 175 |
| seed04 vs seed08 | 166 |
| seed06 vs seed08 | 169 |
| bag3 vs seed04 | 98 |
| bag3 vs seed06 | 103 |
| bag3 vs seed08 | 94 |

Bag3 predicts 14158 used. Probability margin |p-.5| quantiles: q01=.03453, q025=.08448, q05=.16639, q10=.28815. Bagging substantially stabilizes seed variance.

## Bridge to identified public vectors

| reference | known public score | disagreements with bag3 |
|---|---:|---:|
| top60 current best | .94709 | 542 |
| D1+D2 | .94652 | 588 |
| H-AVG plus full70k | .94471 | 447 |
| H-AVG B10+variant-c | .94514 | 427 |
| OptionA bag3 | .93695 | 922 |
| cycle1 multimodal | not independently promoted | 490 |

The candidate is closer to two known underperforming H-AVG model submissions than to top60. This is negative transport evidence. The internal four-seed confirm gain does not overcome that bridge risk.

## Decision

Do not promote the full model candidate or spend Kaggle quota. Keep probabilities as a component for a later high-confidence hybrid only after a prequential bridge/backtest demonstrates that its disagreement ranking adds value over top60. Do not infer individual labels from the aggregate public scores.

The useful result is methodological: 50/50 native+NB is robust in train-only seller-group validation, but full70k transport remains unproven and resembles earlier model families that underperformed publicly.
