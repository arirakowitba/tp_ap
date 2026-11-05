# Multi-seed audit: native-text + nested-NB blend

Date UTC: 2026-11-05
Source data SHA256: 3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886
Protocol: train-only seller-group fit/tune/confirm; probabilities fit only; fixed threshold 0.50 reported separately from tune-selected thresholds.

## Paired confirm results at threshold 0.50

| seed | split | n confirm | nested-NB hits | 50/50 native+NB hits | delta blend vs NB | native alone delta vs NB |
|---:|---|---:|---:|---:|---:|---:|
| 20261104 | mixed seen/new | 12624 | 11998 | 12012 | +14 | -20 |
| 20261106 | mixed seen/new | 12177 | 11550 | 11584 | +34 | +22 |
| 20261107 | pure seller-group/new | 10510 | 9559 | 9588 | +29 | -66 |
| 20261108 | mixed seen/new | 12728 | 12064 | 12081 | +17 | -19 |

Mean paired delta blend vs nested-NB: +23.5 confirm hits per split. Sign is positive in 4/4 seeds. Native-text alone is unstable, so the evidence supports probability blending, not replacement by native-text.

Logloss of the blend versus nested-NB improves on seeds 20261106 and 20261108, worsens on 20261104 and pure-group 20261107. Tune-selected thresholds vary materially (.40-.57), therefore no threshold transfer is justified. Fixed .50 is the stable comparison.

## Reproducibility

- seed20261104 pairs SHA256: 9b0ad0cd2a5ec7cffb1dba3b96dba3a4309b3e590131f6c80be0c0338a8af24f
- seed20261106 pairs SHA256: d2ec31a6fc0097498775d4b3bcc60a932b44f64d8f5c849c66b5a95e0a64ca5d
- seed20261107 pairs SHA256: c6ebac4dde79e45e70fac6d2493a7b3991276c8a7351a62282c9bd428b3612ab
- seed20261108 pairs SHA256: 42876b8e0edaa7ca32ca30f98bc9426db24c0c0edb2419f2e4348a4b74a9eb8a

## Limits and decision

These folds overlap across seeds and are development evidence, not four independent holdouts. They do not establish public improvement or justify label inference. However, this is the first model direction in the current cycle with same-sign paired gain over the strong nested-NB baseline across every mixed and pure-group replication.

Next: build full70k bagged probabilities from the exact two components, preserve .50 blend and threshold .50 as predeclared default, compare disagreement IDs/hashes against public anchor, and run a bridge/backtest against historical model submissions before any promotion.
