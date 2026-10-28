# Round55 — fresh-cut CatBoost/FastText consensus audit (2026-10-26)

Reused round54 seed 55555 predictions only; no new fit or test labels. HistGB/FastText decisions use .475; CatBoost uses corrected .5. Rule was specified earlier: where CatBoost and FastText agree against HistGB, take their decision. Fit/tune/confirm have disjoint sellers within this split.

| Segment | Tune n/net vs HistGB | Confirm n/net vs HistGB | Seller bootstrap 95% net, tune / confirm |
|---|---:|---:|---:|
| All consensus against HistGB | 92/+12 | 83/+5 | [−9,+34] / [−16,+25] |
| Used→new subset | 46/0 | 42/+12 | [−15,+16] / [−1,+25] |

The full consensus rule is weak on fresh confirm; neither bootstrap interval excludes zero. The used→new subset was neutral on tune and positive on confirm but its confirm interval still touches a loss. Previous seller cuts overlap this cut, so they are not independent evidence. Do not create a test submission from this rule. The exact public-best submission 41 remains .94628. Round54 checkpoint: Library `libfile_454ba4773eb08191b8bf035f0f2dc844`. No Kaggle quota used.
