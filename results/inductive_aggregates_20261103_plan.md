# Fit-only aggregation validation plan (2026-11-03)

Predeclared before training. Use the intact original 70k train labels (`used=1`) and 30k test rows without labels. Run seller-group seeds 20261132 and 20261133, each with disjoint fit/tune/confirm rows. Common strong text word+char TF-IDF OOF and target encodings are computed only inside fit. Compare the same CatBoost specification under two structural-feature contexts:

1. Historical transductive extraction from all 70k train rows, including tune/confirm features but never their labels.
2. Strictly inductive maps for frequencies, seller summary means, seller category count, and category median price, computed from fit rows and applied to tune/confirm. Other structural features stay unchanged.

Each arm selects early stopping and accuracy threshold from tune only. Evaluate confirm once. Report paired accuracy, log loss, fixes/breaks, seen/new, seller bootstrap, change counts and SHA-256 of inputs and paired CSVs. Deduplicate overlapping confirm IDs when pooling seeds. CI or gain on one split does not justify a public submission. If fit-only is consistently better, next compare against historical strong/bag3 on another fresh split and audit final full70k map construction; if not, treat as a validation-transport diagnostic.

Code: `scripts/audit_refit_context_transport_20261103.py` with `AP_CONTEXT_MODE=fit`. The script names the alternate arm `ctx100` internally for compatibility with its 70k-versus-100k mode; in this plan and results that arm means **fit-only**.
