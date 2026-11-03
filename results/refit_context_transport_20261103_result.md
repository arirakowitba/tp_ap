# Unlabeled feature context transport audit (2026-11-03)

This audit tests whether extracting the 101 structural features from 70k labeled train rows versus 70k train plus 30k unlabeled test rows changes a strong CatBoost model on fresh seller-group splits. Test labels and public scores were not used. The comparison is paired within each split; text TF-IDF OOF and target encodings were identical in the two arms. Both arms were fit on the same fit rows; early stopping and accuracy threshold were selected on tune only. Confirm was evaluated once per predeclared seed. Code: `scripts/audit_refit_context_transport_20261103.py`.

Input ZIP SHA-256: `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. `used=1`.

| Seed | Fit / tune / confirm | 70k context accuracy | 100k context accuracy | Paired net 100k minus 70k | Fix / break | 70k / 100k log loss | Seller bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261130 | 45,232 / 12,632 / 12,136 | .952126 | .950313 | -22 | 54 / 76 | .125994 / .127140 | [-49, +3] |
| 20261131 | 44,726 / 12,424 / 12,850 | .951984 | .952451 | +6 | 86 / 80 | .122937 / .123012 | [-30, +42] |

The confirm sets overlap in 2,829 IDs. Deduplicating 24,986 rows yields 22,157 unique IDs. Keeping the first seed's prediction on overlap: net -22, bootstrap seller CI [-62, +18]; keeping the second: net -19, CI [-61, +24]. A common .500 threshold instead of the independently tuned thresholds yields -15 and -5 by seed; deduplicated -22 or -18. These uncertainty intervals include zero, so this is a transport risk, not a confirmed predictive gain for either extraction context.

Thirteen of 101 feature columns changed for training rows when the 30k unlabeled test rows entered extraction. The largest effects are `freq_seller_id` (43,626 rows), `freq_category_id` (62,702), `freq_city` (68,285), `seller_n_cats` (35,361), seller mean features, and `rel_price_cat` (42,345). The CatBoost best iterations shifted 892 to 632 and 893 to 713. This validates that the two pipelines are materially different. The 100k refit context must not be assumed to inherit the 70k validation estimate.

The present two arms both compute unsupervised features using all train rows, including confirm rows. This is a transductive validation design, not a strict inductive one. A further test should fit frequency, seller summary and category price aggregation maps on fit rows only, transform tune/confirm, and train on full70k maps for final test; compare that strictly inductive pipeline with historical and bag3 baselines on fresh seller-group splits. Do not promote a public CSV from this audit alone.

Private paired rows include `item_id`, seller, label, both probabilities and predictions. SHA-256: seed 20261130 `2c9b835a872298f71166c9686edfedc18136c216dfba76a35508f86f3a6d9734`; seed 20261131 `a9d24362e7220bdd98190b725df21efc24cabc79a8b2fe21a393f0053ed9d1a5`. JSON reports have the 13-column change counts and input hashes. The paired files are kept outside the public repository.
