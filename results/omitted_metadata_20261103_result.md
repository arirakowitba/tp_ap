# Omitted listing metadata ablation (2026-11-03)

The predeclared plan is `results/omitted_metadata_20261103_plan.md`; code is `scripts/ablate_omitted_metadata_20261103.py`. The candidate adds original-row `status` (categorical), `has_description`, `n_descriptions`, and `secure_thumb_missing` to the historical strong CatBoost baseline with word+char TF-IDF OOF and seller categorical. There are no public-score or test-label features. Both arms use identical seller-group fit/tune/confirm, supervised TE and text OOF. Early stopping and threshold are selected on tune, then confirm is evaluated once per seed. Input ZIP SHA-256: `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`.

| Seed | Confirm N | Historical / candidate accuracy | Net | Fixes / breaks | Historical / candidate log loss | Seller bootstrap 95% CI | Net at shared .500 threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261134 | 12,131 | .950540 / .950540 | 0 | 45 / 45 | .132895 / .133370 | [-17, +18] | -12 |
| 20261135 | 12,253 | .953726 / .953726 | 0 | 36 / 36 | .127869 / .128808 | [-16, +16] | +2 |

The confirm sets share 2,621 IDs. Deduplicating gives 21,763 unique IDs and net -1 whether the first or last seed has priority; seller bootstrap CI [-23, +20] or [-24, +21]. The candidate is not a predictive improvement. Both cuts show worse log loss; tune threshold optimization preserved accuracy but did not improve it. These four omitted fields are deprioritized as a package. This does not rule out other metadata or interactions, but no public submission is warranted from this result.

Private paired CSV SHA-256: seed 20261134 `bb2ae490af89914a8dfc9682ac05c309dc0243b10020ee0d6239d6c93dcda5d9`; seed 20261135 `d6143028acae2334671932a1f14cee4c8bbc7def833039778c208f0974988bfa`. The CSVs include item ID, seller, label, both probabilities, decisions and deltas. JSON reports include input/split hashes and seen/new breakdown. Keep paired rows private.
