# Fit-only structural aggregation audit (2026-11-03)

Predeclared plan: `results/inductive_aggregates_20261103_plan.md`. Code: `scripts/audit_refit_context_transport_20261103.py` with `AP_CONTEXT_MODE=fit`. Input ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`; 70k real train labels, `used=1`, no public scores or test labels used.

The two CatBoost arms share the same seller-group fit/tune/confirm split, text word+char TF-IDF OOF, and supervised target encodings computed within fit. The historical arm calculates 13 unsupervised frequencies, seller summary statistics and relative category price using all 70k train feature rows. The alternative fits those 13 maps only on fit feature rows and applies them to tune and confirm. The script calls the alternative `ctx100` internally for compatibility with its prior context experiment; here `ctx100` means **fit-only**, not 100k. Each arm chooses early stopping and accuracy threshold on tune, then evaluates confirm once.

| Seed | Confirm N | Historical accuracy | Fit-only accuracy | Fit-only net | Fixes / breaks | Historical / fit-only log loss | Seller bootstrap 95% CI | Net at common .500 threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261132 | 12,738 | .948422 | .949129 | +9 | 87 / 78 | .134049 / .138143 | [-17, +35] | -25 |
| 20261133 | 12,210 | .954464 | .950860 | -44 | 59 / 103 | .125611 / .132766 | [-76, -14] | -52 |

The confirm sets overlap in 2,680 IDs. From 24,948 rows there are 22,268 unique IDs. Keeping the first seed's prediction on overlap gives net -26 with seller-bootstrap CI [-62, +9]; keeping the second gives -34, CI [-70, +3]. At the shared .500 threshold the unique-ID net is -62 under either priority. The fit-only arm has worse log loss in both cuts by .00409 and .00716. Its first-cut tuned accuracy advantage depended on a low tune-selected threshold (.44 vs .52 for historical); it did not generalize to the second cut.

**Decision:** do not promote this straightforward fit-only mapping as a submission or replace the current baseline. It changes feature distributions and calibration substantially. The historical extraction itself is transductive with respect to confirm feature rows, though never confirm labels; the result does not prove that transduction improves public test performance. A stricter final pipeline would need feature-map fitting and validation/refit consistency, plus calibration and fresh paired comparisons against the strong baseline and bag3. This audit closes only the uncalibrated fit-only mapping tested here.

Private paired CSVs contain ID, seller, label, probabilities, predictions and deltas. SHA-256: seed 20261132 `576efff31efba915b8e57a0e5c5fd311d5baba9ba83f6c260ad825e0638cc2df`; seed 20261133 `7cbbba855b6fdb0129413e3e5f94b45b67ef8d03ddb3643efebeb55f530c187a`. JSON reports include split counts, changed-feature counts, thresholds, seen/new and input hashes. Keep the paired files private.
