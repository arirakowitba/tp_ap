# Submission 53 postmortem (2026-10-27)

Ariel observed public 53 = 0.94114; 48 = 0.94633. Exact CSV SHA256: 53 `a894af2ebabbe3166caf984842813b1a47d0504b181a5800198655e2b15813a8`; 48 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`. Accuracy delta −0.00519; if 21,000 public rows, about −109 correct in aggregate. Actual public subset and individual labels are unknown.

53 changed 411 test predictions relative to 48, all on seen sellers, 320 distinct sellers; 250 to used, 161 to new. 127/411 changed rows had at most two training records from their seller. The 411 changed predictions exactly matched the alpha1 gate. Of these, 269 moved toward the HistGB+FastText baseline and 142 away. No per-ID correctness can be inferred from the public score.

The train confirms for alpha1 gate compared against HistGB+FastText, **not an exact train analogue of submission 48**. The public result decisively rejects replacing 48 with the 411-change gate policy. A spot audit found feature positions in the full fit matched validation (two seller features before word/char teacher columns), and 53 ID/row order, seen-only gate and SHA match the prepared artifact. A subtle train/test feature or distribution failure remains possible; this audit does not prove the cause.

On three previously saved labelled confirms (105/108 random row cuts and 110 temporal), the gate versus baseline was positive in each seller-count band; even seller count >=31 net +6,+8,+10. Thus filtering by seller count alone has no train-only support for recovering public performance against 48. The confirms overlap and share model design. Their positive result does not transfer to the stronger public 48 policy.

Decision: demote 53 and do not send 51/52, which use the same seller LOO family, without a new, independently validated comparison to a faithful 48 proxy. Restore exact 48 as the strongest observed public submission. Do not use public aggregate delta as labels for choosing a subset of the 411 IDs. Next work: reconstruct a train policy comparable to 48 using its generation steps, then evaluate changes against it on a seller-disjoint and chronological confirm; measure paired accuracy, log loss, seller bootstrap and seen/new segments. Test predictions remain diagnostic only.

Checkpoint Library: libfile_4d3395b194c4819191ddf9c46206101e.
