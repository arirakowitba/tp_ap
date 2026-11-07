# Ledger reconciliation delta — five scored events of 2026-11-07

## Outcome

The apparent “106 submissions versus ledger 83” mismatch is now decomposed correctly.

Base canonical catalog (through the prior five-slot day):
- 128 total events;
- 101 scored events with a preserved vector;
- 7 scored events without a recoverable vector;
- 2 errors without score;
- 18 offline drafts;
- 89 unique scored vector hashes.

Appending the five newly reported scored events yields:
- **133 total catalog events**;
- **113 scored events**;
- **106 scored events with a preserved vector**;
- 7 scored events without vector;
- 2 errors;
- 18 drafts;
- **94 unique scored vector hashes**.

Thus, 106 is not the total number of Kaggle score events. It is the updated count of scored events for which a 30k vector can be represented in the repository.

## Five audited vectors

| Event | Score | Hits (Npub 20998) | Flips vs 0.94719 anchor | Normalized vector SHA-256 | Pinned commit |
|---|---:|---:|---:|---|---|
| EVT-129 / Slot 1 | 0.94738 | 19893 | 25 | `7b5dcb6f7d773629b487bb8f1b15872b30802220877db6629578843e82fb37bc` | `b8aa4c091829bd7882a281df1e6703f47e2a181b` |
| EVT-130 / Slot 2 | 0.94738 | 19893 | 45 | `e80b5750e0b13d3d32664ada985873be68a4cc060e044b187286d1f755abcf40` | `78e5dd5d50a2992f21b69f22ab949e645b3328db` |
| EVT-131 / Slot 3 | 0.94709 | 19887 | 40 | `cb6c478cd77e26703d476ca393a4b546a23f1572a20877179b938eca7fb6e1fd` | `dc6d9ae040022862c81306c2156dc2bee92f39fd` |
| EVT-132 / Slot 4 | **0.94752** | **19896** | 41 | `11aabc766711e97998ca48258208ca2ddde7b25cf4bd27ff8da051ffa5fdca81` | `23ff472333665c86fb82e69a9cd00671d057bceb` |
| EVT-133 / Slot 5 | 0.94738 | 19893 | 60 | `ffb7c7c7317af022df419a622056db71146e7ff29ef8a95fb396da722f0c11ec` | `44e02fb08b88eedfcae296944ec0cdba384bac1d` |

Each vector contains 30,000 unique IDs and is new relative to the 89 hashes in the base catalog.

## Critical branch/path integrity issue

The files at the same paths on `main` are stale portfolio versions and are not the evaluated AG vectors:
- current main Slot 3: 65 flips; evaluated commit Slot 3: 40;
- current main Slot 4: 85 flips; evaluated commit Slot 4: 41;
- current main Slot 5: 105 flips; evaluated commit Slot 5: 60.

Slot 1 and Slot 2 also have different normalized hashes between main and the AG evaluated revisions. Therefore filename alone is not a valid primary key, and rebuilding from the current main path would silently bind scores to the wrong prediction vectors.

The delta pins both commit and normalized `ID,target` hash. Source:
`results/canonical_ledger_delta_20261107/events_129_133.json`.

## Remaining limitation

The repository records the score-to-vector relationship through AG messages, filenames and commit chronology. Kaggle did not provide an uploaded-file SHA in the evidence. Consequently the mapping is strongly reconstructed but not cryptographically certified by Kaggle. A future authoritative export should record upload timestamp, filename, score and uploaded-file hash when available.

## Statistical consequence

Appending five new unique vectors does not make per-ID effects identifiable. The previous system had 89 equations, rank 83 and 2,725 active IDs. Even if all five rows increased rank by one, rank would be at most 88, leaving at least 2,637 null directions. Scores remain aggregate constraints only.

No Kaggle action or quota was used.
