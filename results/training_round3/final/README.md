# Final ensembles, round3

Two families preregistered: five new bases only on7k tune, and historical+samefive on mutually excluded common tune rows. C selected5CV on tune, threshold.5 fixed. Both selection locks precede loading confirm vectors. No candidate was selected by confirmation.

All base hyperparameter searches used the same tune set, so inner CV here is conditional; all historical holdouts were reused. Confirmation comparisons are exploratory, not independent proof of public improvement. Historical production is hybrid: base70k with older auxiliary fits frozen. Seen/new is relative to newdev56k.

Both CSVs are offline model-only candidates. No flips, recovered labels, public-score calibration or individual public memberships enter training. No missing base may be replaced silently. Public scores are unobserved.
