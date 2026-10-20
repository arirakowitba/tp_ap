"""Update disjoint flip cohorts from public scores without inventing test labels.

All probes in a round must be built from the same baseline.  A cohort may be
present in several probes; the resulting equations are solved *jointly*.
The public mask is modeled as a uniformly sampled, fixed-size subset.  The
prior assumes independent exchangeable flips within each cohort; this is a
sensitivity model, not a source of individual ground truth.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from scipy.stats import betabinom, hypergeom

from ap2026q2_submission_review import score_count_candidates


def public_correct(score: str, n_public: int = 21000, mode: str = "truncate") -> int:
    """Reject ambiguous score displays instead of silently choosing a count."""
    candidates = score_count_candidates(float(score), n_public, mode)
    if len(candidates) != 1:
        raise ValueError(f"score {score}: possible numerators {candidates.tolist()}; "
                         "check public fraction and leaderboard display")
    return int(candidates[0])


def cohort_prior(n: int, a: float = 2., b: float = 2.,
                 test_total: int = 30000, public_total: int = 21000) -> pd.DataFrame:
    """P(public gain=g), E(hidden gain|g), E[public rows|g].

    For M public rows and K correct alternate labels, g=2K-M.  Integrate
    unknown M with Hypergeom and K with BetaBinomial(a,b).  For the hidden
    rows, the conditional expected correctness is (a+K)/(a+b+M).
    """
    if not (0 < n <= test_total and 0 < public_total < test_total and a > 0 and b > 0):
        raise ValueError("invalid cohort size, public size or beta prior")
    groups: dict[int, list[tuple[float, int, float]]] = {}
    for m in range(max(0, n - (test_total - public_total)), min(n, public_total) + 1):
        mask_log = hypergeom.logpmf(m, test_total, public_total, n)
        for k in range(m + 1):
            g = 2 * k - m
            log_weight = mask_log + betabinom.logpmf(k, m, a, b)
            groups.setdefault(g, []).append((log_weight, m,
                (n - m) * (2 * (a + k) / (a + b + m) - 1)))
    rows = []
    for g, items in groups.items():
        logs = np.array([i[0] for i in items])
        weights = np.exp(logs - logsumexp(logs))
        rows.append((g, logsumexp(logs),
                     float(weights @ np.array([i[2] for i in items])),
                     float(weights @ np.array([i[1] for i in items]))))
    result = pd.DataFrame(rows, columns=["gain", "log_mass", "hidden_gain", "public_rows"])
    result["probability"] = np.exp(result.log_mass - logsumexp(result.log_mass))
    assert np.isclose(result.probability.sum(), 1)
    return result.sort_values("gain").reset_index(drop=True)


def infer_cohorts(sizes: dict[str, int], experiments: dict[str, list[str]],
                  scores: dict[str, str | None], baseline_score: str,
                  n_public: int = 21000, mode: str = "truncate",
                  a: float = 2., b: float = 2.) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact posterior over cohort public gains consistent with *all* scores.

    Five cohorts and the five planned contrasts are cheap to enumerate.  For
    larger underdetermined designs, split cohorts or add an independent probe.
    Return (per-cohort summary, joint feasible configurations).  Missing
    scores contribute no evidence.  Observed scores with a missing group
    definition, an overlapping audit or inconsistent equations are errors.
    """
    names = list(sizes)
    if len(names) > 5 or not names or any(n < 1 or n > 128 for n in sizes.values()):
        raise ValueError("supports one to five cohorts of 1-128 flips")
    unknown = set().union(*[set(x) for x in experiments.values()]) - set(names)
    if unknown:
        raise ValueError(f"undefined cohorts: {sorted(unknown)}")
    baseline = public_correct(baseline_score, n_public, mode)
    observed = [(name, experiments[name], public_correct(score, n_public, mode) - baseline)
                for name, score in scores.items() if score is not None and name in experiments]
    if any(name not in experiments for name, score in scores.items() if score is not None):
        raise ValueError("scored submission without registered cohort pattern")
    matrix = np.array([[int(group in members) for group in names]
                       for _, members, _ in observed], dtype=float).reshape(-1, len(names))
    target = np.array([gain for _, _, gain in observed], dtype=float)
    # Pick linearly independent rows and pivot columns; redundant scores must
    # still match the complete matrix below.
    row_ix: list[int] = []
    for i in range(len(matrix)):
        candidate = row_ix + [i]
        if np.linalg.matrix_rank(matrix[candidate]) > len(row_ix):
            row_ix.append(i)
    pivots: list[int] = []
    for j in range(len(names)):
        if len(pivots) == len(row_ix):
            break
        candidate = pivots + [j]
        if np.linalg.matrix_rank(matrix[row_ix][:, candidate]) > len(pivots):
            pivots.append(j)
    free = [i for i in range(len(names)) if i not in pivots]
    priors = {name: cohort_prior(n, a, b, public_total=n_public) for name, n in sizes.items()}
    lookup = {name: table.set_index("gain") for name, table in priors.items()}
    if np.prod([len(lookup[names[i]]) for i in free], dtype=float) > 250000:
        raise ValueError("underidentified design is too large: first score the "
                         "planned earlier probes, then rerun the posterior")
    records = []
    free_ranges = [lookup[names[i]].index.to_numpy(dtype=int) for i in free]
    for values in product(*free_ranges):
        gains = np.zeros(len(names), dtype=int)
        gains[free] = values
        if pivots:
            residual = target[row_ix] - matrix[row_ix][:, free] @ gains[free]
            calculated = np.linalg.solve(matrix[row_ix][:, pivots], residual)
            if np.any(np.abs(calculated - np.rint(calculated)) > 1e-8):
                continue
            gains[pivots] = np.rint(calculated).astype(int)
        if len(target) and not np.array_equal(matrix.astype(int) @ gains, target.astype(int)):
            continue
        if any(int(gains[i]) not in lookup[name].index for i, name in enumerate(names)):
            continue
        log_mass = sum(float(lookup[name].loc[int(gains[i]), "log_mass"])
                       for i, name in enumerate(names))
        records.append({**dict(zip(names, gains.tolist())), "log_mass": log_mass})
    if not records:
        raise ValueError("no configuration explains all public scores; check masks, "
                         "rounding, baseline, overlap or submitted files")
    joint = pd.DataFrame(records)
    joint["posterior"] = np.exp(joint.log_mass - logsumexp(joint.log_mass))
    summaries = []
    for name in names:
        g = joint[name].to_numpy(dtype=int)
        w = joint.posterior.to_numpy()
        hidden = lookup[name].loc[g, "hidden_gain"].to_numpy(dtype=float)
        summaries.append({"cohort": name, "n": sizes[name],
                          "expected_public_gain": float(w @ g),
                          "expected_hidden_gain": float(w @ hidden),
                          "prob_public_gain_positive": float(w[g > 0].sum()),
                          "feasible_public_gains": int(np.unique(g).size),
                          "min_public_gain": int(g.min()), "max_public_gain": int(g.max())})
    return pd.DataFrame(summaries), joint


def choose_combination(summary: pd.DataFrame, joint: pd.DataFrame,
                       cohorts: dict[str, pd.DataFrame], baseline: pd.DataFrame,
                       hidden_transfer: float = .5, require_observed_positive: bool = True
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank all disjoint combinations by expected hidden gain, then export one.

    A conservative gate requires a positive *identified* public gain for a
    tested bundle or for each individual cohort.  This admits A+B if their
    joint test wins, even before A and B can be distinguished separately.
    Set it to False only for a deliberate probe.
    A transfer factor between zero and one discounts reuse of public feedback.
    """
    if not 0 <= hidden_transfer <= 1:
        raise ValueError("hidden_transfer must be in [0,1]")
    validate_baseline(baseline)
    ids_seen: set[int] = set()
    for name, frame in cohorts.items():
        if not {"ID", "alternative_label"} <= set(frame):
            raise ValueError(f"{name}: expected ID,alternative_label")
        ids = frame.ID.astype(int).tolist()
        if len(ids) != len(set(ids)) or ids_seen.intersection(ids):
            raise ValueError("cohorts overlap or repeat IDs")
        ids_seen.update(ids)
        if name not in set(summary.cohort) or len(ids) != int(summary.set_index("cohort").loc[name, "n"]):
            raise ValueError(f"{name}: size differs from posterior")
        original = baseline.set_index("ID").loc[ids, "condition"].to_numpy()
        target = frame.alternative_label.to_numpy()
        if not set(target) <= {"new", "used"} or np.any(original == target):
            raise ValueError(f"{name}: invalid label or non-flip")
    if set(cohorts) != set(summary.cohort):
        raise ValueError("missing candidate cohort audit")
    available = list(cohorts)
    row_by_name = summary.set_index("cohort")
    options = []
    for bits in product((0, 1), repeat=len(available)):
        chosen = [name for name, bit in zip(available, bits) if bit]
        expected_hidden = sum(float(row_by_name.loc[x, "expected_hidden_gain"]) for x in chosen)
        public_gains = joint[chosen].sum(axis=1).to_numpy() if chosen else np.zeros(len(joint))
        if require_observed_positive and chosen:
            identified_bundle = np.unique(public_gains).size == 1 and public_gains[0] > 0
            individually_positive = all(row_by_name.loc[x, "feasible_public_gains"] == 1 and
                                        row_by_name.loc[x, "expected_public_gain"] > 0
                                        for x in chosen)
            if not (identified_bundle or individually_positive):
                continue
        options.append({"cohorts": ",".join(chosen), "flips": sum(len(cohorts[x]) for x in chosen),
                        "expected_public_gain": float(joint.posterior @ public_gains),
                        "expected_hidden_gain_discounted": hidden_transfer * expected_hidden,
                        "prob_public_improvement": float(joint.posterior[public_gains > 0].sum())})
    ranking = pd.DataFrame(options).sort_values(
        ["expected_hidden_gain_discounted", "expected_public_gain", "flips"],
        ascending=[False, False, True]).reset_index(drop=True)
    winner = ranking.iloc[0].cohorts.split(",") if ranking.iloc[0].cohorts else []
    result = baseline.copy()
    for name in winner:
        labels = cohorts[name].set_index("ID").alternative_label
        mask = result.ID.isin(labels.index)
        result.loc[mask, "condition"] = result.loc[mask, "ID"].map(labels).to_numpy()
    validate_baseline(result)
    assert int((result.condition != baseline.condition).sum()) == int(ranking.iloc[0].flips)
    return ranking, result


def validate_baseline(frame: pd.DataFrame, rows: int = 30000) -> None:
    if list(frame.columns) != ["ID", "condition"] or len(frame) != rows or not np.array_equal(
        frame.ID.to_numpy(), np.arange(1, rows + 1)) or not set(frame.condition) <= {"new", "used"}:
        raise ValueError("submission must have exactly 30000 ordered ID,condition rows")
