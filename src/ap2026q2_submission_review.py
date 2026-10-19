"""Statistical helpers for the submission-review notebook.
Only aggregate score evidence is used. Model probabilities and posterior
assumptions never become independently verified labels.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.special import betaln, gammaln, logsumexp, betainc
from scipy.stats import beta, binom, betabinom, hypergeom
from scipy.optimize import brentq, milp, Bounds, LinearConstraint
from numpy.polynomial.legendre import leggauss

def score_count_candidates(score: float, n_public: int, mode: str = "truncate",
                           decimals: int = 5) -> np.ndarray:
    """Integer correct counts compatible with a displayed accuracy."""
    step = 10.0 ** (-decimals)
    if mode == "truncate":
        lower, upper = score, score + step
    elif mode == "round":
        lower, upper = score - step / 2, score + step / 2
    else:
        raise ValueError("mode must be truncate or round")
    first = int(np.ceil(lower * n_public - 1e-8))
    stop = int(np.ceil(upper * n_public - 1e-8))
    return np.arange(max(0, first), min(n_public + 1, stop), dtype=int)

def displayed_score(correct: int, n_public: int, mode: str) -> float:
    raw = correct / n_public * 1e5
    return float(np.floor(raw + (0.5 if mode == "round" else 1e-8)) / 1e5)

def cohort_posterior(n: int, delta: int, pi: float = .7, a: float = 1.,
                     b: float = 1., public_total: int | None = None,
                     test_total: int = 30000) -> dict:
    """Beta mixture integrating unknown public count K.
    If public_total is supplied, use a uniform fixed-size public-mask
    model (hypergeometric). Otherwise use its Bernoulli approximation.
    """
    if n < 1 or abs(delta) > n or a <= 0 or b <= 0 or not 0 < pi < 1:
        raise ValueError("Invalid n, delta, pi, a or b")
    k = np.arange(abs(delta), n + 1)
    k = k[(k + delta) % 2 == 0]
    successes = (k + delta) // 2
    if public_total is None:
        mask_log_prob = binom.logpmf(k, n, pi)
    else:
        mask_log_prob = hypergeom.logpmf(k, test_total, public_total, n)
    log_choose = gammaln(k + 1) - gammaln(successes + 1) - gammaln(k - successes + 1)
    weights = mask_log_prob + log_choose + betaln(a + successes, b + k - successes) - betaln(a, b)
    weights = np.exp(weights - logsumexp(weights))
    aa, bb = a + successes, b + k - successes
    means = aa / (aa + bb)
    def cdf(x):
        return float(weights @ betainc(aa, bb, x))
    interval = [brentq(lambda x: cdf(x) - q, 1e-12, 1 - 1e-12) for q in [.025, .975]]
    finite_pmf = np.zeros(n + 1)
    for kk, ss, alpha, bet, weight in zip(k, successes, aa, bb, weights):
        u = np.arange(n - kk + 1)
        finite_pmf[ss + u] += weight * betabinom.pmf(u, n - kk, alpha, bet)
    result = {
        "n": n, "delta": delta, "pi": pi, "a": a, "b": b,
        "k": k, "successes": successes, "weights": weights,
        "alpha_components": aa, "beta_components": bb,
        "theta_mean": float(weights @ means),
        "theta_ci95": interval, "theta_probability_gt_half": 1 - cdf(.5),
        "finite_cohort_accuracy_mean": float(finite_pmf @ np.arange(n + 1) / n),
        "finite_correct_count_pmf": finite_pmf,
        "expected_public_rows": float(weights @ k),
        "mask_model": "hypergeometric" if public_total is not None else "binomial",
    }
    assert np.isclose(weights.sum(), 1) and np.isclose(finite_pmf.sum(), 1)
    return result

def theta_density(posterior: dict, x: np.ndarray) -> np.ndarray:
    return np.sum([
        w * beta.pdf(x, a, b) for w, a, b in zip(
            posterior["weights"], posterior["alpha_components"], posterior["beta_components"])
    ], axis=0)

def net_distribution(probabilities, pi: float = .7) -> np.ndarray:
    """P(sum Z=d), with P(Z=-1,0,+1) = (pi*(1-p),1-pi,pi*p)."""
    result = np.ones(1)
    for p in np.asarray(probabilities):
        result = np.convolve(result, [pi * (1 - p), 1 - pi, pi * p])
    assert np.isclose(result.sum(), 1)
    return result

def shared_population_grid(cohorts, pi=.7, a=1., b=1., nodes=192):
    """Posterior grid for one shared theta across disjoint observed cohorts.
    Exchangeability is a modeling assumption; overlapping cohorts must
    first be reduced to a joint system, rather than multiplied here.
    """
    theta, weights = leggauss(nodes)
    theta, weights = (theta + 1) / 2, weights / 2
    weights *= np.exp((a - 1) * np.log(theta) + (b - 1) * np.log1p(-theta) - betaln(a, b))
    for n, delta in cohorts:
        likelihood = np.array([net_distribution(np.repeat(t, n), pi)[n + delta] for t in theta])
        weights *= likelihood
    weights /= weights.sum()
    return theta, weights

def predictive_net(n, theta, weights, pi=.7):
    law = np.sum([w * net_distribution(np.repeat(t, n), pi) for t, w in zip(theta, weights)], axis=0)
    delta = np.arange(-n, n + 1)
    valid = law > 1e-15
    cdf = np.cumsum(law)
    return {
        "n": n, "delta": delta, "law": law,
        "mean_net": float(delta @ law),
        "probability_gain": float(law[delta > 0].sum()),
        "probability_tie": float(law[delta == 0].sum()),
        "interval95": [int(delta[np.searchsorted(cdf, q)]) for q in [.025, .975]],
        "outcome_entropy_bits": float(-law[valid] @ np.log2(law[valid])),
    }

def joint_integer_bound(system, atom: int, time_limit: float = 5.) -> tuple[int, int]:
    P = np.asarray(system["patterns"], float)
    counts = np.asarray(system["atom_counts"], int)
    known = np.asarray(system["known_alternative"], int)
    lo, hi = -counts.copy(), counts.copy()
    lo[known == 1] = 0
    hi[known == 0] = 0
    constraint = LinearConstraint(P, system["delta"], system["delta"])
    objective = np.zeros(len(counts)); objective[atom] = 1
    left = milp(objective, integrality=np.ones(len(counts)), bounds=Bounds(lo, hi),
                constraints=constraint, options={"time_limit": time_limit})
    right = milp(-objective, integrality=np.ones(len(counts)), bounds=Bounds(lo, hi),
                 constraints=constraint, options={"time_limit": time_limit})
    if not (left.success and right.success):
        raise RuntimeError("Integer bound did not complete; do not label it exact.")
    return int(round(left.fun)), int(round(-right.fun))

def load_snapshot(root: Path):
    path = Path(root) / "results/revision_submissions/snapshot_20261018.json"
    return json.loads(path.read_text(encoding="utf-8"))

