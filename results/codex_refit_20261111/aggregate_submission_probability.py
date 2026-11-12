"""Conditional probability of a target gain in binary classification accuracy.

The total number of corrected flips is a scenario input. This module only
models the random public subset, drawn uniformly without replacement.
"""
from scipy.stats import hypergeom


def probability_of_public_gain(*, population, public_size, flips, fixes, minimum_net):
    """Return P(public corrections - public regressions >= minimum_net)."""
    if not 0 <= public_size <= population:
        raise ValueError("public_size must be between 0 and population")
    if not 0 <= fixes <= flips <= population:
        raise ValueError("Require 0 <= fixes <= flips <= population")
    if not population:
        return float(minimum_net <= 0)

    low = max(0, public_size - (population - flips), minimum_net)
    high = min(flips, public_size)
    if low > high:
        return 0.0
    if flips == 0:
        return float(minimum_net <= 0)

    public_flips = range(low, high + 1)
    needed = [(minimum_net + selected + 1) // 2 for selected in public_flips]
    public_coverage = hypergeom.pmf(public_flips, population, flips, public_size)
    success_given_coverage = hypergeom.sf([n - 1 for n in needed], flips, fixes, public_flips)
    return float((public_coverage * success_given_coverage).sum().clip(0, 1))
