"""Plan the accuracy of the complete test, keeping hidden uncertainty explicit."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import math
import numpy as np

PUBLIC_ROWS = 21000
HIDDEN_ROWS = 9000


def global_accuracy(public_accuracy: float, hidden_accuracy: float,
                    public_rows: int = PUBLIC_ROWS, hidden_rows: int = HIDDEN_ROWS) -> float:
    if not (0 <= public_accuracy <= 1 and 0 <= hidden_accuracy <= 1 and
            public_rows > 0 and hidden_rows > 0):
        raise ValueError("invalid rates or partition sizes")
    return (public_rows * public_accuracy + hidden_rows * hidden_accuracy) / (public_rows + hidden_rows)


def required_public_correct(hidden_accuracy: float, target: float = .96,
                            public_rows: int = PUBLIC_ROWS, hidden_rows: int = HIDDEN_ROWS) -> int:
    """Minimum public numerator for a fixed assumed hidden rate, using exact decimals.

    A value above public_rows signals an infeasible scenario. This is an
    accounting threshold, not an estimate or confidence bound for hidden rows.
    """
    if not (0 <= hidden_accuracy <= 1 and 0 <= target <= 1 and public_rows > 0 and hidden_rows > 0):
        raise ValueError("invalid planning inputs")
    value = Decimal(str(target)) * (public_rows + hidden_rows) - Decimal(str(hidden_accuracy)) * hidden_rows
    return max(0, int(value.to_integral_value(rounding=ROUND_CEILING)))


def required_hidden_correct(public_correct: int, target: float = .96,
                            public_rows: int = PUBLIC_ROWS, hidden_rows: int = HIDDEN_ROWS) -> int:
    if not (0 <= public_correct <= public_rows and int(public_correct) == public_correct and
            public_rows > 0 and hidden_rows > 0 and 0 <= target <= 1):
        raise ValueError("invalid public count or target")
    total = int((Decimal(str(target)) * (public_rows + hidden_rows)).to_integral_value(rounding=ROUND_CEILING))
    return max(0, total - int(public_correct))


def expected_flip_budget(net_gain: float, flip_correctness: float, credit_fraction: float = 1.) -> int | None:
    """Expected budget only: correctness and transfer assumptions are inputs.

    Use credit_fraction=.7 when only public improvement is credited for
    candidates sampled uniformly across test. Neither expectation nor a
    point estimate of correctness proves a high probability of success.
    """
    if net_gain < 0 or not 0 <= flip_correctness <= 1 or not 0 < credit_fraction <= 1:
        raise ValueError("invalid flip budget inputs")
    if net_gain == 0:
        return 0
    if flip_correctness <= .5:
        return None
    return math.ceil(net_gain / (credit_fraction * (2 * flip_correctness - 1)) - 1e-10)


def evaluate_joint_hidden_draws(public_correct: int, hidden_correct_draws,
                                target: float = .96, confidence: float = .95,
                                public_rows: int = PUBLIC_ROWS, hidden_rows: int = HIDDEN_ROWS) -> dict:
    """Evaluate joint predictive counts; do not generate independent row draws here.

    Each input draw must describe one complete, jointly sampled hidden
    outcome. Correlated seller, category and policy errors belong in that
    upstream predictive model. Passing this gate is conditional on it, not
    a certificate from observed private labels.
    """
    threshold = required_hidden_correct(public_correct, target, public_rows, hidden_rows)
    if not 0 < confidence < 1:
        raise ValueError("invalid confidence")
    draws = np.asarray(hidden_correct_draws, dtype=float)
    if (draws.ndim != 1 or not len(draws) or not np.isfinite(draws).all() or
            np.any(draws < 0) or np.any(draws > hidden_rows) or np.any(draws != np.rint(draws))):
        raise ValueError("provide finite joint integer hidden counts")
    global_draws = (public_correct + draws) / (public_rows + hidden_rows)
    success = float(np.mean(draws >= threshold))
    return {"hidden_correct_required": threshold,
            "expected_global_accuracy": float(global_draws.mean()),
            "global_lower_predictive_quantile": float(np.quantile(global_draws, 1 - confidence, method="inverted_cdf")),
            "prob_global_at_least_target": success,
            "passes_conditional_gate": bool(success + 1e-12 >= confidence),
            "confidence": confidence, "status": "conditional_on_valid_predictive_draws"}
