"""Backtesting a VaR model.

A VaR estimate is a falsifiable claim: at 99%, losses should exceed it on
about 1 day in 100. Backtesting checks whether they did. Without it a
risk model is an opinion with decimal places.

Two things can be wrong, and they need separate tests:

* the *rate* is off -- too many exceedances means the model understates
  risk, too few means capital is tied up against a danger that is not
  there. Kupiec's proportion-of-failures test looks at this.
* the *timing* is off -- the right number of exceedances, but clustered
  into one bad fortnight instead of scattered. That is the worse
  failure: a model that is fine on average and blind during the only
  week that mattered. Christoffersen's independence test looks at this.

Both are likelihood ratio tests, so a large statistic and a small
p-value mean the model is rejected. Note the direction: failing to
reject is not proof the model is right, it means one year of data was
not enough to prove it wrong. At 99% a year of data contains about 2.5
expected exceedances, so these tests have very little power. Treat a
pass as "not obviously broken".

Everything here uses the `var` sign convention: VaR is a positive loss,
returns are signed, and an exceedance is `return < -var`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy import special, stats

from src.var import _check_confidence, historical_var

# Basel's traffic light, from the 1996 market risk amendment. The zones
# are cumulative binomial probabilities of seeing at most this many
# exceedances if the model were correct: past 95% the model is doubted,
# past 99.99% it is rejected outright.
_YELLOW_THRESHOLD = 0.95
_RED_THRESHOLD = 0.9999


@dataclass(frozen=True)
class TestResult:
    """One likelihood ratio test."""

    # pytest collects any class whose name starts with "Test". This is a
    # result object, not a test suite; renaming it would be the other
    # way out, but TestResult is what the literature calls it.
    __test__: ClassVar[bool] = False

    name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_obs: int
    n_exceedances: int
    expected_exceedances: float

    def rejects(self, alpha: float = 0.05) -> bool:
        """Whether the model is rejected at the given significance level.

        A method rather than a stored field so the threshold stays a
        decision the reader makes, not one baked into the result.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        return bool(self.p_value < alpha)

    def __str__(self) -> str:
        return (
            f"{self.name}: LR={self.statistic:.3f}, p={self.p_value:.4f}, "
            f"{self.n_exceedances} exceedances vs {self.expected_exceedances:.1f} expected"
        )


@dataclass(frozen=True)
class BacktestReport:
    """Everything a rolling backtest produced."""

    var: pd.Series = field(repr=False)
    returns: pd.Series = field(repr=False)
    breaches: pd.Series = field(repr=False)
    kupiec: TestResult
    independence: TestResult
    conditional_coverage: TestResult
    basel_zone: str
    confidence: float

    @property
    def exceedance_rate(self) -> float:
        """Observed share of days on which VaR was breached."""
        return float(self.breaches.mean())

    @property
    def breach_dates(self) -> pd.DatetimeIndex:
        """When the model failed. Usually more informative than the rate."""
        return self.breaches.index[self.breaches]

    def summary(self) -> pd.DataFrame:
        """The three tests side by side, for a report."""
        return pd.DataFrame(
            [
                {
                    "test": result.name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "rejects_at_5pct": result.rejects(0.05),
                }
                for result in (self.kupiec, self.independence, self.conditional_coverage)
            ]
        ).set_index("test")


def exceedances(returns: pd.Series, var: float | pd.Series) -> pd.Series:
    """Days on which the loss was worse than VaR said it would be.

    `var` may be one number (a static model) or a series aligned to
    `returns` (a model re-estimated each day). The comparison is strict:
    a loss exactly equal to VaR is not a breach, matching the definition
    of the quantile.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be a pandas Series indexed by date")

    clean = returns.dropna()
    if clean.empty:
        raise ValueError("No returns to test")

    if isinstance(var, pd.Series):
        aligned = var.reindex(clean.index)
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(
                f"The VaR series is missing {missing} of the {len(clean)} dates being "
                f"tested. Align it to the returns before backtesting."
            )
    else:
        aligned = pd.Series(float(var), index=clean.index)

    return (clean < -aligned).rename("breach")


def kupiec_pof(
    returns: pd.Series, var: float | pd.Series, confidence: float = 0.99
) -> TestResult:
    """Kupiec's proportion-of-failures test (unconditional coverage).

    Asks whether the observed exceedance rate is consistent with
    `1 - confidence`, ignoring when the exceedances happened. The
    statistic is chi-square with one degree of freedom under the null
    that the model is correctly calibrated.
    """
    _check_confidence(confidence)
    breaches = exceedances(returns, var)
    return _kupiec_from_breaches(breaches, confidence)


def christoffersen_independence(breaches: pd.Series) -> TestResult:
    """Christoffersen's test that exceedances do not cluster.

    Compares a first-order Markov chain -- where the chance of a breach
    tomorrow depends on whether there was one today -- against a model
    where it does not. If breaches arrive in clusters, the Markov model
    fits much better and the statistic is large.

    Clustering is the symptom of a model that ignores volatility
    persistence, which is what an equally weighted historical VaR does:
    it reacts to a spike in volatility only once the spike has entered
    the estimation window, by which time the losses have happened. This
    test is the reason `covariance.ewma_covariance` exists.
    """
    breaches = _as_breach_series(breaches)

    previous = breaches.to_numpy()[:-1]
    current = breaches.to_numpy()[1:]

    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))

    # Transition probabilities, with the convention that an unobserved
    # starting state contributes nothing rather than dividing by zero.
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) else 0.0

    unrestricted = (
        special.xlogy(n00, 1 - pi01)
        + special.xlogy(n01, pi01)
        + special.xlogy(n10, 1 - pi11)
        + special.xlogy(n11, pi11)
    )
    restricted = special.xlogy(n00 + n10, 1 - pi) + special.xlogy(n01 + n11, pi)

    statistic = float(-2.0 * (restricted - unrestricted))
    # Floating point can push a statistic that is exactly zero slightly
    # negative, which would give a p-value above one.
    statistic = max(statistic, 0.0)

    return TestResult(
        name="Christoffersen independence",
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, 1)),
        degrees_of_freedom=1,
        n_obs=len(breaches),
        n_exceedances=int(breaches.sum()),
        expected_exceedances=float("nan"),  # this test says nothing about the rate
    )


def christoffersen_conditional_coverage(
    returns: pd.Series, var: float | pd.Series, confidence: float = 0.99
) -> TestResult:
    """The joint test: right rate *and* no clustering.

    The two statistics simply add, because the null of correct
    conditional coverage factors into correct unconditional coverage and
    independence. Chi-square with two degrees of freedom.

    Worth reporting alongside its two halves rather than instead of
    them: a rejection here does not say which half failed, and the
    remedy differs -- a wrong rate means recalibrating the confidence
    level, clustering means the volatility model is too slow.
    """
    _check_confidence(confidence)
    breaches = exceedances(returns, var)

    pof = _kupiec_from_breaches(breaches, confidence)
    independence = christoffersen_independence(breaches)
    statistic = pof.statistic + independence.statistic

    return TestResult(
        name="Christoffersen conditional coverage",
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, 2)),
        degrees_of_freedom=2,
        n_obs=pof.n_obs,
        n_exceedances=pof.n_exceedances,
        expected_exceedances=pof.expected_exceedances,
    )


def basel_traffic_light(
    n_exceedances: int, n_obs: int, confidence: float = 0.99
) -> str:
    """Basel's supervisory zone: "green", "yellow" or "red".

    A blunt instrument next to the likelihood ratio tests -- it only
    counts exceedances, and it is one-sided, so a model with too *few*
    breaches stays green however badly calibrated it is. It is here
    because it is what a supervisor actually applies, and because the
    thresholds are a useful sanity check: on 250 days at 99%, green runs
    to 4 exceedances, yellow from 5 to 9, red from 10.
    """
    _check_confidence(confidence)
    if n_obs < 1:
        raise ValueError("Need at least one observation")
    if not 0 <= n_exceedances <= n_obs:
        raise ValueError(f"{n_exceedances} exceedances out of {n_obs} observations is impossible")

    cumulative = float(stats.binom.cdf(n_exceedances, n_obs, 1.0 - confidence))
    if cumulative < _YELLOW_THRESHOLD:
        return "green"
    if cumulative < _RED_THRESHOLD:
        return "yellow"
    return "red"


def rolling_var_backtest(
    returns: pd.Series,
    estimator: Callable[..., float] = historical_var,
    *,
    window: int = 250,
    confidence: float = 0.99,
) -> BacktestReport:
    """Re-estimate VaR each day on a trailing window, then test it.

    This is the only honest way to backtest. Estimating VaR once on the
    whole sample and testing it against that same sample guarantees
    roughly the right number of exceedances -- the quantile was fitted
    to produce them -- and so tests nothing. Here each day's VaR is
    computed from the `window` days *before* it, so every comparison is
    genuinely out of sample.

    `estimator` is any function taking a return series and a `confidence`
    keyword, which is every estimator in `var`. Swapping it is the
    experiment worth running: historical versus normal on the same
    returns shows which assumption survives contact with the data.

    The default 250-day window is one trading year, the Basel standard.
    It is also short enough that at 99% the estimate rests on two or
    three observations, which is the trade-off the whole exercise is
    about.
    """
    _check_confidence(confidence)
    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be a pandas Series indexed by date")

    clean = returns.dropna()
    if window < 30:
        raise ValueError(f"A {window}-day window is too short to estimate a tail quantile")
    if len(clean) <= window:
        raise ValueError(
            f"Need more than {window} returns to backtest a {window}-day window, "
            f"got {len(clean)}. The first {window} days are consumed by the first estimate."
        )

    estimates = []
    values = clean.to_numpy()

    # The thin-tail warning is correct and would fire on every one of
    # the thousands of iterations. Collect it once instead of drowning
    # the caller, but do not swallow it: the caveat is the point.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for end in range(window, len(values)):
            estimates.append(float(estimator(values[end - window : end], confidence=confidence)))

    for message in _unique_messages(caught):
        warnings.warn(f"{message} (raised on each of {len(estimates)} windows)", stacklevel=2)

    var = pd.Series(estimates, index=clean.index[window:], name="var")
    tested = clean.loc[var.index]
    breaches = exceedances(tested, var)

    return BacktestReport(
        var=var,
        returns=tested,
        breaches=breaches,
        kupiec=_kupiec_from_breaches(breaches, confidence),
        independence=christoffersen_independence(breaches),
        conditional_coverage=christoffersen_conditional_coverage(tested, var, confidence),
        basel_zone=basel_traffic_light(int(breaches.sum()), len(breaches), confidence),
        confidence=confidence,
    )


# --- internals -------------------------------------------------------


def _kupiec_from_breaches(breaches: pd.Series, confidence: float) -> TestResult:
    """Kupiec's statistic, given the exceedance indicators."""
    n_obs = len(breaches)
    n_breaches = int(breaches.sum())
    p = 1.0 - confidence
    observed = n_breaches / n_obs

    # xlogy gives 0 * log(0) == 0, which is the right limit for the
    # cases where there were no exceedances at all, or nothing but.
    under_null = special.xlogy(n_obs - n_breaches, 1 - p) + special.xlogy(n_breaches, p)
    under_observed = special.xlogy(n_obs - n_breaches, 1 - observed) + special.xlogy(
        n_breaches, observed
    )

    statistic = max(float(-2.0 * (under_null - under_observed)), 0.0)

    return TestResult(
        name="Kupiec proportion of failures",
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, 1)),
        degrees_of_freedom=1,
        n_obs=n_obs,
        n_exceedances=n_breaches,
        expected_exceedances=n_obs * p,
    )


def _as_breach_series(breaches: pd.Series) -> pd.Series:
    """Validate a boolean exceedance indicator."""
    if not isinstance(breaches, pd.Series):
        raise TypeError("Expected a boolean Series of exceedances, as `exceedances` returns")
    if breaches.dtype != bool:
        raise TypeError(f"Exceedances must be boolean, got dtype {breaches.dtype}")
    if len(breaches) < 2:
        raise ValueError("At least two days are needed to look at clustering")
    return breaches


def _unique_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    """Collapse thousands of identical warnings down to their text."""
    seen: list[str] = []
    for entry in caught:
        text = str(entry.message)
        if text not in seen:
            seen.append(text)
    return seen
