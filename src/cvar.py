"""Conditional VaR, also called Expected Shortfall.

VaR says how bad a day has to be before it counts as bad. It says
nothing about what happens once you are past that point -- a portfolio
whose worst 1% of days lose 3% and one whose worst 1% lose 30% can have
identical VaR. Expected Shortfall is the average loss *given* that VaR
was breached, so it reads the shape of the tail rather than one point
on it.

That difference is why the Basel Committee moved market risk capital
from 99% VaR to 97.5% ES in FRTB: ES is sub-additive, meaning the ES of
a combined portfolio can never exceed the sum of its parts. VaR is not,
so under VaR a desk can sometimes reduce measured risk by splitting a
position in two -- an artefact of the measure, not a real hedge.

Same conventions as `var`: a positive number is a loss, input is simple
returns of the aggregated portfolio, and ES is always at least as large
as the VaR at the same confidence level (the tests assert this).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.var import (
    _as_sample,
    _check_confidence,
    _fit_student_t,
    _simulate_portfolio,
    historical_var,
)


def historical_cvar(returns: pd.Series | Sequence[float], confidence: float = 0.99) -> float:
    """Mean of the observed returns at or below the historical VaR.

    Inherits the strength and the weakness of `historical_var`: no
    distribution is assumed, and nothing worse than the worst observed
    day can come out. At 99% on a single year of data this is the
    average of two or three observations, which is an estimate in name
    only -- `var.historical_var` warns about exactly that.
    """
    sample = _as_sample(returns)
    _check_confidence(confidence)

    var = historical_var(returns, confidence)
    # `historical_var` uses the `lower` quantile, so -var is an
    # observation that actually occurred and the tail is never empty.
    tail = sample[sample <= -var]
    return float(-tail.mean())


def normal_cvar(returns: pd.Series | Sequence[float], confidence: float = 0.99) -> float:
    """Expected Shortfall under a fitted normal, in closed form.

    ES = -mu + sigma * phi(z) / (1 - confidence), with z the standard
    normal quantile at 1 - confidence.

    Computed analytically rather than by averaging simulated draws: the
    closed form has no simulation error, and having one method that is
    exact makes it possible to test the Monte Carlo estimator against
    something other than itself.
    """
    sample = _as_sample(returns)
    _check_confidence(confidence)

    mu = float(np.mean(sample))
    sigma = float(np.std(sample, ddof=1))
    tail_probability = 1.0 - confidence

    z = stats.norm.ppf(tail_probability)
    shortfall = mu - sigma * stats.norm.pdf(z) / tail_probability
    return float(-shortfall)


def student_t_cvar(returns: pd.Series | Sequence[float], confidence: float = 0.99) -> float:
    """Expected Shortfall under a fitted Student's t, in closed form.

    For a standardised t with `df` degrees of freedom the tail mean is

        f(t_q) / (1 - c) * (df + t_q**2) / (df - 1)

    where t_q is the t quantile at 1 - c. The `(df + t_q**2)/(df - 1)`
    factor is what makes the t's tail heavier than the normal's: it
    grows as df falls, and diverges as df approaches 1, where the mean
    of the distribution stops existing and an average loss is no longer
    a meaningful thing to quote.
    """
    sample = _as_sample(returns)
    _check_confidence(confidence)

    df, loc, scale = _fit_student_t(sample)
    if df <= 1.0:
        raise ValueError(
            f"Fitted {df:.2f} degrees of freedom. A Student's t with one degree of "
            f"freedom or fewer has no finite mean, so its Expected Shortfall is "
            f"infinite. Use historical_cvar, or a longer sample."
        )

    tail_probability = 1.0 - confidence
    t_q = stats.t.ppf(tail_probability, df)
    standardised = stats.t.pdf(t_q, df) / tail_probability * (df + t_q**2) / (df - 1.0)

    # Undo the location-scale transform: the formula above is for the
    # standardised t, the fit was to the returns themselves.
    return float(-loc + scale * standardised)


def monte_carlo_cvar(
    returns: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
    confidence: float = 0.99,
    *,
    n_paths: int = 100_000,
    seed: int | None = None,
    distribution: str = "normal",
) -> float:
    """Expected Shortfall from simulated portfolio returns.

    Draws the same scenarios as `var.monte_carlo_var` and averages the
    ones past the simulated VaR. Pass the same `seed` to both if you
    need a VaR and an ES that describe one simulated sample; with
    different seeds they are two estimates of the same quantities and
    will disagree in the third decimal.

    Simulation is where ES earns its keep: unlike the historical
    estimator it can produce a loss larger than anything in the sample,
    because the fitted distribution keeps going past the last
    observation.
    """
    _check_confidence(confidence)
    simulated = _simulate_portfolio(
        returns,
        weights,
        n_paths=n_paths,
        seed=seed,
        distribution=distribution,
    )
    cutoff = np.quantile(simulated, 1.0 - confidence, method="lower")
    tail = simulated[simulated <= cutoff]
    return float(-tail.mean())
