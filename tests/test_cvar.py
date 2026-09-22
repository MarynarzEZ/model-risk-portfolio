"""Tests for Expected Shortfall.

The two closed-form estimators are checked against numerical integration
of the same fitted distribution. That is the only independent check
available: comparing a formula to itself rewritten proves nothing, and
comparing it to a simulation only shows the two agree to within Monte
Carlo error. If the algebra in `cvar` is wrong, `quad` will disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from src.cvar import historical_cvar, monte_carlo_cvar, normal_cvar, student_t_cvar
from src.var import _fit_student_t, historical_var, monte_carlo_var, normal_var, student_t_var


@pytest.fixture
def normal_sample() -> pd.Series:
    rng = np.random.default_rng(20240101)
    index = pd.bdate_range("2016-01-01", periods=2000, name="date")
    return pd.Series(rng.normal(0.0005, 0.012, size=2000), index=index, name="portfolio")


@pytest.fixture
def fat_tailed_sample() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(stats.t.rvs(4, size=4000, random_state=rng) * 0.01 + 0.0003)


def tail_mean_by_integration(distribution, confidence: float) -> float:
    """E[X | X <= q] computed numerically, returned as a positive loss."""
    tail_probability = 1.0 - confidence
    quantile = distribution.ppf(tail_probability)
    integral, _ = integrate.quad(
        lambda x: x * distribution.pdf(x), -np.inf, quantile, limit=200
    )
    return float(-integral / tail_probability)


# --- the defining property -------------------------------------------


def test_expected_shortfall_never_undershoots_var(normal_sample):
    """ES is the mean of the tail past VaR, so it cannot be smaller."""
    for confidence in (0.90, 0.95, 0.99):
        assert historical_cvar(normal_sample, confidence) >= historical_var(
            normal_sample, confidence
        )
        assert normal_cvar(normal_sample, confidence) >= normal_var(normal_sample, confidence)
        assert student_t_cvar(normal_sample, confidence) >= student_t_var(
            normal_sample, confidence
        )
        assert monte_carlo_cvar(normal_sample, confidence=confidence, seed=7) >= monte_carlo_var(
            normal_sample, confidence=confidence, seed=7
        )


def test_expected_shortfall_is_a_positive_loss(normal_sample):
    assert normal_cvar(normal_sample) > 0
    assert historical_cvar(normal_sample) > 0
    assert student_t_cvar(normal_sample) > 0
    assert monte_carlo_cvar(normal_sample, seed=1) > 0


def test_higher_confidence_means_higher_shortfall(normal_sample):
    assert normal_cvar(normal_sample, 0.99) > normal_cvar(normal_sample, 0.95)
    assert historical_cvar(normal_sample, 0.99) > historical_cvar(normal_sample, 0.95)


# --- closed forms against numerical integration ----------------------


def test_normal_shortfall_matches_numerical_integration(normal_sample):
    mu = float(normal_sample.mean())
    sigma = float(normal_sample.std(ddof=1))
    fitted = stats.norm(loc=mu, scale=sigma)

    assert normal_cvar(normal_sample, 0.99) == pytest.approx(
        tail_mean_by_integration(fitted, 0.99), rel=1e-6
    )


def test_student_t_shortfall_matches_numerical_integration(fat_tailed_sample):
    df, loc, scale = _fit_student_t(fat_tailed_sample.to_numpy())
    fitted = stats.t(df, loc=loc, scale=scale)

    assert student_t_cvar(fat_tailed_sample, 0.99) == pytest.approx(
        tail_mean_by_integration(fitted, 0.99), rel=1e-6
    )


def test_student_t_shortfall_matches_integration_at_other_levels(fat_tailed_sample):
    df, loc, scale = _fit_student_t(fat_tailed_sample.to_numpy())
    fitted = stats.t(df, loc=loc, scale=scale)

    for confidence in (0.90, 0.95, 0.975):
        assert student_t_cvar(fat_tailed_sample, confidence) == pytest.approx(
            tail_mean_by_integration(fitted, confidence), rel=1e-6
        )


# --- how the methods relate to each other ----------------------------


def test_the_t_shortfall_exceeds_the_normal_one_on_fat_tails(fat_tailed_sample):
    """ES is where the fat tail shows up most, because it averages it."""
    assert student_t_cvar(fat_tailed_sample, 0.99) > normal_cvar(fat_tailed_sample, 0.99)


def test_monte_carlo_converges_to_the_analytic_normal(normal_sample):
    simulated = monte_carlo_cvar(normal_sample, confidence=0.99, n_paths=400_000, seed=42)

    assert simulated == pytest.approx(normal_cvar(normal_sample, 0.99), rel=0.03)


def test_historical_shortfall_is_the_mean_of_the_worst_days():
    """Hand-checkable: 100 returns, 99% leaves exactly the worst one."""
    sample = pd.Series(np.round(np.arange(-1.00, 0.0, 0.01), 2))

    with pytest.warns(UserWarning):
        assert historical_cvar(sample, 0.99) == pytest.approx(1.00)


def test_historical_shortfall_averages_the_whole_tail():
    # 1000 points from -0.50 to 0.499; the worst 5% are the 50 lowest,
    # averaging -0.4755.
    sample = pd.Series(np.arange(-500, 500) / 1000.0)

    assert historical_cvar(sample, 0.95) == pytest.approx(0.4755, abs=1e-6)


def test_historical_shortfall_is_capped_by_the_sample(fat_tailed_sample):
    """However far into the tail you ask, the data runs out."""
    assert historical_cvar(fat_tailed_sample, 0.999) <= -fat_tailed_sample.min()


def test_simulation_is_not_capped_by_the_sample(fat_tailed_sample):
    """The reason to simulate at all.

    A fitted distribution has unbounded support, so it produces losses
    worse than any that have happened yet. Whether that shows up in the
    ES at a given confidence level depends on how extreme the sample's
    own worst day was, so the property is asserted on the simulated
    draws rather than on the summary statistic.
    """
    from src.var import _simulate_portfolio

    simulated = _simulate_portfolio(
        fat_tailed_sample, None, n_paths=200_000, seed=8, distribution="t"
    )

    assert simulated.min() < fat_tailed_sample.min()


# --- input handling --------------------------------------------------


def test_a_dataframe_is_rejected(normal_sample):
    frame = pd.DataFrame({"AAA": normal_sample, "BBB": normal_sample})

    with pytest.raises(TypeError, match="portfolio_returns"):
        normal_cvar(frame)


def test_tail_probability_passed_as_confidence_is_rejected(normal_sample):
    with pytest.raises(ValueError, match="not the tail probability"):
        historical_cvar(normal_sample, 0.01)


def test_monte_carlo_shortfall_is_reproducible(normal_sample):
    first = monte_carlo_cvar(normal_sample, seed=21, n_paths=20_000)
    second = monte_carlo_cvar(normal_sample, seed=21, n_paths=20_000)

    assert first == second


def test_multivariate_shortfall_prices_diversification():
    rng = np.random.default_rng(4)
    cov = np.array([[0.012**2, 0.5 * 0.012 * 0.015], [0.5 * 0.012 * 0.015, 0.015**2]])
    assets = pd.DataFrame(
        rng.multivariate_normal([0.0, 0.0], cov, size=2000), columns=["AAA", "BBB"]
    )

    combined = monte_carlo_cvar(assets, {"AAA": 0.5, "BBB": 0.5}, seed=6, n_paths=100_000)
    standalone = [monte_carlo_cvar(assets[c], seed=6, n_paths=100_000) for c in ("AAA", "BBB")]

    assert combined < 0.5 * standalone[0] + 0.5 * standalone[1]
