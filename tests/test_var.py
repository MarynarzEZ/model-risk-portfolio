"""Tests for the VaR estimators.

Where a closed form exists the test checks against it, not against
another call into the same module. The Monte Carlo estimator is checked
against the analytic normal it is simulating, which is the only way to
tell a correct simulation from a confidently wrong one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.var import (
    annualised_volatility,
    historical_var,
    monte_carlo_var,
    normal_var,
    scale_var,
    student_t_var,
    var_summary,
)


@pytest.fixture
def normal_sample() -> pd.Series:
    """2000 draws from a known normal, so the true VaR is known too."""
    rng = np.random.default_rng(20240101)
    index = pd.bdate_range("2016-01-01", periods=2000, name="date")
    return pd.Series(rng.normal(0.0005, 0.012, size=2000), index=index, name="portfolio")


@pytest.fixture
def two_assets() -> pd.DataFrame:
    """Two correlated assets, for the multivariate simulation path."""
    rng = np.random.default_rng(7)
    cov = np.array([[0.012**2, 0.6 * 0.012 * 0.018], [0.6 * 0.012 * 0.018, 0.018**2]])
    draws = rng.multivariate_normal([0.0004, 0.0006], cov, size=2000)
    index = pd.bdate_range("2016-01-01", periods=2000, name="date")
    return pd.DataFrame(draws, index=index, columns=["AAA", "BBB"])


# --- sign convention -------------------------------------------------


def test_var_is_a_positive_loss(normal_sample):
    """The convention the whole package rests on."""
    for estimate in (
        historical_var(normal_sample),
        normal_var(normal_sample),
        student_t_var(normal_sample),
        monte_carlo_var(normal_sample, seed=1),
    ):
        assert estimate > 0


def test_a_portfolio_that_only_gains_has_negative_var():
    """Not an error: the bad tail was still a gain, and we say so."""
    only_gains = pd.Series(np.linspace(0.01, 0.05, 500))

    assert normal_var(only_gains, 0.99) < 0
    assert historical_var(only_gains, 0.99) < 0


def test_a_bigger_loss_means_a_bigger_var():
    calm = pd.Series(np.linspace(-0.01, 0.01, 1000))
    wild = calm * 3

    assert normal_var(wild) > normal_var(calm)
    assert historical_var(wild) > historical_var(calm)


# --- historical ------------------------------------------------------


def test_historical_var_is_an_observation_from_the_sample():
    """`lower` interpolation must never invent a return."""
    sample = pd.Series(np.linspace(-0.10, 0.10, 101))

    var = historical_var(sample, 0.95)

    assert -var in sample.to_numpy()


def test_historical_var_matches_hand_counted_quantile():
    # 100 returns from -1.00 to -0.01 in steps of 0.01, sorted ascending.
    # At 99% the cutoff index is floor(0.01 * 99) = 0, the worst one.
    sample = pd.Series(np.round(np.arange(-1.00, 0.0, 0.01), 2))

    assert historical_var(sample, 0.99) == pytest.approx(1.00)


def test_historical_var_cannot_exceed_the_worst_day(normal_sample):
    assert historical_var(normal_sample, 0.999) <= -normal_sample.min()


def test_thin_tail_warns():
    sample = pd.Series(np.linspace(-0.05, 0.05, 100))

    with pytest.warns(UserWarning, match="fall in the"):
        historical_var(sample, 0.99)


# --- normal ----------------------------------------------------------


def test_normal_var_matches_the_closed_form(normal_sample):
    mu = normal_sample.mean()
    sigma = normal_sample.std(ddof=1)

    expected = -(mu + sigma * stats.norm.ppf(0.01))

    assert normal_var(normal_sample, 0.99) == pytest.approx(expected)


def test_normal_var_recovers_the_true_parameters(normal_sample):
    """With 2000 draws the estimate should be close to the truth."""
    truth = -(0.0005 + 0.012 * stats.norm.ppf(0.01))

    assert normal_var(normal_sample, 0.99) == pytest.approx(truth, rel=0.05)


def test_higher_confidence_means_higher_var(normal_sample):
    assert normal_var(normal_sample, 0.999) > normal_var(normal_sample, 0.99)
    assert normal_var(normal_sample, 0.99) > normal_var(normal_sample, 0.95)


# --- Student's t -----------------------------------------------------


def test_student_t_finds_a_fat_tail_the_normal_misses():
    """The whole point of the t estimator."""
    rng = np.random.default_rng(3)
    fat = pd.Series(stats.t.rvs(4, size=4000, random_state=rng) * 0.01)

    assert student_t_var(fat, 0.99) > normal_var(fat, 0.99)


def test_student_t_collapses_to_the_normal_on_normal_data(normal_sample):
    """With no excess kurtosis the extra parameter should not matter."""
    assert student_t_var(normal_sample, 0.99) == pytest.approx(
        normal_var(normal_sample, 0.99), rel=0.10
    )


def test_tiny_sample_warns_about_the_fit():
    sample = pd.Series(np.linspace(-0.05, 0.05, 20))

    with pytest.warns(UserWarning, match="degrees of freedom"):
        student_t_var(sample, 0.95)


# --- Monte Carlo -----------------------------------------------------


def test_monte_carlo_converges_to_the_analytic_normal(normal_sample):
    simulated = monte_carlo_var(normal_sample, confidence=0.99, n_paths=200_000, seed=42)

    assert simulated == pytest.approx(normal_var(normal_sample, 0.99), rel=0.02)


def test_monte_carlo_is_reproducible_with_a_seed(normal_sample):
    first = monte_carlo_var(normal_sample, seed=11, n_paths=20_000)
    second = monte_carlo_var(normal_sample, seed=11, n_paths=20_000)

    assert first == second


def test_monte_carlo_differs_without_a_seed(normal_sample):
    first = monte_carlo_var(normal_sample, n_paths=20_000)
    second = monte_carlo_var(normal_sample, n_paths=20_000)

    assert first != second


def test_multivariate_simulation_prices_diversification(two_assets):
    """A correlated pair must be less risky than the sum of its parts."""
    weights = {"AAA": 0.5, "BBB": 0.5}

    combined = monte_carlo_var(two_assets, weights, seed=5, n_paths=100_000)
    standalone = [
        monte_carlo_var(two_assets[name], seed=5, n_paths=100_000) for name in ("AAA", "BBB")
    ]

    assert combined < 0.5 * standalone[0] + 0.5 * standalone[1]


def test_multivariate_simulation_matches_the_aggregated_series(two_assets):
    """Simulating assets then weighting must agree with weighting first."""
    from src.returns import portfolio_returns

    weights = {"AAA": 0.4, "BBB": 0.6}
    aggregated = portfolio_returns(two_assets, weights)

    from_assets = monte_carlo_var(two_assets, weights, seed=9, n_paths=200_000)
    from_series = normal_var(aggregated, 0.99)

    assert from_assets == pytest.approx(from_series, rel=0.03)


def test_t_innovations_give_a_fatter_tail_than_normal_ones():
    rng = np.random.default_rng(13)
    fat = pd.Series(stats.t.rvs(4, size=4000, random_state=rng) * 0.01)

    normal_draws = monte_carlo_var(fat, seed=2, n_paths=100_000, distribution="normal")
    t_draws = monte_carlo_var(fat, seed=2, n_paths=100_000, distribution="t")

    assert t_draws > normal_draws


def test_unknown_distribution_raises(normal_sample):
    with pytest.raises(ValueError, match="normal' or 't'"):
        monte_carlo_var(normal_sample, distribution="cauchy")


def test_too_few_paths_raises(normal_sample):
    with pytest.raises(ValueError, match="at least 1000"):
        monte_carlo_var(normal_sample, n_paths=100)


def test_weights_are_required_for_multiple_assets(two_assets):
    with pytest.raises(ValueError, match="Weights are required"):
        monte_carlo_var(two_assets)


def test_collinear_assets_still_simulate():
    """A singular covariance matrix must not crash the simulation."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, size=500)
    twins = pd.DataFrame({"AAA": base, "BBB": base})  # perfectly correlated

    var = monte_carlo_var(twins, {"AAA": 0.5, "BBB": 0.5}, seed=4, n_paths=20_000)

    assert var > 0


# --- scaling and helpers ---------------------------------------------


def test_square_root_of_time_scaling():
    assert scale_var(0.02, 4) == pytest.approx(0.04)
    assert scale_var(0.02, 1) == pytest.approx(0.02)


def test_scaling_rejects_a_nonsense_horizon():
    with pytest.raises(ValueError, match="positive whole number"):
        scale_var(0.02, 0)
    with pytest.raises(ValueError, match="positive whole number"):
        scale_var(0.02, 2.5)


def test_annualised_volatility_scales_by_root_252(normal_sample):
    expected = normal_sample.std(ddof=1) * np.sqrt(252)

    assert annualised_volatility(normal_sample) == pytest.approx(expected)


# --- input handling --------------------------------------------------


def test_a_dataframe_is_rejected_with_a_useful_message(two_assets):
    with pytest.raises(TypeError, match="portfolio_returns"):
        normal_var(two_assets)


def test_tail_probability_passed_as_confidence_is_rejected(normal_sample):
    with pytest.raises(ValueError, match="not the tail probability"):
        normal_var(normal_sample, 0.01)


def test_confidence_of_one_is_rejected(normal_sample):
    with pytest.raises(ValueError, match="between 0.5 and 1"):
        normal_var(normal_sample, 1.0)


def test_non_finite_returns_are_rejected():
    broken = pd.Series([0.01, np.inf, -0.02, 0.005] * 100)

    with pytest.raises(ValueError, match="NaN or infinity"):
        normal_var(broken)


def test_missing_values_are_dropped_not_rejected(normal_sample):
    with_gaps = normal_sample.copy()
    with_gaps.iloc[[3, 50, 900]] = np.nan

    assert normal_var(with_gaps) == pytest.approx(normal_var(normal_sample.drop(
        normal_sample.index[[3, 50, 900]]
    )))


def test_too_few_returns_are_rejected():
    with pytest.raises(ValueError, match="At least two returns"):
        normal_var(pd.Series([0.01]))


# --- summary table ---------------------------------------------------


def test_summary_has_a_row_per_method(normal_sample):
    table = var_summary(normal_sample, confidence=0.99, seed=3)

    assert list(table.index) == ["historical", "normal", "student_t", "monte_carlo"]
    assert list(table.columns) == ["var", "expected_shortfall", "es_var_ratio"]


def test_summary_expected_shortfall_always_exceeds_var(normal_sample):
    table = var_summary(normal_sample, confidence=0.99, seed=3)

    assert (table["expected_shortfall"] >= table["var"]).all()
    assert (table["es_var_ratio"] >= 1.0).all()


def test_summary_records_its_inputs(normal_sample):
    table = var_summary(normal_sample, confidence=0.95, seed=3)

    assert table.attrs["confidence"] == 0.95
    assert table.attrs["n_obs"] == len(normal_sample)


def test_summary_accepts_assets_and_weights(two_assets):
    table = var_summary(two_assets, {"AAA": 0.3, "BBB": 0.7}, confidence=0.99, seed=3)

    assert len(table) == 4
    assert (table["var"] > 0).all()
