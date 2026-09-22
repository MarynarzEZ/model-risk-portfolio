"""Tests for the covariance and correlation estimators.

The EWMA estimators are checked against a weighted sum written out by
hand in the test, not against a pandas `ewm` call, so a change in
pandas' seeding or bias correction cannot silently redefine what this
package means by EWMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.covariance import (
    DEFAULT_LAMBDA,
    correlation,
    ewma_covariance,
    ewma_volatility,
    is_positive_semidefinite,
    portfolio_volatility,
    rolling_correlation,
    sample_covariance,
)


@pytest.fixture
def three_assets() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    cov = np.array(
        [
            [0.012**2, 0.6 * 0.012 * 0.018, 0.2 * 0.012 * 0.009],
            [0.6 * 0.012 * 0.018, 0.018**2, 0.1 * 0.018 * 0.009],
            [0.2 * 0.012 * 0.009, 0.1 * 0.018 * 0.009, 0.009**2],
        ]
    )
    draws = rng.multivariate_normal([0.0, 0.0, 0.0], cov, size=1500)
    index = pd.bdate_range("2018-01-01", periods=1500, name="date")
    return pd.DataFrame(draws, index=index, columns=["AAA", "BBB", "CCC"])


# --- sample covariance and correlation -------------------------------


def test_sample_covariance_is_symmetric_and_psd(three_assets):
    cov = sample_covariance(three_assets)

    assert cov.shape == (3, 3)
    assert list(cov.index) == list(cov.columns) == ["AAA", "BBB", "CCC"]
    assert is_positive_semidefinite(cov)


def test_sample_covariance_diagonal_is_the_variance(three_assets):
    cov = sample_covariance(three_assets)

    for name in three_assets.columns:
        assert cov.loc[name, name] == pytest.approx(three_assets[name].var(ddof=1))


def test_annualising_multiplies_variance_by_the_period_count(three_assets):
    daily = sample_covariance(three_assets)
    yearly = sample_covariance(three_assets, annualise=True)

    pd.testing.assert_frame_equal(yearly, daily * 252)


def test_correlation_recovers_the_true_structure(three_assets):
    corr = correlation(three_assets)

    assert np.allclose(np.diag(corr), 1.0)
    assert corr.loc["AAA", "BBB"] == pytest.approx(0.6, abs=0.05)
    assert corr.loc["AAA", "CCC"] == pytest.approx(0.2, abs=0.05)


# --- EWMA ------------------------------------------------------------


def test_ewma_weights_sum_to_one_and_favour_the_recent():
    """Checked against the weighted sum written out longhand."""
    values = np.array([[0.02], [-0.01], [0.03]])
    frame = pd.DataFrame(values, columns=["AAA"])
    lam = 0.9

    result = ewma_covariance(frame, lam=lam)

    raw = np.array([lam**2, lam**1, lam**0])
    weights = raw / raw.sum()
    expected = float((weights * values.ravel() ** 2).sum())
    assert result.loc["AAA", "AAA"] == pytest.approx(expected)


def test_ewma_approaches_the_sample_covariance_as_lambda_approaches_one(three_assets):
    """With equal weights the two estimators are the same thing."""
    ewma = ewma_covariance(three_assets, lam=0.99999, demean=True)
    # Sample covariance divides by n-1, EWMA weights sum to 1 (i.e. n),
    # so the two differ by exactly that factor.
    sample = sample_covariance(three_assets) * (len(three_assets) - 1) / len(three_assets)

    assert np.allclose(ewma.to_numpy(), sample.to_numpy(), rtol=1e-3)


def test_ewma_reacts_faster_to_a_volatility_jump():
    """The reason to use it at all."""
    calm = np.full(200, 0.001)
    storm = np.full(20, 0.05)
    series = pd.DataFrame({"AAA": np.concatenate([calm, storm])})

    ewma = ewma_covariance(series, lam=DEFAULT_LAMBDA).loc["AAA", "AAA"]
    equal_weight = sample_covariance(series).loc["AAA", "AAA"]

    assert ewma > equal_weight


def test_ewma_covariance_is_positive_semidefinite(three_assets):
    """True by construction; asserted because the model depends on it."""
    assert is_positive_semidefinite(ewma_covariance(three_assets))


def test_ewma_rejects_an_impossible_lambda(three_assets):
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ewma_covariance(three_assets, lam=bad)


def test_ewma_volatility_uses_only_past_returns():
    """A forecast that peeked at today would be useless in a backtest."""
    quiet = pd.Series([0.001] * 50 + [0.20] + [0.001] * 5)

    vol = ewma_volatility(quiet, lam=0.94)

    # The shock is at position 50. The estimate *for* that day cannot
    # know about it; the next day's must.
    assert vol.iloc[50] < 0.01
    assert vol.iloc[51] > 0.04


def test_ewma_volatility_keeps_the_input_shape(three_assets):
    as_frame = ewma_volatility(three_assets)
    as_series = ewma_volatility(three_assets["AAA"])

    assert isinstance(as_frame, pd.DataFrame)
    assert list(as_frame.columns) == ["AAA", "BBB", "CCC"]
    assert isinstance(as_series, pd.Series)
    assert as_series.name == "AAA"
    assert as_frame.index.equals(three_assets.index)


def test_ewma_volatility_is_positive_and_finite(three_assets):
    vol = ewma_volatility(three_assets)

    assert (vol > 0).all(axis=None)
    assert np.isfinite(vol.to_numpy()).all()


def test_annualising_volatility_scales_by_root_252(three_assets):
    daily = ewma_volatility(three_assets)
    yearly = ewma_volatility(three_assets, annualise=True)

    pd.testing.assert_frame_equal(yearly, daily * np.sqrt(252))


# --- rolling correlation ---------------------------------------------


def test_rolling_correlation_detects_a_regime_change():
    """Independent assets that suddenly move as one."""
    rng = np.random.default_rng(11)
    independent = rng.normal(0, 0.01, size=(300, 2))
    shared = rng.normal(0, 0.01, size=300)
    together = np.column_stack([shared, shared + rng.normal(0, 0.001, size=300)])

    frame = pd.DataFrame(np.vstack([independent, together]), columns=["AAA", "BBB"])
    rolling = rolling_correlation(frame, window=60)

    assert rolling.iloc[100] < 0.3, "should be near zero while the assets are independent"
    assert rolling.iloc[-1] > 0.9, "should be near one once they move together"


def test_rolling_correlation_length_and_alignment(three_assets):
    rolling = rolling_correlation(three_assets, window=60)

    assert len(rolling) == len(three_assets) - 59
    assert rolling.index[0] == three_assets.index[59]
    assert rolling.index[-1] == three_assets.index[-1]
    assert rolling.name == "average_correlation"


def test_rolling_correlation_of_identical_assets_is_one():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.01, size=100)
    twins = pd.DataFrame({"AAA": base, "BBB": base})

    assert rolling_correlation(twins, window=30).iloc[-1] == pytest.approx(1.0)


def test_rolling_correlation_needs_two_assets(three_assets):
    with pytest.raises(ValueError, match="At least 2 assets"):
        rolling_correlation(three_assets[["AAA"]], window=30)


def test_window_longer_than_the_sample_raises(three_assets):
    with pytest.raises(ValueError, match="longer than"):
        rolling_correlation(three_assets, window=5000)


# --- portfolio volatility --------------------------------------------


def test_diversification_lowers_portfolio_volatility(three_assets):
    cov = sample_covariance(three_assets)
    weights = {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}

    combined = portfolio_volatility(weights, cov)
    weighted_average = sum(w * np.sqrt(cov.loc[n, n]) for n, w in weights.items())

    assert combined < weighted_average


def test_portfolio_volatility_of_a_single_asset_is_its_own(three_assets):
    cov = sample_covariance(three_assets)

    assert portfolio_volatility({"AAA": 1.0, "BBB": 0.0, "CCC": 0.0}, cov) == pytest.approx(
        np.sqrt(cov.loc["AAA", "AAA"])
    )


def test_perfectly_correlated_assets_get_no_diversification():
    """The limiting case: risk is exactly the weighted average."""
    sigma = 0.02
    cov = pd.DataFrame(
        [[sigma**2, sigma**2], [sigma**2, sigma**2]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )

    assert portfolio_volatility({"AAA": 0.5, "BBB": 0.5}, cov) == pytest.approx(sigma)


def test_portfolio_volatility_matches_hand_arithmetic():
    cov = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0009]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    # 0.5^2*0.0004 + 0.5^2*0.0009 + 2*0.5*0.5*0.0001 = 0.000375
    assert portfolio_volatility({"AAA": 0.5, "BBB": 0.5}, cov) == pytest.approx(np.sqrt(0.000375))


def test_weights_are_matched_to_the_matrix_by_name():
    cov = pd.DataFrame(
        [[0.0004, 0.0], [0.0, 0.0009]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    one_way = portfolio_volatility({"AAA": 0.25, "BBB": 0.75}, cov)
    other_way = portfolio_volatility({"BBB": 0.75, "AAA": 0.25}, cov)

    assert one_way == pytest.approx(other_way)


# --- PSD check -------------------------------------------------------


def test_psd_accepts_a_real_covariance_matrix(three_assets):
    assert is_positive_semidefinite(sample_covariance(three_assets))


def test_psd_rejects_a_matrix_with_a_negative_eigenvalue():
    broken = pd.DataFrame(
        [[0.0004, 0.01], [0.01, 0.0009]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    assert not is_positive_semidefinite(broken)


def test_psd_accepts_a_singular_matrix():
    """Two identical assets are degenerate, not invalid."""
    singular = pd.DataFrame(
        [[0.0004, 0.0004], [0.0004, 0.0004]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    assert is_positive_semidefinite(singular)


def test_psd_rejects_an_asymmetric_matrix():
    assert not is_positive_semidefinite(np.array([[1.0, 0.5], [0.2, 1.0]]))


def test_psd_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        is_positive_semidefinite(np.ones((2, 3)))


# --- input handling --------------------------------------------------


def test_a_series_is_rejected_with_a_useful_message(three_assets):
    with pytest.raises(TypeError, match="at least two assets"):
        sample_covariance(three_assets["AAA"])


def test_more_assets_than_days_warns(three_assets):
    with pytest.warns(UserWarning, match="singular"):
        sample_covariance(three_assets.iloc[:2])
