"""Tests for the VaR backtests.

The likelihood ratio statistics are checked against the cases where the
algebra collapses to something you can verify by hand: a perfectly
calibrated model gives exactly zero, and a model whose exceedances are
deterministic gives a statistic that can be worked out from the counts.
Everything else is checked as a direction -- a worse model must score
worse -- which is what the tests are for in the first place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    BacktestReport,
    TestResult,
    basel_traffic_light,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    exceedances,
    kupiec_pof,
    rolling_var_backtest,
)
from src.var import historical_var, normal_var


def returns_with_breaches(n_obs: int, breach_positions: list[int], var: float = 0.02) -> pd.Series:
    """A return series that breaches `var` exactly where asked."""
    values = np.full(n_obs, 0.001)
    values[breach_positions] = -(var + 0.01)
    index = pd.bdate_range("2020-01-01", periods=n_obs, name="date")
    return pd.Series(values, index=index, name="portfolio")


@pytest.fixture
def realistic_returns() -> pd.Series:
    """Ten years of returns with volatility clustering built in.

    Deliberately not iid: a constant-volatility sample would make every
    VaR model look good and the backtests would have nothing to find.
    """
    rng = np.random.default_rng(404)
    n = 2600
    volatility = np.empty(n)
    volatility[0] = 0.01
    for t in range(1, n):
        # A simple GARCH-like recursion: today's volatility remembers
        # yesterday's shock.
        volatility[t] = np.sqrt(
            0.00001 + 0.88 * volatility[t - 1] ** 2 + 0.10 * (volatility[t - 1] * rng.normal()) ** 2
        )
    index = pd.bdate_range("2014-01-01", periods=n, name="date")
    return pd.Series(rng.normal(0.0003, volatility), index=index, name="portfolio")


# --- exceedances -----------------------------------------------------


def test_exceedances_flag_losses_worse_than_var():
    returns = returns_with_breaches(100, [10, 50, 90], var=0.02)

    breaches = exceedances(returns, 0.02)

    assert breaches.sum() == 3
    assert list(breaches[breaches].index) == list(returns.index[[10, 50, 90]])


def test_a_loss_exactly_equal_to_var_is_not_a_breach():
    """The quantile is the boundary, and the boundary is not past it."""
    index = pd.bdate_range("2020-01-01", periods=3, name="date")
    returns = pd.Series([-0.02, -0.0201, 0.01], index=index)

    breaches = exceedances(returns, 0.02)

    assert not breaches.iloc[0]
    assert breaches.iloc[1]


def test_exceedances_accept_a_time_varying_var():
    index = pd.bdate_range("2020-01-01", periods=4, name="date")
    returns = pd.Series([-0.03, -0.03, -0.03, -0.03], index=index)
    var = pd.Series([0.02, 0.05, 0.02, 0.05], index=index)

    breaches = exceedances(returns, var)

    assert list(breaches) == [True, False, True, False]


def test_a_misaligned_var_series_raises():
    returns = returns_with_breaches(10, [5])
    var = pd.Series([0.02] * 5, index=returns.index[:5])

    with pytest.raises(ValueError, match="missing 5 of the 10 dates"):
        exceedances(returns, var)


# --- Kupiec ----------------------------------------------------------


def test_kupiec_is_zero_when_the_rate_is_exactly_right():
    """1000 days at 99%, exactly 10 breaches: nothing to explain."""
    returns = returns_with_breaches(1000, list(range(0, 1000, 100)))

    result = kupiec_pof(returns, 0.02, 0.99)

    assert result.n_exceedances == 10
    assert result.expected_exceedances == pytest.approx(10.0)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)
    assert not result.rejects()


def test_kupiec_rejects_a_model_that_breaches_far_too_often():
    returns = returns_with_breaches(1000, list(range(0, 1000, 10)))  # 100 breaches

    result = kupiec_pof(returns, 0.02, 0.99)

    assert result.n_exceedances == 100
    assert result.rejects(0.01)
    assert result.p_value < 1e-20


def test_kupiec_rejects_a_model_that_never_breaches():
    """Too few exceedances is also a failure: capital sits idle."""
    calm = pd.Series(np.full(2000, 0.001), index=pd.bdate_range("2020-01-01", periods=2000))

    result = kupiec_pof(calm, 0.05, 0.99)

    assert result.n_exceedances == 0
    assert result.rejects()


def test_kupiec_handles_zero_exceedances_without_blowing_up():
    """0 * log(0) must be 0, not NaN."""
    calm = pd.Series(np.full(100, 0.001), index=pd.bdate_range("2020-01-01", periods=100))

    result = kupiec_pof(calm, 0.05, 0.99)

    assert np.isfinite(result.statistic)
    assert np.isfinite(result.p_value)


def test_a_mild_miss_is_not_rejected():
    """One year at 99%: 2 breaches against 2.5 expected proves nothing."""
    returns = returns_with_breaches(250, [40, 200])

    assert not kupiec_pof(returns, 0.02, 0.99).rejects()


# --- Christoffersen independence -------------------------------------


def test_independence_passes_when_breaches_are_spread_out():
    returns = returns_with_breaches(1000, list(range(0, 1000, 100)))

    result = christoffersen_independence(exceedances(returns, 0.02))

    assert not result.rejects()


def test_independence_rejects_clustered_breaches():
    """The failure that matters: right count, all in one week."""
    returns = returns_with_breaches(1000, list(range(500, 510)))

    result = christoffersen_independence(exceedances(returns, 0.02))

    assert result.n_exceedances == 10
    assert result.rejects(0.01)


def test_clustering_is_invisible_to_kupiec():
    """Why the two tests both exist."""
    spread = returns_with_breaches(1000, list(range(0, 1000, 100)))
    clustered = returns_with_breaches(1000, list(range(500, 510)))

    assert kupiec_pof(spread, 0.02, 0.99).statistic == pytest.approx(
        kupiec_pof(clustered, 0.02, 0.99).statistic
    )
    assert christoffersen_independence(exceedances(clustered, 0.02)).statistic > (
        christoffersen_independence(exceedances(spread, 0.02)).statistic
    )


def test_independence_handles_no_breaches_at_all():
    calm = pd.Series(np.full(100, 0.001), index=pd.bdate_range("2020-01-01", periods=100))

    result = christoffersen_independence(exceedances(calm, 0.05))

    assert result.statistic == pytest.approx(0.0)
    assert np.isfinite(result.p_value)


def test_independence_rejects_a_non_boolean_series():
    with pytest.raises(TypeError, match="must be boolean"):
        christoffersen_independence(pd.Series([0.1, 0.2, 0.3]))


# --- conditional coverage --------------------------------------------


def test_conditional_coverage_is_the_sum_of_its_parts():
    returns = returns_with_breaches(1000, list(range(500, 510)))

    pof = kupiec_pof(returns, 0.02, 0.99)
    independence = christoffersen_independence(exceedances(returns, 0.02))
    joint = christoffersen_conditional_coverage(returns, 0.02, 0.99)

    assert joint.statistic == pytest.approx(pof.statistic + independence.statistic)
    assert joint.degrees_of_freedom == 2


def test_conditional_coverage_catches_what_either_half_catches():
    clustered = returns_with_breaches(1000, list(range(500, 510)))
    too_many = returns_with_breaches(1000, list(range(0, 1000, 10)))

    assert christoffersen_conditional_coverage(clustered, 0.02, 0.99).rejects(0.01)
    assert christoffersen_conditional_coverage(too_many, 0.02, 0.99).rejects(0.01)


# --- Basel traffic light ---------------------------------------------


def test_basel_zones_at_the_published_250_day_thresholds():
    """Green to 4, yellow 5 to 9, red from 10. The published table."""
    assert basel_traffic_light(0, 250) == "green"
    assert basel_traffic_light(4, 250) == "green"
    assert basel_traffic_light(5, 250) == "yellow"
    assert basel_traffic_light(9, 250) == "yellow"
    assert basel_traffic_light(10, 250) == "red"
    assert basel_traffic_light(20, 250) == "red"


def test_basel_is_one_sided():
    """Too few breaches stays green, however badly calibrated."""
    assert basel_traffic_light(0, 10_000) == "green"


def test_basel_rejects_impossible_counts():
    with pytest.raises(ValueError, match="impossible"):
        basel_traffic_light(300, 250)


# --- rolling backtest ------------------------------------------------


def test_rolling_backtest_is_out_of_sample(realistic_returns):
    report = rolling_var_backtest(realistic_returns, window=250, confidence=0.99)

    assert len(report.var) == len(realistic_returns) - 250
    assert report.var.index[0] == realistic_returns.index[250]
    assert report.returns.index.equals(report.var.index)


def test_rolling_backtest_returns_a_full_report(realistic_returns):
    report = rolling_var_backtest(realistic_returns, window=250)

    assert isinstance(report, BacktestReport)
    assert isinstance(report.kupiec, TestResult)
    assert report.basel_zone in {"green", "yellow", "red"}
    assert 0.0 <= report.exceedance_rate <= 1.0
    assert set(report.summary().index) == {
        "Kupiec proportion of failures",
        "Christoffersen independence",
        "Christoffersen conditional coverage",
    }


def test_rolling_backtest_breach_rate_is_in_the_right_ballpark(realistic_returns):
    """A working model breaches near 1% of the time, not 5% or 0%."""
    report = rolling_var_backtest(realistic_returns, window=250, confidence=0.99)

    assert 0.002 < report.exceedance_rate < 0.04


def test_breach_dates_line_up_with_the_flags(realistic_returns):
    report = rolling_var_backtest(realistic_returns, window=250)

    assert len(report.breach_dates) == int(report.breaches.sum())
    for date in report.breach_dates:
        assert report.returns[date] < -report.var[date]


def test_the_estimator_is_swappable(realistic_returns):
    """The experiment the module exists to make possible."""
    historical = rolling_var_backtest(realistic_returns, historical_var, window=250)
    parametric = rolling_var_backtest(realistic_returns, normal_var, window=250)

    assert len(historical.var) == len(parametric.var)
    assert not historical.var.equals(parametric.var)


def test_a_deliberately_broken_model_is_caught(realistic_returns):
    """VaR at 80% tested as if it were 99% should fail loudly."""
    too_low = pd.Series(
        [historical_var(realistic_returns.iloc[:250], 0.80)] * (len(realistic_returns) - 250),
        index=realistic_returns.index[250:],
    )
    tested = realistic_returns.iloc[250:]

    assert kupiec_pof(tested, too_low, 0.99).rejects(0.01)


def test_thin_tail_warning_is_raised_once_not_thousands_of_times(realistic_returns):
    with pytest.warns(UserWarning, match="raised on each of") as caught:
        rolling_var_backtest(realistic_returns, window=250, confidence=0.99)

    assert len(caught) == 1


def test_window_longer_than_the_sample_raises():
    short = pd.Series(np.full(100, 0.001), index=pd.bdate_range("2020-01-01", periods=100))

    with pytest.raises(ValueError, match="Need more than 250 returns"):
        rolling_var_backtest(short, window=250)


def test_a_tiny_window_is_rejected(realistic_returns):
    with pytest.raises(ValueError, match="too short"):
        rolling_var_backtest(realistic_returns, window=10)


# --- result objects --------------------------------------------------


def test_rejects_validates_its_threshold():
    returns = returns_with_breaches(1000, list(range(0, 1000, 100)))
    result = kupiec_pof(returns, 0.02, 0.99)

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        result.rejects(1.5)


def test_result_prints_the_numbers_that_matter():
    returns = returns_with_breaches(1000, list(range(0, 1000, 100)))

    text = str(kupiec_pof(returns, 0.02, 0.99))

    assert "10 exceedances" in text
    assert "10.0 expected" in text
