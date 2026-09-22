"""Tests for the stress testing module.

Prices are built so the answer can be worked out on paper: an asset that
halves over the window contributes exactly -0.5 times its weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import DataError
from src.stress import (
    HISTORICAL_SCENARIOS,
    HistoricalScenario,
    HypotheticalScenario,
    run_historical,
    run_hypothetical,
    stress_table,
    uniform_shock,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Daily prices over 2007-2023, so every built-in scenario fits."""
    index = pd.bdate_range("2006-01-02", "2023-12-29", name="date")
    n = len(index)
    # Deterministic and strictly positive; the exact path does not
    # matter because the tests that check numbers use `flat_prices`.
    drift = np.linspace(0.0, 0.6, n)
    wobble = 0.1 * np.sin(np.linspace(0, 40, n))
    return pd.DataFrame(
        {
            "AAA": 100 * np.exp(drift + wobble),
            "BBB": 50 * np.exp(drift - wobble),
        },
        index=index,
    )


@pytest.fixture
def flat_prices() -> pd.DataFrame:
    """AAA halves over the window, BBB doubles. Hand-checkable."""
    index = pd.bdate_range("2020-02-19", "2020-03-23", name="date")
    n = len(index)
    return pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 50.0, n),
            "BBB": np.linspace(50.0, 100.0, n),
        },
        index=index,
    )


COVID = HistoricalScenario("COVID-19 crash", "2020-02-19", "2020-03-23")


# --- historical replay -----------------------------------------------


def test_historical_return_matches_hand_arithmetic(flat_prices):
    """AAA -50% at weight 0.6, BBB +100% at weight 0.4 => +0.10."""
    result = run_historical(flat_prices, {"AAA": 0.6, "BBB": 0.4}, COVID)

    assert result.contributions["AAA"] == pytest.approx(-0.30)
    assert result.contributions["BBB"] == pytest.approx(0.40)
    assert result.portfolio_return == pytest.approx(0.10)


def test_contributions_add_up_to_the_portfolio_return(prices):
    """Buy-and-hold makes the attribution exact, not approximate."""
    result = run_historical(prices, {"AAA": 0.7, "BBB": 0.3}, COVID)

    assert result.contributions.sum() == pytest.approx(result.portfolio_return)


def test_worst_contributor_accounts_for_position_size():
    """A big fall in a small position is not the biggest problem."""
    index = pd.bdate_range("2020-02-19", "2020-03-23", name="date")
    n = len(index)
    frame = pd.DataFrame(
        {
            "SMALL": np.linspace(100.0, 20.0, n),  # -80%, weight 0.02
            "BIG": np.linspace(100.0, 88.0, n),  # -12%, weight 0.98
        },
        index=index,
    )

    result = run_historical(frame, {"SMALL": 0.02, "BIG": 0.98}, COVID)

    assert result.contributions["SMALL"] == pytest.approx(-0.016)
    assert result.contributions["BIG"] == pytest.approx(-0.1176)
    assert result.worst_contributor == "BIG"


def test_result_records_the_actual_window(flat_prices):
    result = run_historical(flat_prices, {"AAA": 0.5, "BBB": 0.5}, COVID)

    assert result.start == flat_prices.index[0]
    assert result.end == flat_prices.index[-1]
    assert result.n_days == len(flat_prices)


def test_scenario_outside_the_price_history_raises(prices):
    ancient = HistoricalScenario("Black Monday", "1987-10-01", "1987-10-31")

    with pytest.raises(DataError, match="but the prices only cover"):
        run_historical(prices, {"AAA": 0.5, "BBB": 0.5}, ancient)


def test_partially_covered_scenario_raises(flat_prices):
    """A truncated window would understate the loss, so refuse it."""
    longer = HistoricalScenario("Extended", "2020-02-19", "2020-06-30")

    with pytest.raises(DataError, match="but the prices only cover"):
        run_historical(flat_prices, {"AAA": 0.5, "BBB": 0.5}, longer)


def test_weights_must_still_sum_to_one(prices):
    with pytest.raises(ValueError, match="must sum to 1"):
        run_historical(prices, {"AAA": 0.5, "BBB": 0.2}, COVID)


def test_missing_weight_raises(prices):
    with pytest.raises(ValueError, match="No weight given for BBB"):
        run_historical(prices, {"AAA": 1.0}, COVID)


# --- hypothetical shocks ---------------------------------------------


def test_hypothetical_shock_is_the_weighted_sum():
    scenario = HypotheticalScenario("Rates +300bp", {"AAA": -0.25, "BBB": -0.05})

    result = run_hypothetical({"AAA": 0.4, "BBB": 0.6}, scenario)

    assert result.portfolio_return == pytest.approx(0.4 * -0.25 + 0.6 * -0.05)
    assert result.start is None, "an instantaneous shock has no window"


def test_uniform_shock_applies_to_every_holding():
    scenario = uniform_shock("Equities -20%", ["AAA", "BBB", "CCC"], -0.20)

    result = run_hypothetical({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, scenario)

    assert result.portfolio_return == pytest.approx(-0.20)


def test_a_forgotten_asset_raises_rather_than_surviving_untouched():
    scenario = HypotheticalScenario("Partial", {"AAA": -0.30})

    with pytest.raises(ValueError, match="no shock given for BBB"):
        run_hypothetical({"AAA": 0.5, "BBB": 0.5}, scenario)


def test_ticker_case_does_not_matter():
    scenario = HypotheticalScenario("Crash", {"aaa": -0.30, "bbb": -0.10})

    result = run_hypothetical({"AAA": 0.5, "BBB": 0.5}, scenario)

    assert result.portfolio_return == pytest.approx(-0.20)


def test_shocks_cannot_be_mutated_after_construction():
    """A scenario that changed between two runs would be invisible."""
    scenario = HypotheticalScenario("Crash", {"AAA": -0.30})

    with pytest.raises(TypeError):
        scenario.shocks["AAA"] = 0.0  # type: ignore[index]


def test_an_empty_scenario_is_rejected():
    with pytest.raises(ValueError, match="says nothing"):
        HypotheticalScenario("Nothing", {})


def test_a_backwards_window_is_rejected():
    with pytest.raises(ValueError, match="end must be after start"):
        HistoricalScenario("Backwards", "2020-03-23", "2020-02-19")


def test_a_bare_sequence_of_weights_is_rejected():
    scenario = uniform_shock("Equities -20%", ["AAA", "BBB"], -0.20)

    with pytest.raises(TypeError, match="keyed by ticker"):
        run_hypothetical([0.5, 0.5], scenario)


# --- the table -------------------------------------------------------


def test_table_runs_every_covered_scenario(prices):
    table = stress_table(prices, {"AAA": 0.5, "BBB": 0.5})

    assert len(table) == len(HISTORICAL_SCENARIOS)
    assert table.index.name == "scenario"


def test_table_is_sorted_worst_first(prices):
    table = stress_table(prices, {"AAA": 0.5, "BBB": 0.5})

    assert table["portfolio_return"].is_monotonic_increasing


def test_table_skips_scenarios_the_history_does_not_reach(flat_prices):
    """Only COVID fits in a five-week price history."""
    with pytest.warns(UserWarning, match="Skipped"):
        table = stress_table(flat_prices, {"AAA": 0.5, "BBB": 0.5})

    assert list(table.index) == ["COVID-19 crash"]


def test_table_can_be_told_to_refuse_instead_of_skipping(flat_prices):
    with pytest.raises(DataError, match="but the prices only cover"):
        stress_table(flat_prices, {"AAA": 0.5, "BBB": 0.5}, skip_uncovered=False)


def test_table_raises_when_nothing_can_be_run():
    index = pd.bdate_range("2024-01-01", periods=30, name="date")
    recent = pd.DataFrame({"AAA": np.linspace(100, 90, 30)}, index=index)

    with pytest.warns(UserWarning), pytest.raises(DataError, match="No scenario could be run"):
        stress_table(recent, {"AAA": 1.0})


def test_table_accepts_hypothetical_scenarios_too(prices):
    scenarios = [
        HISTORICAL_SCENARIOS[5],  # COVID
        uniform_shock("Equities -20%", ["AAA", "BBB"], -0.20),
    ]

    table = stress_table(prices, {"AAA": 0.5, "BBB": 0.5}, scenarios)

    assert len(table) == 2
    assert table.loc["Equities -20%", "portfolio_return"] == pytest.approx(-0.20)
    assert pd.isna(table.loc["Equities -20%", "start"])


# --- the built-in scenario list --------------------------------------


def test_every_builtin_scenario_is_well_formed():
    for scenario in HISTORICAL_SCENARIOS:
        assert scenario.name
        assert scenario.note, "a scenario without a note is a date with no meaning"
        assert pd.Timestamp(scenario.start) < pd.Timestamp(scenario.end)


def test_builtin_scenario_names_are_unique():
    names = [s.name for s in HISTORICAL_SCENARIOS]

    assert len(names) == len(set(names))


# --- input handling --------------------------------------------------


def test_returns_passed_instead_of_prices_are_rejected(prices):
    returns = prices.pct_change().dropna()

    with pytest.raises(DataError, match="strictly positive"):
        run_historical(returns, {"AAA": 0.5, "BBB": 0.5}, COVID)


def test_a_non_date_index_is_rejected():
    frame = pd.DataFrame({"AAA": [100.0, 90.0]})

    with pytest.raises(TypeError, match="indexed by date"):
        run_historical(frame, {"AAA": 1.0}, COVID)
