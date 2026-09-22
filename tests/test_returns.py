"""Tests for the returns transformations.

The numbers here are checked against arithmetic done by hand rather than
against another pandas call, so a change in pandas' own semantics shows
up as a failure instead of being confirmed by itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.returns import log_returns, portfolio_returns, simple_returns


@pytest.fixture
def prices() -> pd.DataFrame:
    """Prices chosen so the returns are exact in decimal."""
    index = pd.bdate_range("2024-01-01", periods=4, name="date")
    return pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 99.0, 108.9],  # +10%, -10%, +10%
            "BBB": [50.0, 50.0, 55.0, 49.5],  # 0%, +10%, -10%
        },
        index=index,
    )


def test_simple_returns_match_hand_arithmetic(prices):
    result = simple_returns(prices)

    assert result["AAA"].tolist() == pytest.approx([0.10, -0.10, 0.10])
    assert result["BBB"].tolist() == pytest.approx([0.00, 0.10, -0.10])


def test_simple_returns_drop_the_first_row(prices):
    result = simple_returns(prices)

    assert len(result) == len(prices) - 1
    assert result.index[0] == prices.index[1]
    assert result.notna().all(axis=None)


def test_log_returns_are_the_log_of_one_plus_simple(prices):
    simple = simple_returns(prices)
    logs = log_returns(prices)

    pd.testing.assert_frame_equal(np.log1p(simple), logs)


def test_log_returns_add_up_over_time(prices):
    """The property that makes log returns worth having."""
    logs = log_returns(prices)
    total = float(logs["AAA"].sum())

    expected = np.log(prices["AAA"].iloc[-1] / prices["AAA"].iloc[0])
    assert total == pytest.approx(expected)


def test_returns_work_on_a_single_series(prices):
    result = simple_returns(prices["AAA"])

    assert isinstance(result, pd.Series)
    assert result.tolist() == pytest.approx([0.10, -0.10, 0.10])


def test_portfolio_return_is_the_weighted_sum(prices):
    returns = simple_returns(prices)

    combined = portfolio_returns(returns, {"AAA": 0.6, "BBB": 0.4})

    expected = 0.6 * returns["AAA"] + 0.4 * returns["BBB"]
    assert combined.tolist() == pytest.approx(expected.tolist())
    assert combined.name == "portfolio"


def test_weights_are_matched_by_name_not_position(prices):
    returns = simple_returns(prices)

    by_name = portfolio_returns(returns, {"BBB": 0.4, "AAA": 0.6})
    reordered = portfolio_returns(returns[["BBB", "AAA"]], {"AAA": 0.6, "BBB": 0.4})

    assert by_name.tolist() == pytest.approx(reordered.tolist())


def test_weights_may_be_a_series_or_a_sequence(prices):
    returns = simple_returns(prices)
    expected = portfolio_returns(returns, {"AAA": 0.6, "BBB": 0.4}).tolist()

    as_series = portfolio_returns(returns, pd.Series({"AAA": 0.6, "BBB": 0.4}))
    as_list = portfolio_returns(returns, [0.6, 0.4])  # in column order

    assert as_series.tolist() == pytest.approx(expected)
    assert as_list.tolist() == pytest.approx(expected)


def test_short_positions_are_allowed(prices):
    returns = simple_returns(prices)

    combined = portfolio_returns(returns, {"AAA": 1.5, "BBB": -0.5})

    expected = 1.5 * returns["AAA"] - 0.5 * returns["BBB"]
    assert combined.tolist() == pytest.approx(expected.tolist())


def test_weights_that_do_not_sum_to_one_raise(prices):
    returns = simple_returns(prices)

    with pytest.raises(ValueError, match="must sum to 1"):
        portfolio_returns(returns, {"AAA": 0.6, "BBB": 0.3})


def test_hand_typed_thirds_are_accepted(prices):
    """0.33 + 0.33 + 0.34 is a portfolio, not a typo."""
    index = prices.index
    three = pd.DataFrame(
        {"AAA": [100.0, 101.0], "BBB": [50.0, 50.5], "CCC": [20.0, 20.2]},
        index=index[:2],
    )
    returns = simple_returns(three)

    combined = portfolio_returns(returns, {"AAA": 0.33, "BBB": 0.33, "CCC": 0.34})

    assert combined.iloc[0] == pytest.approx(0.01)


def test_missing_weight_raises(prices):
    returns = simple_returns(prices)

    with pytest.raises(ValueError, match="No weight given for BBB"):
        portfolio_returns(returns, {"AAA": 1.0})


def test_weight_for_an_unknown_ticker_raises(prices):
    returns = simple_returns(prices)

    with pytest.raises(ValueError, match="not in the data"):
        portfolio_returns(returns, {"AAA": 0.5, "BBB": 0.4, "CCC": 0.1})


def test_wrong_number_of_positional_weights_raises(prices):
    returns = simple_returns(prices)

    with pytest.raises(ValueError, match="3 weights for 2 assets"):
        portfolio_returns(returns, [0.5, 0.3, 0.2])


def test_non_finite_weights_raise(prices):
    returns = simple_returns(prices)

    with pytest.raises(ValueError, match="finite"):
        portfolio_returns(returns, {"AAA": float("nan"), "BBB": 1.0})


def test_non_positive_prices_raise():
    broken = pd.DataFrame({"AAA": [100.0, 0.0, 90.0]}, index=pd.bdate_range("2024-01-01", periods=3))

    with pytest.raises(ValueError, match="strictly positive"):
        log_returns(broken)


def test_a_single_price_cannot_produce_a_return():
    one = pd.DataFrame({"AAA": [100.0]}, index=pd.bdate_range("2024-01-01", periods=1))

    with pytest.raises(ValueError, match="two price observations"):
        simple_returns(one)
