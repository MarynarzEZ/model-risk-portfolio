"""Tests for position sizing and currency handling.

The arithmetic here is simple enough that a bug would be embarrassing
and easy to miss: a rate applied the wrong way round turns 100 000 zloty
into 410 000 dollars and the number still looks like money. So the
round-trips are checked explicitly, in both directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.money import (
    USD_PLN,
    Portfolio,
    build_portfolio,
    convert_prices,
    format_money,
    normalise_weights,
    risk_table,
    warn_if_stale,
)

RATE = 4.1


@pytest.fixture
def portfolio() -> Portfolio:
    return build_portfolio(
        {"AAA": 60, "BBB": 40}, total=100_000, currency="USD", fx_rate=RATE
    )


@pytest.fixture
def prices_with_fx() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=10, name="date")
    return pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 110.0, 10),
            "BBB": np.linspace(50.0, 45.0, 10),
            USD_PLN: np.linspace(4.0, 4.2, 10),
        },
        index=index,
    )


# --- weights ---------------------------------------------------------


def test_weights_are_scaled_to_one_and_the_original_sum_comes_back():
    """The caller needs the original sum to be able to say what it did."""
    weights, total = normalise_weights({"AAA": 30, "BBB": 30, "CCC": 20, "DDD": 20})

    assert total == pytest.approx(100.0)
    assert weights.sum() == pytest.approx(1.0)
    assert weights["AAA"] == pytest.approx(0.30)


def test_proportions_survive_normalisation():
    """Rescaling must not change the portfolio, only its units."""
    weights, total = normalise_weights({"AAA": 3, "BBB": 1})

    assert total == pytest.approx(4.0)
    assert weights["AAA"] / weights["BBB"] == pytest.approx(3.0)


def test_percentages_and_fractions_give_the_same_portfolio():
    as_percent, _ = normalise_weights({"AAA": 60, "BBB": 40})
    as_fraction, _ = normalise_weights({"AAA": 0.6, "BBB": 0.4})

    pd.testing.assert_series_equal(as_percent, as_fraction)


def test_ticker_case_and_whitespace_are_normalised():
    weights, _ = normalise_weights({" aaa ": 50, "Bbb": 50})

    assert list(weights.index) == ["AAA", "BBB"]


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="Negative weights"):
        normalise_weights({"AAA": 1.5, "BBB": -0.5})


def test_an_empty_portfolio_is_rejected():
    with pytest.raises(ValueError, match="at least one holding"):
        normalise_weights({})


def test_weights_summing_to_zero_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        normalise_weights({"AAA": 0.0, "BBB": 0.0})


# --- currency conversion ---------------------------------------------


def test_dollars_convert_to_zloty_at_the_rate(portfolio):
    assert portfolio.total_usd == pytest.approx(100_000)
    assert portfolio.total_pln == pytest.approx(410_000)


def test_zloty_convert_to_dollars_at_the_rate():
    """The inverse. A rate applied backwards still looks like money."""
    from_pln = build_portfolio({"AAA": 1.0}, total=410_000, currency="PLN", fx_rate=RATE)

    assert from_pln.total_pln == pytest.approx(410_000)
    assert from_pln.total_usd == pytest.approx(100_000)


def test_the_two_directions_round_trip():
    usd_first = build_portfolio({"AAA": 1.0}, 100_000, "USD", fx_rate=RATE)
    pln_first = build_portfolio({"AAA": 1.0}, usd_first.total_pln, "PLN", fx_rate=RATE)

    assert pln_first.total_usd == pytest.approx(usd_first.total_usd)


def test_positions_split_the_total_by_weight(portfolio):
    positions = portfolio.positions

    assert positions.loc["AAA", "value_usd"] == pytest.approx(60_000)
    assert positions.loc["BBB", "value_usd"] == pytest.approx(40_000)
    assert positions.loc["AAA", "value_pln"] == pytest.approx(246_000)
    assert positions["value_usd"].sum() == pytest.approx(portfolio.total_usd)


def test_a_fraction_becomes_money_in_either_currency(portfolio):
    """A 5% loss on 100k is 5k, or 20 500 zloty."""
    assert portfolio.amount(0.05, "USD") == pytest.approx(5_000)
    assert portfolio.amount(0.05, "PLN") == pytest.approx(20_500)


def test_an_unknown_currency_is_rejected(portfolio):
    with pytest.raises(ValueError, match="Currency must be one of"):
        portfolio.amount(0.05, "EUR")


def test_a_non_positive_total_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        build_portfolio({"AAA": 1.0}, total=0, currency="USD", fx_rate=RATE)


def test_a_non_positive_rate_is_rejected():
    with pytest.raises(ValueError, match="Exchange rate must be positive"):
        build_portfolio({"AAA": 1.0}, 100_000, "USD", fx_rate=0.0)


def test_weights_are_normalised_on_the_way_in():
    portfolio = build_portfolio({"AAA": 30, "BBB": 30}, 100_000, "USD", fx_rate=RATE)

    assert portfolio.weights.sum() == pytest.approx(1.0)
    assert portfolio.positions["value_usd"].sum() == pytest.approx(100_000)


# --- restating prices in zloty ---------------------------------------


def test_converted_prices_carry_the_exchange_rate(prices_with_fx):
    """A stock flat in dollars is not flat in zloty if the rate moves."""
    flat = prices_with_fx.copy()
    flat["AAA"] = 100.0

    converted = convert_prices(flat)

    assert converted["AAA"].iloc[0] == pytest.approx(400.0)
    assert converted["AAA"].iloc[-1] == pytest.approx(420.0)


def test_conversion_drops_the_rate_column(prices_with_fx):
    converted = convert_prices(prices_with_fx)

    assert list(converted.columns) == ["AAA", "BBB"]
    assert converted.index.equals(prices_with_fx.index)


def test_zloty_returns_include_fx_moves(prices_with_fx):
    """The whole reason convert_prices exists."""
    from src.returns import simple_returns

    flat_in_dollars = prices_with_fx.copy()
    flat_in_dollars["AAA"] = 100.0

    usd_returns = simple_returns(flat_in_dollars[["AAA"]])
    pln_returns = simple_returns(convert_prices(flat_in_dollars)[["AAA"]])

    assert (usd_returns["AAA"] == 0).all(), "flat in dollars means no dollar risk"
    assert (pln_returns["AAA"] != 0).any(), "but a zloty investor still carries the rate"


def test_conversion_without_a_rate_column_is_rejected(prices_with_fx):
    with pytest.raises(ValueError, match="load_prices"):
        convert_prices(prices_with_fx.drop(columns=[USD_PLN]))


def test_conversion_of_a_rate_only_frame_is_rejected(prices_with_fx):
    with pytest.raises(ValueError, match="only the exchange rate"):
        convert_prices(prices_with_fx[[USD_PLN]])


# --- presentation ----------------------------------------------------


def test_risk_table_puts_percentages_next_to_amounts(portfolio):
    table = risk_table({"VaR": 0.056, "ES": 0.073}, portfolio, "PLN")

    assert list(table.index) == ["VaR", "ES"]
    assert table.loc["VaR", "pct"] == pytest.approx(0.056)
    assert table.loc["VaR", "amount_pln"] == pytest.approx(0.056 * 410_000)


def test_money_is_formatted_per_currency():
    assert format_money(1234567, "USD") == "$1,234,567"
    assert format_money(1234567, "PLN") == "1 234 567 zł"


def test_a_stale_rate_is_reported():
    old = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)

    with pytest.warns(UserWarning, match="days old"):
        message = warn_if_stale(old)

    assert "30 days old" in message


def test_a_fresh_rate_says_nothing():
    assert warn_if_stale(pd.Timestamp.today().normalize()) is None
    assert warn_if_stale(None) is None
