"""Position sizing and currency.

The risk modules work in fractions: a VaR of 0.056 means "5.6% of the
portfolio". Nobody sets a risk limit in fractions. This module turns
them into amounts, in dollars and in zloty, and handles the part that
is easy to get quietly wrong -- what currency the risk is measured in.

Two different numbers, and the difference is not cosmetic
----------------------------------------------------------
A Polish investor holding US stocks carries two risks: the stocks can
fall, and the dollar can fall against the zloty. Converting a dollar
VaR to zloty at today's rate measures only the first and reports it in
zloty, which understates the real exposure -- USDPLN has moved more
than 2% in a day often enough to matter.

`convert_prices` exists for that reason. It restates the prices
themselves in zloty before any risk is computed, so the FX move is part
of the return series and every downstream estimator picks it up with no
changes at all. Use it when the investor's money is zloty. Use the plain
dollar figures when it is dollars. Both are correct answers to different
questions; the app shows which one it is answering.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.data import DataError, load_prices

# Yahoo quotes this as zloty per dollar: 4.1 means one dollar buys 4.1
# zloty. The inverse pair (PLNUSD=X) also exists and is the usual way to
# get this backwards.
USD_PLN = "USDPLN=X"

# How far back to look for a rate when the requested day was a holiday
# or the feed is lagging. Beyond a fortnight the "current" rate is stale
# enough to be worth an error rather than a silent approximation.
_MAX_FX_STALENESS_DAYS = 14

SUPPORTED_CURRENCIES = ("USD", "PLN")


@dataclass(frozen=True)
class Portfolio:
    """A portfolio sized in both currencies.

    `weights` always sum to 1. `total_usd` and `total_pln` are the same
    money seen through `fx_rate`, not two independent amounts.
    """

    weights: pd.Series
    total_usd: float
    total_pln: float
    fx_rate: float
    fx_date: pd.Timestamp | None = None

    @property
    def positions(self) -> pd.DataFrame:
        """One row per holding: weight and value in both currencies."""
        return pd.DataFrame(
            {
                "weight": self.weights,
                "value_usd": self.weights * self.total_usd,
                "value_pln": self.weights * self.total_pln,
            }
        )

    def amount(self, fraction: float, currency: str = "USD") -> float:
        """Turn a fraction of the portfolio into money.

        `amount(0.056, "PLN")` answers "what is a 5.6% loss worth?".
        """
        total = self.total_usd if _check_currency(currency) == "USD" else self.total_pln
        return float(fraction * total)


def latest_fx_rate(
    pair: str = USD_PLN,
    as_of: str | pd.Timestamp | None = None,
    *,
    cache_dir=None,
) -> tuple[float, pd.Timestamp]:
    """The most recent exchange rate at or before `as_of`, and its date.

    Returns the date as well as the number, because an exchange rate
    without a timestamp invites being read as live. It is not: Yahoo's
    FX quotes are delayed, and this is a daily close. Fine for sizing a
    position, not for pricing a trade.
    """
    end = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=_MAX_FX_STALENESS_DAYS)

    kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
    rates = load_prices(pair, start, end, **kwargs)

    if rates.empty:
        raise DataError(
            f"No quote for {pair} in the {_MAX_FX_STALENESS_DAYS} days to {end.date()}"
        )

    last = rates.iloc[-1, 0]
    return float(last), rates.index[-1]


def build_portfolio(
    weights: Mapping[str, float],
    total: float,
    currency: str = "USD",
    *,
    fx_rate: float | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> Portfolio:
    """Size a portfolio from weights and one total amount.

    `currency` says which currency `total` is in; the other is derived
    from the exchange rate. Weights are normalised here -- see
    `normalise_weights` for why that is acceptable in this one place.
    """
    currency = _check_currency(currency)
    if total <= 0:
        raise ValueError(f"Portfolio value must be positive, got {total}")

    normalised, _ = normalise_weights(weights)

    fx_date = None
    if fx_rate is None:
        fx_rate, fx_date = latest_fx_rate(as_of=as_of)
    if fx_rate <= 0:
        raise ValueError(f"Exchange rate must be positive, got {fx_rate}")

    if currency == "USD":
        total_usd, total_pln = float(total), float(total) * fx_rate
    else:
        total_usd, total_pln = float(total) / fx_rate, float(total)

    return Portfolio(
        weights=normalised,
        total_usd=total_usd,
        total_pln=total_pln,
        fx_rate=float(fx_rate),
        fx_date=fx_date,
    )


def normalise_weights(weights: Mapping[str, float]) -> tuple[pd.Series, float]:
    """Scale weights to sum to 1, returning the original sum too.

    The library refuses weights that do not sum to 1, because silently
    rescaling them turns a typo into a different portfolio. That rule
    holds for code. A person typing into a form is a different case:
    they will enter 30/30/20/20 and mean 100%, and three of those
    numbers summing to 0.9999 is not a bug worth an error dialog.

    The compromise is that the rescaling happens here and the original
    sum comes back with it, so the caller can show what it did. Rescaling
    is only acceptable when it is visible.
    """
    if not weights:
        raise ValueError("A portfolio needs at least one holding")

    series = pd.Series(
        {str(k).strip().upper(): float(v) for k, v in weights.items()}, dtype="float64"
    )
    if series.isna().any():
        raise ValueError("Weights must all be numbers")
    if (series < 0).any():
        # Shorts are a real thing, but they make "total value" ambiguous
        # and the GUI has no way to express the borrow. Out of scope.
        raise ValueError("Negative weights (short positions) are not supported here")

    total = float(series.sum())
    if total <= 0:
        raise ValueError("Weights must add up to something positive")

    return series / total, total


def convert_prices(prices: pd.DataFrame, fx_column: str = USD_PLN) -> pd.DataFrame:
    """Restate dollar prices in zloty, using the FX column in the frame.

    The FX series has to come from the same `load_prices` call as the
    stock prices, so both are already restricted to the days every one
    of them traded. Converting with a rate loaded separately would
    silently pair a Monday price with a Friday rate wherever the two
    calendars disagree.

    After this, every estimator in the package measures the risk a zloty
    investor actually carries -- the FX move is inside the returns.
    """
    if fx_column not in prices.columns:
        raise ValueError(
            f"No {fx_column} column. Load it alongside the stocks: "
            f"load_prices([*tickers, {fx_column!r}], start, end)"
        )

    stocks = prices.drop(columns=[fx_column])
    if stocks.empty:
        raise ValueError("Nothing to convert: the frame holds only the exchange rate")

    return stocks.mul(prices[fx_column], axis=0)


def risk_table(
    fractions: Mapping[str, float], portfolio: Portfolio, currency: str = "USD"
) -> pd.DataFrame:
    """Lay fractional risk measures out as money.

    Takes {"VaR": 0.056, "ES": 0.073} and gives back a frame with the
    percentage and the amount, which is the form a risk limit is
    actually written in.
    """
    currency = _check_currency(currency)
    rows = [
        {
            "measure": name,
            "pct": fraction,
            f"amount_{currency.lower()}": portfolio.amount(fraction, currency),
        }
        for name, fraction in fractions.items()
    ]
    return pd.DataFrame(rows).set_index("measure")


def format_money(amount: float, currency: str = "USD") -> str:
    """Render an amount the way each currency is normally written."""
    currency = _check_currency(currency)
    if currency == "USD":
        return f"${amount:,.0f}"
    # Polish convention: amount first, space, then the symbol.
    return f"{amount:,.0f} zł".replace(",", " ")


def _check_currency(currency: str) -> str:
    upper = str(currency).strip().upper()
    if upper not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"Currency must be one of {', '.join(SUPPORTED_CURRENCIES)}, got {currency!r}"
        )
    return upper


def warn_if_stale(fx_date: pd.Timestamp | None, tolerance_days: int = 4) -> str | None:
    """Message to show when the exchange rate is older than it looks."""
    if fx_date is None:
        return None
    age = (pd.Timestamp.today().normalize() - pd.Timestamp(fx_date)).days
    if age > tolerance_days:
        message = f"Exchange rate is from {pd.Timestamp(fx_date):%Y-%m-%d}, {age} days old."
        warnings.warn(message, stacklevel=2)
        return message
    return None
