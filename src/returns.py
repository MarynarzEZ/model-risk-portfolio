"""Turning prices into returns.

Two conventions live here and they are not interchangeable:

* simple returns, p_t / p_{t-1} - 1, add up *across assets*, so a
  portfolio return is the weighted sum of its holdings' returns;
* log returns, ln(p_t / p_{t-1}), add up *across time*, so a 10-day
  return is the sum of ten daily ones, and they are closer to normally
  distributed, which parametric VaR assumes.

Neither is "the right one". Use simple returns to aggregate a portfolio,
log returns to aggregate over time. The functions below are separate
rather than sharing a `kind="log"` flag so that the choice is visible at
every call site, since picking the wrong one is silent and wrong by a
few basis points a day -- small enough to look plausible in a VaR figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

# Weights are allowed to miss 1.0 by this much before we refuse them.
# Loose enough for hand-typed thirds (0.33 + 0.33 + 0.34), tight enough
# to catch a portfolio that is accidentally 10% or 200% invested.
_WEIGHT_TOLERANCE = 1e-4


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Period-over-period arithmetic returns.

    The first row is dropped rather than kept as NaN: every risk measure
    downstream would have to drop it anyway, and a leading NaN that
    survives into a covariance estimate quietly shrinks the sample.
    """
    _check_prices(prices)
    # pandas 3 no longer fills gaps in pct_change by default, which is
    # what we want -- prices arrive already aligned from `data.load_prices`.
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Continuously compounded returns, ln(p_t / p_{t-1})."""
    _check_prices(prices)
    return np.log(prices).diff().iloc[1:]


def portfolio_returns(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
) -> pd.Series:
    """Collapse per-asset returns into one portfolio series.

    `weights` may be a mapping or Series keyed by ticker (matched by
    name, so column order does not matter) or a plain sequence in column
    order. They must sum to 1; negative weights are fine and mean a
    short position.

    Pass *simple* returns. Log returns do not add up across assets --
    ln of a weighted sum is not the weighted sum of the logs -- so
    feeding them in here understates the portfolio return, and the error
    grows with volatility, which is precisely when the risk numbers
    matter. This cannot be detected from the data, so it is on the
    caller.

    Weights are held fixed: this is the return of a portfolio rebalanced
    back to `weights` every period, not of one bought once and left
    alone. For daily VaR over a short horizon the difference is
    negligible, and fixed weights are what a risk limit is written
    against.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("portfolio_returns expects a DataFrame of per-asset returns")
    if returns.empty:
        raise ValueError("No returns to aggregate")

    aligned = _align_weights(returns.columns, weights)

    total = float(aligned.sum())
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        # Renormalising silently would be friendlier and wrong: it turns
        # a typo into a different portfolio and reports its risk as if
        # it were the one that was asked for.
        raise ValueError(f"Weights must sum to 1, got {total:.6f}")

    return returns.mul(aligned, axis=1).sum(axis=1).rename("portfolio")


def _align_weights(
    columns: pd.Index,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
) -> pd.Series:
    """Match weights to return columns, by name where possible."""
    if isinstance(weights, pd.Series):
        weights = weights.to_dict()

    if isinstance(weights, Mapping):
        named = {str(key).strip().upper(): float(value) for key, value in weights.items()}
        missing = [str(col) for col in columns if str(col).upper() not in named]
        if missing:
            raise ValueError(f"No weight given for {', '.join(missing)}")
        extra = sorted(set(named) - {str(col).upper() for col in columns})
        if extra:
            # Silently ignoring these would hide a typo'd ticker, and the
            # remaining weights would no longer sum to 1 anyway.
            raise ValueError(f"Weights given for columns not in the data: {', '.join(extra)}")
        values = [named[str(col).upper()] for col in columns]
    else:
        values = [float(value) for value in weights]
        if len(values) != len(columns):
            raise ValueError(
                f"Got {len(values)} weights for {len(columns)} assets. Pass a "
                f"mapping keyed by ticker if you want order-independent matching."
            )

    aligned = pd.Series(values, index=columns, dtype="float64")
    if not np.isfinite(aligned).all():
        raise ValueError("Weights must all be finite numbers")
    return aligned


def _check_prices(prices: pd.DataFrame | pd.Series) -> None:
    """Reject price input that cannot produce meaningful returns."""
    if not isinstance(prices, (pd.DataFrame, pd.Series)):
        raise TypeError("Prices must be a pandas Series or DataFrame")
    if len(prices) < 2:
        raise ValueError("At least two price observations are needed for a return")
    if (prices <= 0).any(axis=None):
        # Matches the check in `data`, for prices built by hand or read
        # from somewhere other than `load_prices`.
        raise ValueError("Prices must be strictly positive")
