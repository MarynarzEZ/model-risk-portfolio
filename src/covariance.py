"""Covariance and correlation structure.

A portfolio's risk is not the sum of its positions' risks -- it is
whatever is left after the assets partly cancel each other out. This
module estimates the thing that decides how much cancelling happens.

The uncomfortable part is that the cancelling is least reliable exactly
when it matters. Correlations between equities rise sharply in a sell-off
(everything is sold at once, for reasons that have nothing to do with
the individual companies), so a diversification benefit measured over a
calm sample quietly evaporates in the crisis it was supposed to cushion.
`rolling_correlation` exists to make that visible rather than to assume
it away.

Inputs are simple returns, as everywhere else in this package.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252

# RiskMetrics' 1996 technical document fitted this on daily data across
# a range of markets and it has been the default ever since. It puts a
# half-life of about 11 days on an observation, so a month-old return
# carries roughly a seventh of yesterday's weight. It is a calibration
# choice, not a constant of nature: a lower lambda reacts faster to a
# regime change and is noisier for it.
DEFAULT_LAMBDA = 0.94


def sample_covariance(
    returns: pd.DataFrame,
    *,
    annualise: bool = False,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Equally weighted sample covariance.

    Every day in the sample counts the same, which is the estimator's
    one honest virtue -- there is nothing to tune -- and its main flaw:
    a crisis from three years ago pushes today's risk estimate up just
    as hard as last week's, and a long quiet stretch hides a regime
    change until it is well underway.

    Annualising multiplies by the number of periods per year (variance
    scales linearly with time under independence), which carries the
    same iid caveat as `var.scale_var`.
    """
    frame = _check_returns(returns)
    cov = frame.cov(ddof=1)
    if annualise:
        cov = cov * periods_per_year
    return cov


def correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix.

    Pearson measures *linear* co-movement. When two assets move together
    only in crashes -- which is the case that matters for a risk model --
    a single Pearson number averaged over calm and crisis understates the
    dependence that shows up when you need it. Use `rolling_correlation`
    to see the variation this number hides.
    """
    frame = _check_returns(returns)
    return frame.corr(method="pearson")


def ewma_covariance(
    returns: pd.DataFrame,
    lam: float = DEFAULT_LAMBDA,
    *,
    demean: bool = False,
) -> pd.DataFrame:
    """Exponentially weighted covariance, RiskMetrics style.

    Weight on an observation `k` days old is proportional to `lam ** k`,
    normalised to sum to one. The result is a covariance matrix that
    responds to a change in regime within days rather than the months a
    long equally weighted window takes.

    `demean=False` by default, following RiskMetrics: the mean daily
    return is statistically indistinguishable from zero on any sample
    short enough to be relevant, so subtracting an estimate of it adds
    noise without removing bias. Set it to True on a long sample, or on
    returns with an obvious drift.

    The result is positive semi-definite by construction -- it is a
    positively weighted sum of outer products -- which matters because a
    covariance matrix that is not PSD can imply a portfolio with
    negative variance.
    """
    frame = _check_returns(returns)
    _check_lambda(lam)

    values = frame.to_numpy()
    if demean:
        values = values - values.mean(axis=0)

    weights = _ewma_weights(len(values), lam)
    weighted = (values * weights[:, None]).T @ values
    return pd.DataFrame(weighted, index=frame.columns, columns=frame.columns)


def ewma_volatility(
    returns: pd.DataFrame | pd.Series,
    lam: float = DEFAULT_LAMBDA,
    *,
    annualise: bool = False,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame | pd.Series:
    """Conditional volatility over time, one value per date.

    Recursive form: var_t = lam * var_{t-1} + (1 - lam) * r_{t-1}**2.
    Note the lag -- the estimate for day t uses returns up to t-1 only,
    so it is a genuine forecast and can be fed to a backtest without
    leaking tomorrow's data into today's risk number.

    Seeded with the variance of the first 20 observations rather than
    with zero or with the first squared return; either of those takes
    dozens of days to wash out and makes the start of the series
    meaningless.
    """
    frame, was_series, name = _as_frame(returns)
    _check_lambda(lam)

    values = frame.to_numpy()
    seed_window = min(20, len(values))
    variance = np.var(values[:seed_window], axis=0, ddof=1)

    out = np.empty_like(values)
    for row in range(len(values)):
        out[row] = variance
        variance = lam * variance + (1.0 - lam) * values[row] ** 2

    volatility = np.sqrt(out)
    if annualise:
        volatility = volatility * np.sqrt(periods_per_year)

    result = pd.DataFrame(volatility, index=frame.index, columns=frame.columns)
    return result[name].rename(name) if was_series else result


def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Average pairwise correlation in a rolling window.

    One number per date summarising how tightly the whole portfolio is
    moving together. Watch it through a crash: on a diversified equity
    book it typically runs around 0.3-0.5 in calm markets and jumps past
    0.8 in a sell-off. That jump is the diversification benefit
    disappearing, and it is the single most useful plot in a risk report.

    Averages the off-diagonal entries only, since the diagonal is 1 by
    construction and would just dilute the number towards one.
    """
    frame = _check_returns(returns, min_columns=2)
    if window < 2:
        raise ValueError(f"Window must span at least two days, got {window}")
    if window > len(frame):
        raise ValueError(
            f"Window of {window} days is longer than the {len(frame)} days of returns given"
        )

    n_assets = frame.shape[1]
    n_pairs = n_assets * (n_assets - 1) // 2

    averages = []
    values = frame.to_numpy()
    for end in range(window, len(values) + 1):
        block = np.corrcoef(values[end - window : end], rowvar=False)
        # Sum of the strict upper triangle, divided by the pair count.
        averages.append((np.sum(block) - n_assets) / 2.0 / n_pairs)

    return pd.Series(averages, index=frame.index[window - 1 :], name="average_correlation")


def portfolio_volatility(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    cov: pd.DataFrame,
) -> float:
    """Portfolio standard deviation, sqrt(w' * cov * w).

    This is where diversification is actually priced: the cross terms
    are negative whenever two assets are less than perfectly correlated,
    so the result is below the weighted average of the individual
    volatilities.
    """
    if not isinstance(cov, pd.DataFrame):
        raise TypeError("cov must be a DataFrame indexed by asset")
    if not cov.index.equals(cov.columns):
        raise ValueError("Covariance matrix must have matching row and column labels")

    from src.returns import _align_weights

    aligned = _align_weights(cov.columns, weights).to_numpy()
    variance = float(aligned @ cov.to_numpy() @ aligned)

    if variance < 0:
        # Only reachable with a covariance matrix that is not PSD, which
        # is a broken input rather than a portfolio with negative risk.
        raise ValueError(
            f"Portfolio variance came out negative ({variance:.3e}). The covariance "
            f"matrix is not positive semi-definite -- check it with "
            f"is_positive_semidefinite()."
        )
    return float(np.sqrt(variance))


def is_positive_semidefinite(matrix: pd.DataFrame | np.ndarray, tolerance: float = 1e-10) -> bool:
    """Whether every eigenvalue is non-negative, within tolerance.

    A covariance matrix that fails this implies some portfolio has
    negative variance, which is impossible. It happens for real reasons:
    more assets than observations, two tickers that are the same
    security, or a matrix repaired by hand. The tolerance absorbs
    floating-point noise around zero, which a legitimately singular
    matrix will produce.
    """
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {values.shape}")
    if not np.allclose(values, values.T, atol=tolerance):
        return False
    return bool(np.min(np.linalg.eigvalsh(values)) >= -tolerance)


# --- internals -------------------------------------------------------


def _ewma_weights(n_obs: int, lam: float) -> np.ndarray:
    """Normalised exponential weights, oldest observation first."""
    # Computed from the newest backwards, then reversed, so that the
    # largest weight is always 1 before normalisation and lam ** n never
    # underflows to zero for a long sample.
    raw = lam ** np.arange(n_obs - 1, -1, -1.0)
    return raw / raw.sum()


def _check_lambda(lam: float) -> None:
    if not 0.0 < lam < 1.0:
        raise ValueError(
            f"Lambda must be between 0 and 1 exclusive, got {lam}. Values near 1 weight "
            f"history almost equally; near 0 only yesterday counts."
        )


def _check_returns(returns: pd.DataFrame, min_columns: int = 1) -> pd.DataFrame:
    """Validate a per-asset return frame."""
    if isinstance(returns, pd.Series):
        raise TypeError(
            "Covariance needs at least two assets; got a single series. Pass the "
            "per-asset return frame, not the aggregated portfolio."
        )
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame with one column per asset")

    frame = returns.dropna(how="any")
    if frame.shape[1] < min_columns:
        raise ValueError(f"At least {min_columns} assets are required, got {frame.shape[1]}")
    if len(frame) < 2:
        raise ValueError("At least two dates are needed to estimate a covariance")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Returns contain infinities; clean the data first")

    if len(frame) < frame.shape[1]:
        warnings.warn(
            f"{len(frame)} observations for {frame.shape[1]} assets: the sample "
            f"covariance matrix is singular and cannot be inverted. Use "
            f"ewma_covariance or a longer sample.",
            stacklevel=3,
        )
    return frame


def _as_frame(returns: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame, bool, str]:
    """Accept either shape, remembering which one to hand back."""
    if isinstance(returns, pd.Series):
        name = returns.name or "portfolio"
        return returns.dropna().to_frame(name=name), True, str(name)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas Series or DataFrame")

    frame = returns.dropna(how="any")
    if len(frame) < 2:
        raise ValueError("At least two returns are needed")
    return frame, False, ""
