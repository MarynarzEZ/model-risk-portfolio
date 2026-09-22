"""Value at Risk.

VaR answers one question: over the next day, what loss will not be
exceeded with probability `confidence`? Four estimators live here and
they disagree on purpose -- the disagreement is the interesting part of
a risk report, because it measures how much of the answer comes from the
data and how much from the distribution you assumed.

Sign convention
---------------
Every function returns VaR as a **positive loss**: 0.0231 means "a loss
of 2.31% of portfolio value". Returns go in signed (negative = loss),
VaR comes out positive. Mixing the two conventions is the single most
common bug in risk code, and it is invisible -- a sign error produces a
plausible-looking number, not a crash -- so the convention is stated in
every docstring and asserted in the tests.

A negative VaR is possible and is not an error: it means even the bad
tail of the sample was a gain. It is reported rather than clamped to
zero, because clamping would hide a sample that is too short or too
lucky to be measuring risk at all.

Input
-----
Pass *simple* returns of the thing you want the risk of -- usually the
output of `returns.portfolio_returns`. Passing a DataFrame of per-asset
returns is rejected rather than handled column by column: a column of
VaRs is not a portfolio VaR, it ignores diversification, and summing
them overstates risk by a wide margin.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Below this many observations in the tail, the historical estimate is
# reading one or two data points and the number it prints has no more
# precision than a guess. 99% over one trading year leaves 2.5.
_MIN_TAIL_OBSERVATIONS = 5

# Student's t with fewer than 2 degrees of freedom has infinite
# variance; below 1 the mean does not exist either. A fit landing there
# says the sample has an outlier the model cannot cope with.
_MIN_SENSIBLE_DF = 2.0

_TRADING_DAYS_PER_YEAR = 252


def historical_var(returns: pd.Series | Sequence[float], confidence: float = 0.99) -> float:
    """VaR read straight off the empirical distribution.

    No distributional assumption at all: sort the observed returns and
    look at the `1 - confidence` quantile. That is the method's strength
    -- it reproduces whatever fat tails and skew the sample actually had
    -- and its weakness, since it cannot produce a loss larger than the
    worst one already seen, and it weights a crisis from five years ago
    exactly like yesterday.

    Uses the `lower` quantile rather than interpolating between the two
    neighbouring observations. Interpolation invents a return that never
    happened; `lower` always reports a loss that genuinely occurred,
    which is both more conservative and easier to defend.
    """
    sample = _as_sample(returns)
    _check_confidence(confidence)
    _warn_on_thin_tail(sample, confidence)

    cutoff = np.quantile(sample, 1.0 - confidence, method="lower")
    return float(-cutoff)


def normal_var(returns: pd.Series | Sequence[float], confidence: float = 0.99) -> float:
    """VaR assuming returns are normally distributed.

    Fast, closed-form, and the benchmark everyone quotes. It is also
    wrong in a known direction: daily equity returns have fatter tails
    than a normal, so at 99% this understates the loss, typically by
    10-25% for a diversified equity portfolio. Compare it against
    `historical_var` and `student_t_var`; if it is the lowest of the
    three, that gap is the cost of the normality assumption, not noise.
    """
    sample = _as_sample(returns)
    _check_confidence(confidence)

    mu = float(np.mean(sample))
    # ddof=1: the mean was estimated from this same sample, so one
    # degree of freedom is already spent. The difference is under half a
    # percent at n=250, but getting it wrong biases every downstream
    # number in the same direction.
    sigma = float(np.std(sample, ddof=1))

    quantile = mu + sigma * stats.norm.ppf(1.0 - confidence)
    return float(-quantile)


def student_t_var(returns: pd.Series | Sequence[float], confidence: float = 0.99) -> float:
    """VaR from a Student's t fitted by maximum likelihood.

    The t has one more parameter than the normal -- the degrees of
    freedom -- and it buys exactly the thing the normal gets wrong: the
    weight of the tail. Fitted on daily equity returns it usually lands
    between 3 and 6 df, which is a visibly fatter tail than a normal.

    Location and scale are fitted alongside the degrees of freedom
    rather than fixed at the sample mean and standard deviation. Fixing
    them would be faster, but the sample standard deviation is inflated
    by the same outliers the t is supposed to model, which partly
    cancels the point of using it.
    """
    sample = _as_sample(returns)
    _check_confidence(confidence)

    df, loc, scale = _fit_student_t(sample)
    quantile = stats.t.ppf(1.0 - confidence, df, loc=loc, scale=scale)
    return float(-quantile)


def monte_carlo_var(
    returns: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
    confidence: float = 0.99,
    *,
    n_paths: int = 100_000,
    seed: int | None = None,
    distribution: str = "normal",
) -> float:
    """VaR from simulated portfolio returns.

    Given a DataFrame of per-asset returns and weights, this estimates
    the mean vector and covariance matrix, draws `n_paths` correlated
    one-day scenarios, and reads the quantile off the simulated
    portfolio. That is what distinguishes it from `normal_var` applied
    to an already-aggregated series: the correlation structure is
    modelled explicitly, so the same machinery extends to positions
    whose payoff is not linear in the underlying (options, for one)
    without rewriting the estimator. Given a single Series it degenerates
    to the one-asset case, which is a useful thing to be able to check.

    `distribution` selects the innovations: "normal" or "t". A flag is
    acceptable here -- unlike the simple/log split in `returns` -- because
    both choices estimate the same quantity and the result stays
    comparable; only the assumed tail thickness changes.

    Pass `seed` for a reproducible figure. Without one, two calls will
    differ in the third decimal, which is fine for exploration and not
    fine for anything you are going to quote.
    """
    _check_confidence(confidence)
    simulated = _simulate_portfolio(
        returns,
        weights,
        n_paths=n_paths,
        seed=seed,
        distribution=distribution,
    )
    return float(-np.quantile(simulated, 1.0 - confidence, method="lower"))


def scale_var(var_1d: float, horizon: int) -> float:
    """Scale a one-day VaR to `horizon` days by the square root of time.

    Valid only if returns are independent and identically distributed
    across days. They are not: volatility clusters, so a calm day is
    followed by a calm day and a violent one by another violent one.
    Square-root scaling therefore understates risk in a crisis and
    overstates it in a quiet market -- and the error grows with the
    horizon, which is why 10-day regulatory VaR is usually estimated
    from 10-day returns directly rather than scaled up from daily.

    Kept because it is the standard first approximation, not because it
    is right.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"Horizon must be a positive whole number of days, got {horizon!r}")
    return float(var_1d * np.sqrt(horizon))


def var_summary(
    returns: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
    confidence: float = 0.99,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Every method's VaR and Expected Shortfall in one table.

    This is the deliverable: a single number is a claim, four numbers
    side by side show how much of that claim rests on the distribution
    that was assumed. A wide spread between historical and normal is not
    a bug in either, it is the fat tail being priced.

    The Monte Carlo row uses one simulated sample for both columns, so
    its VaR and ES are internally consistent.
    """
    _check_confidence(confidence)
    # Imported here rather than at module scope: `cvar` builds on this
    # module's estimators, so a top-level import would be circular. The
    # summary is the one place that needs to reach back the other way.
    from src import cvar as _cvar

    series = _to_portfolio_series(returns, weights)
    sample = _as_sample(series)

    simulated = _simulate_portfolio(
        returns, weights, n_paths=100_000, seed=seed, distribution="normal"
    )
    mc_var = float(-np.quantile(simulated, 1.0 - confidence, method="lower"))
    mc_tail = simulated[simulated <= -mc_var]

    rows = {
        "historical": (historical_var(series, confidence), _cvar.historical_cvar(series, confidence)),
        "normal": (normal_var(series, confidence), _cvar.normal_cvar(series, confidence)),
        "student_t": (student_t_var(series, confidence), _cvar.student_t_cvar(series, confidence)),
        "monte_carlo": (mc_var, float(-mc_tail.mean())),
    }

    table = pd.DataFrame(
        [{"method": name, "var": var, "expected_shortfall": es} for name, (var, es) in rows.items()]
    ).set_index("method")
    table["es_var_ratio"] = table["expected_shortfall"] / table["var"]
    table.attrs["confidence"] = confidence
    table.attrs["n_obs"] = len(sample)
    return table


def annualised_volatility(
    returns: pd.Series | Sequence[float], periods_per_year: int = _TRADING_DAYS_PER_YEAR
) -> float:
    """Daily volatility scaled to a year. Same iid caveat as `scale_var`."""
    sample = _as_sample(returns)
    return float(np.std(sample, ddof=1) * np.sqrt(periods_per_year))


# --- shared internals ------------------------------------------------
#
# `cvar` imports these, so that both modules fit the same distribution to
# the same sample and cannot drift apart.


def _fit_student_t(sample: np.ndarray) -> tuple[float, float, float]:
    """Fit a location-scale Student's t by maximum likelihood."""
    if len(sample) < 30:
        # Three parameters from fewer than 30 points is curve-fitting,
        # not estimation. Still returned, but the caller should know.
        warnings.warn(
            f"Fitting a Student's t to {len(sample)} observations; the degrees of "
            f"freedom will be badly determined.",
            stacklevel=3,
        )

    df, loc, scale = stats.t.fit(sample)

    if df < _MIN_SENSIBLE_DF:
        warnings.warn(
            f"Fitted {df:.2f} degrees of freedom, below {_MIN_SENSIBLE_DF}: the fitted "
            f"distribution has infinite variance, so its VaR is driven by one or two "
            f"extreme observations rather than by the shape of the sample.",
            stacklevel=3,
        )
    return float(df), float(loc), float(scale)


def _simulate_portfolio(
    returns: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None,
    *,
    n_paths: int,
    seed: int | None,
    distribution: str,
) -> np.ndarray:
    """Draw `n_paths` one-day portfolio returns from a fitted model."""
    if distribution not in {"normal", "t"}:
        raise ValueError(f"distribution must be 'normal' or 't', got {distribution!r}")
    if n_paths < 1000:
        # The 99% quantile of 1000 draws is the 10th worst point. The
        # simulation error would swamp the model difference it is meant
        # to show.
        raise ValueError(f"n_paths must be at least 1000 for a 99% quantile, got {n_paths}")

    frame = _to_asset_frame(returns)
    aligned = _weight_vector(frame.columns, weights)

    mu = frame.to_numpy().mean(axis=0)
    cov = np.cov(frame.to_numpy(), rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)

    rng = np.random.default_rng(seed)
    normal_draws = rng.standard_normal((n_paths, len(mu)))

    if distribution == "normal":
        draws = mu + normal_draws @ _psd_factor(cov).T
    else:
        # Degrees of freedom are fitted on the portfolio series rather
        # than estimated jointly across assets. A joint fit is the more
        # principled route but needs far more data to pin down; using the
        # portfolio's own tail keeps the thing being simulated and the
        # thing being measured consistent.
        portfolio = _as_sample(_to_portfolio_series(returns, weights))
        df, _, _ = _fit_student_t(portfolio)
        if df <= _MIN_SENSIBLE_DF:
            raise ValueError(
                f"Fitted {df:.2f} degrees of freedom; a multivariate t needs more than "
                f"{_MIN_SENSIBLE_DF} for its covariance to exist. Use distribution='normal' "
                f"or a longer sample."
            )
        # Scale matrix chosen so the simulated covariance equals `cov`:
        # a multivariate t with scale S has covariance S * df/(df-2).
        scale_matrix = cov * (df - 2.0) / df
        mixing = rng.chisquare(df, size=(n_paths, 1)) / df
        draws = mu + (normal_draws @ _psd_factor(scale_matrix).T) / np.sqrt(mixing)

    return draws @ aligned.to_numpy()


def _psd_factor(cov: np.ndarray) -> np.ndarray:
    """A matrix L with L @ L.T == cov, for correlating normal draws."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Cholesky needs strict positive definiteness, which a sample
        # covariance loses whenever two assets are collinear or there are
        # more assets than observations. Clipping the negative
        # eigenvalues to zero gives the nearest PSD matrix in the
        # Frobenius sense -- the simulation stays valid, it just cannot
        # invent variation along the degenerate directions.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        clipped = np.clip(eigenvalues, 0.0, None)
        if (eigenvalues < -1e-10).any():
            warnings.warn(
                "The sample covariance matrix is not positive semi-definite; negative "
                "eigenvalues were clipped to zero. This usually means two assets move "
                "identically, or there are fewer observations than assets.",
                stacklevel=4,
            )
        return eigenvectors * np.sqrt(clipped)


# --- argument handling -----------------------------------------------


def _as_sample(returns: pd.Series | Sequence[float]) -> np.ndarray:
    """Coerce a return series to a clean 1-d float array."""
    if isinstance(returns, pd.DataFrame):
        raise TypeError(
            "Got a DataFrame of per-asset returns. Aggregate it first with "
            "returns.portfolio_returns(returns, weights): a VaR per column is not a "
            "portfolio VaR -- it ignores diversification, and adding the columns up "
            "assumes every asset has its worst day simultaneously."
        )

    sample = np.asarray(
        returns.dropna() if isinstance(returns, pd.Series) else returns, dtype="float64"
    )
    if sample.ndim != 1:
        raise ValueError(f"Expected a one-dimensional return series, got shape {sample.shape}")
    if sample.size < 2:
        raise ValueError("At least two returns are needed to estimate VaR")
    if not np.isfinite(sample).all():
        raise ValueError("Returns contain NaN or infinity; clean the data before measuring risk")
    return sample


def _to_asset_frame(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Present either input shape as a per-asset DataFrame."""
    if isinstance(returns, pd.DataFrame):
        frame = returns.dropna(how="any")
    elif isinstance(returns, pd.Series):
        frame = returns.dropna().to_frame(name=returns.name or "portfolio")
    else:
        frame = pd.Series(np.asarray(returns, dtype="float64"), name="portfolio").to_frame()

    if frame.empty:
        raise ValueError("No returns left after dropping missing values")
    if len(frame) < 2:
        raise ValueError("At least two returns are needed to estimate a covariance")
    return frame


def _to_portfolio_series(
    returns: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None,
) -> pd.Series:
    """Collapse per-asset returns to one series, if they are not already."""
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            raise ValueError("Weights are required when passing per-asset returns")
        from src.returns import portfolio_returns

        return portfolio_returns(returns.dropna(how="any"), weights)
    if isinstance(returns, pd.Series):
        return returns.dropna()
    return pd.Series(np.asarray(returns, dtype="float64"), name="portfolio")


def _weight_vector(
    columns: pd.Index,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None,
) -> pd.Series:
    """Reuse the weight handling and validation from `returns`."""
    if weights is None:
        if len(columns) != 1:
            raise ValueError("Weights are required when passing per-asset returns")
        return pd.Series([1.0], index=columns, dtype="float64")

    from src.returns import _align_weights

    return _align_weights(columns, weights)


def _check_confidence(confidence: float) -> None:
    """Reject confidence levels that are almost certainly a mistake."""
    if not 0.5 < confidence < 1.0:
        # Catches the common slip of passing the tail probability (0.01)
        # instead of the confidence level (0.99). Below 0.5 the "loss"
        # being measured is the middle of the distribution.
        raise ValueError(
            f"Confidence must be between 0.5 and 1 exclusive, got {confidence}. "
            f"Pass the confidence level (0.99), not the tail probability (0.01)."
        )


def _warn_on_thin_tail(sample: np.ndarray, confidence: float) -> None:
    """Warn when the empirical quantile rests on too few observations."""
    in_tail = len(sample) * (1.0 - confidence)
    if in_tail < _MIN_TAIL_OBSERVATIONS:
        warnings.warn(
            f"Only {in_tail:.1f} of {len(sample)} observations fall in the "
            f"{1 - confidence:.1%} tail, so the historical estimate is reading one or "
            f"two data points. Use a longer sample or a lower confidence level.",
            stacklevel=3,
        )
