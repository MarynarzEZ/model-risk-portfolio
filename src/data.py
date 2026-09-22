"""Loading daily equity prices.

Prices come from Yahoo Finance through `yfinance` and are cached on disk.
The cache is not a speed optimisation: a VaR number is only worth
quoting if it can be reproduced, and Yahoo is rate-limited, silently
revises history and is not always reachable. Caching pins the inputs.

The cache lives under `data/`, which is gitignored -- the prices are
Yahoo's, not ours, so they stay out of the repository.

Prices are adjusted closes (splits and dividends folded in), because
risk is measured on what an investor actually earns. Using raw closes
would inject a fake -3% return on every ex-dividend date.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# This file is src/data.py, so the repository root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "prices"

# Share of the union of trading days we are willing to lose to calendar
# alignment before warning. A US-only portfolio loses nothing; mixing in
# a Warsaw or Tokyo listing loses a few days a year, which is fine. Much
# more than that usually means one ticker is stale or plain wrong.
_ALIGNMENT_WARN_THRESHOLD = 0.02

DateLike = str | date | datetime | pd.Timestamp


class DataError(Exception):
    """Raised when downloaded prices are missing or unusable.

    Separate from `ValueError` on purpose: a `ValueError` here means the
    caller asked for something impossible, a `DataError` means the
    request was fine but the data that came back cannot be trusted.
    """


def load_prices(
    tickers: str | Iterable[str],
    start: DateLike,
    end: DateLike | None = None,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return adjusted daily closes, one column per ticker.

    `start` and `end` are both inclusive (yfinance treats `end` as
    exclusive; that is corrected below, because an inclusive end is what
    everyone assumes). `end` defaults to today.

    Rows are restricted to dates on which *every* ticker traded, so the
    returns computed from this frame are directly comparable across
    assets -- which is what a covariance matrix needs. The alternative,
    forward-filling a missing price, is worse than it looks: it invents
    a zero return, which pulls volatility down and correlations towards
    zero exactly on the illiquid names where risk is underestimated
    already.

    Pass `refresh=True` to ignore the cache and re-download.
    """
    names = _normalise_tickers(tickers)
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    if end_ts < start_ts:
        raise ValueError(f"end ({end_ts.date()}) is before start ({start_ts.date()})")

    cache_dir = Path(cache_dir)
    columns = [
        _load_one(name, start_ts, end_ts, cache_dir=cache_dir, refresh=refresh)
        for name in names
    ]
    return _align(columns)


def _load_one(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.Series:
    """Return one ticker's price series, from cache when possible."""
    if not refresh:
        cached = _read_cache(ticker, start, end, cache_dir)
        if cached is not None:
            return cached

    series = _validate(_fetch_one(ticker, start, end), ticker)
    _write_cache(series, ticker, start, end, cache_dir)
    return series


def _fetch_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download one ticker's adjusted close from Yahoo Finance.

    Deliberately one ticker per call, even though yfinance can batch.
    Batching is faster, but it caches as a single blob keyed to one
    portfolio; per-ticker series are reusable across every portfolio
    that touches the name, and one dead ticker cannot poison the rest
    of the download.

    Tests replace this function, so it is the only place that talks to
    the network.
    """
    frame = yf.download(
        ticker,
        start=start,
        # yfinance's `end` is exclusive, ours is inclusive.
        end=end + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=True,  # `Close` becomes the adjusted close
        actions=False,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise DataError(
            f"Yahoo Finance returned no data for {ticker!r} between "
            f"{start.date()} and {end.date()} -- check the symbol and the dates"
        )
    if "Close" not in frame.columns:
        raise DataError(f"No close price in the data returned for {ticker!r}")
    return frame["Close"].rename(ticker)


def _validate(series: pd.Series, ticker: str) -> pd.Series:
    """Clean and sanity-check a single price series."""
    series = series.rename(ticker).astype("float64")

    # Yahoo timestamps daily bars at midnight in the exchange timezone;
    # dropping the timezone keeps indices from different exchanges
    # joinable. For daily data no information is lost.
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = pd.DatetimeIndex(series.index).normalize()
    series.index.name = "date"

    # A stable sort, so that `keep="last"` below means the last row Yahoo
    # sent rather than whichever of the tied rows quicksort happened to
    # leave at the end. Yahoo occasionally repeats a date, and the later
    # row is the corrected one.
    series = series.sort_index(kind="stable")
    series = series[~series.index.duplicated(keep="last")]
    series = series.dropna()

    if series.empty:
        raise DataError(f"No usable prices for {ticker!r} -- every row was empty")
    if (series <= 0).any():
        # Not a judgement call: a non-positive price is impossible for
        # cash equities, and it would make log returns undefined.
        bad = series.index[series <= 0][0].date()
        raise DataError(f"Non-positive price for {ticker!r} on {bad}")
    return series


def _align(columns: list[pd.Series]) -> pd.DataFrame:
    """Join price series on their common trading days."""
    frame = pd.concat(columns, axis=1, join="outer").sort_index()
    frame.index.name = "date"

    complete = frame.dropna(how="any")
    if complete.empty:
        raise DataError(
            "The tickers have no trading day in common -- check that the "
            "date ranges and exchanges actually overlap"
        )

    dropped = len(frame) - len(complete)
    if dropped and dropped / len(frame) > _ALIGNMENT_WARN_THRESHOLD:
        warnings.warn(
            f"Dropped {dropped} of {len(frame)} dates where at least one "
            f"ticker did not trade ({dropped / len(frame):.1%}). Mixed "
            f"exchange calendars are expected to cost a little; this much "
            f"usually means one series is incomplete.",
            stacklevel=3,
        )
    return complete


# --- cache -----------------------------------------------------------
#
# One CSV per ticker per requested range, named
# `<ticker>__<start>__<end>.csv`. The requested range goes in the
# filename rather than being inferred from the data, because the two are
# not the same: a ticker that listed in 2021 has no rows before 2021, and
# without the requested range recorded we would re-download it forever
# looking for history that does not exist.
#
# CSV rather than parquet to avoid a pyarrow dependency for what is a few
# hundred KB of daily closes, and because a cache you can open in a text
# editor is easier to trust.


def _read_cache(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path
) -> pd.Series | None:
    """Return a cached series covering [start, end], or None."""
    if not cache_dir.is_dir():
        return None

    for path in sorted(cache_dir.glob(f"{_safe_name(ticker)}__*.csv")):
        covered = _parse_cache_range(path)
        if covered is None:
            continue
        cached_start, cached_end = covered
        if cached_start <= start and end <= cached_end:
            series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
            # A one-row cache file squeezes to a scalar, not a Series.
            if not isinstance(series, pd.Series):
                continue
            return _validate(series, ticker).loc[start:end]
    return None


def _write_cache(
    series: pd.Series,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> None:
    """Store a downloaded series. Cache failures must not be fatal."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = f"{_safe_name(ticker)}__{start:%Y-%m-%d}__{end:%Y-%m-%d}.csv"
        series.to_csv(cache_dir / name, header=True)
    except OSError as exc:
        # A read-only or full disk is a reason to be slow, not to fail:
        # the prices are already in memory and perfectly usable.
        warnings.warn(f"Could not cache prices for {ticker!r}: {exc}", stacklevel=3)


def _parse_cache_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Pull the requested range out of a cache filename."""
    parts = path.stem.split("__")
    if len(parts) != 3:
        return None
    try:
        return pd.Timestamp(parts[1]), pd.Timestamp(parts[2])
    except ValueError:
        return None


def _safe_name(ticker: str) -> str:
    """Make a ticker usable as a filename (`BRK.B`, `^GSPC`, `EUR=X`)."""
    return "".join(char if char.isalnum() else "-" for char in ticker)


# --- argument handling -----------------------------------------------


def _normalise_tickers(tickers: str | Iterable[str]) -> list[str]:
    """Accept one ticker or many; return them upper-cased and deduped."""
    if isinstance(tickers, str):
        tickers = [tickers]

    seen: list[str] = []
    for raw in tickers:
        name = str(raw).strip().upper()
        if not name:
            raise ValueError("Ticker symbols cannot be empty")
        # Duplicates would produce two identical columns and a singular
        # covariance matrix further down the line.
        if name not in seen:
            seen.append(name)

    if not seen:
        raise ValueError("At least one ticker is required")
    return seen


def _as_timestamp(value: DateLike) -> pd.Timestamp:
    """Parse a date argument, rejecting anything ambiguous."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not read {value!r} as a date: {exc}") from exc
    if stamp.tz is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()
