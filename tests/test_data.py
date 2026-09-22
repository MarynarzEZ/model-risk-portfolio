"""Tests for the price loader.

No test here touches the network. `src.data._fetch_one` is the single
point where yfinance is called, so replacing it gives full control over
what "Yahoo" returns -- including the cases that are hard to catch in
the wild, like a delisted ticker or a bad print.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src import data
from src.data import DataError, load_prices


def make_series(ticker: str, start: str, periods: int, first: float = 100.0) -> pd.Series:
    """A deterministic, strictly rising price series on business days."""
    index = pd.bdate_range(start=start, periods=periods, name="date")
    values = first * np.power(1.001, np.arange(periods))
    return pd.Series(values, index=index, name=ticker)


@pytest.fixture
def fake_yahoo(monkeypatch):
    """Replace the download with a recorded, in-memory stand-in.

    Returns the call log so tests can assert on cache hits.
    """
    calls: list[str] = []
    series_by_ticker: dict[str, pd.Series] = {}

    def fetch(ticker, start, end):
        calls.append(ticker)
        if ticker not in series_by_ticker:
            raise DataError(f"no data for {ticker!r}")
        return series_by_ticker[ticker].loc[start:end]

    monkeypatch.setattr(data, "_fetch_one", fetch)
    fetch.calls = calls  # type: ignore[attr-defined]
    fetch.series = series_by_ticker  # type: ignore[attr-defined]
    return fetch


def test_returns_one_column_per_ticker_in_request_order(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 20)
    fake_yahoo.series["BBB"] = make_series("BBB", "2024-01-01", 20, first=50.0)

    prices = load_prices(["BBB", "AAA"], "2024-01-01", "2024-01-26", cache_dir=tmp_path)

    assert list(prices.columns) == ["BBB", "AAA"]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.is_monotonic_increasing
    assert prices.index.name == "date"
    assert prices.notna().all(axis=None)


def test_lowercase_and_duplicate_tickers_are_normalised(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 10)

    prices = load_prices(["aaa", "AAA", " aaa "], "2024-01-01", "2024-01-12", cache_dir=tmp_path)

    assert list(prices.columns) == ["AAA"]
    assert fake_yahoo.calls == ["AAA"], "a duplicate ticker should not be downloaded twice"


def test_single_ticker_may_be_a_bare_string(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 10)

    prices = load_prices("AAA", "2024-01-01", "2024-01-12", cache_dir=tmp_path)

    assert list(prices.columns) == ["AAA"]


def test_keeps_only_days_when_every_ticker_traded(fake_yahoo, tmp_path):
    # BBB is missing one mid-sample day, as a foreign listing on a local
    # holiday would be.
    aaa = make_series("AAA", "2024-01-01", 20)
    bbb = make_series("BBB", "2024-01-01", 20, first=50.0)
    holiday = bbb.index[5]
    fake_yahoo.series["AAA"] = aaa
    fake_yahoo.series["BBB"] = bbb.drop(holiday)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prices = load_prices(["AAA", "BBB"], "2024-01-01", "2024-01-26", cache_dir=tmp_path)

    assert holiday not in prices.index
    assert len(prices) == 19


def test_warns_when_alignment_drops_many_days(fake_yahoo, tmp_path):
    aaa = make_series("AAA", "2024-01-01", 40)
    fake_yahoo.series["AAA"] = aaa
    fake_yahoo.series["BBB"] = make_series("BBB", "2024-01-01", 40, first=50.0).iloc[::2]

    with pytest.warns(UserWarning, match="did not trade"):
        load_prices(["AAA", "BBB"], "2024-01-01", "2024-02-23", cache_dir=tmp_path)


def test_date_range_is_inclusive_at_both_ends(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 30)

    prices = load_prices("AAA", "2024-01-03", "2024-01-10", cache_dir=tmp_path)

    assert prices.index.min() == pd.Timestamp("2024-01-03")
    assert prices.index.max() == pd.Timestamp("2024-01-10")


def test_second_call_is_served_from_cache(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 20)

    first = load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path)
    second = load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path)

    assert fake_yahoo.calls == ["AAA"]
    pd.testing.assert_frame_equal(first, second)


def test_cache_covers_a_narrower_request(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 30)
    load_prices("AAA", "2024-01-01", "2024-02-08", cache_dir=tmp_path)

    narrower = load_prices("AAA", "2024-01-10", "2024-01-20", cache_dir=tmp_path)

    assert fake_yahoo.calls == ["AAA"], "a sub-range should not trigger a download"
    assert narrower.index.min() >= pd.Timestamp("2024-01-10")
    assert narrower.index.max() <= pd.Timestamp("2024-01-20")


def test_cache_does_not_cover_a_wider_request(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 60)
    load_prices("AAA", "2024-01-10", "2024-01-20", cache_dir=tmp_path)

    load_prices("AAA", "2024-01-01", "2024-02-08", cache_dir=tmp_path)

    assert fake_yahoo.calls == ["AAA", "AAA"]


def test_refresh_bypasses_the_cache(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 20)
    load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path)

    load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path, refresh=True)

    assert fake_yahoo.calls == ["AAA", "AAA"]


def test_cached_prices_survive_a_round_trip(fake_yahoo, tmp_path):
    """The CSV cache must not quietly lose precision or reorder rows."""
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 20)

    fresh = load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path)
    cached = load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path)

    pd.testing.assert_frame_equal(fresh, cached, check_exact=False, rtol=1e-12)


def test_unknown_ticker_raises(fake_yahoo, tmp_path):
    with pytest.raises(DataError):
        load_prices("NOPE", "2024-01-01", "2024-01-26", cache_dir=tmp_path)


def test_non_positive_price_raises(fake_yahoo, tmp_path):
    broken = make_series("AAA", "2024-01-01", 20)
    broken.iloc[7] = 0.0
    fake_yahoo.series["AAA"] = broken

    with pytest.raises(DataError, match="Non-positive price"):
        load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path)


def test_tickers_with_no_common_dates_raise(fake_yahoo, tmp_path):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 10)
    fake_yahoo.series["BBB"] = make_series("BBB", "2024-06-03", 10, first=50.0)

    with pytest.raises(DataError, match="no trading day in common"):
        load_prices(["AAA", "BBB"], "2024-01-01", "2024-06-14", cache_dir=tmp_path)


def test_end_before_start_raises(fake_yahoo, tmp_path):
    with pytest.raises(ValueError, match="before start"):
        load_prices("AAA", "2024-02-01", "2024-01-01", cache_dir=tmp_path)


def test_empty_ticker_list_raises(fake_yahoo, tmp_path):
    with pytest.raises(ValueError, match="At least one ticker"):
        load_prices([], "2024-01-01", "2024-01-26", cache_dir=tmp_path)


def test_unparseable_date_raises(fake_yahoo, tmp_path):
    with pytest.raises(ValueError, match="as a date"):
        load_prices("AAA", "last tuesday", cache_dir=tmp_path)


def test_awkward_ticker_symbols_are_cacheable(fake_yahoo, tmp_path):
    """`^GSPC` and `BRK.B` are valid symbols and awkward filenames."""
    for symbol in ("^GSPC", "BRK.B"):
        fake_yahoo.series[symbol] = make_series(symbol, "2024-01-01", 20)

    load_prices(["^GSPC", "BRK.B"], "2024-01-01", "2024-01-26", cache_dir=tmp_path)
    prices = load_prices(["^GSPC", "BRK.B"], "2024-01-01", "2024-01-26", cache_dir=tmp_path)

    assert fake_yahoo.calls == ["^GSPC", "BRK.B"]
    assert list(prices.columns) == ["^GSPC", "BRK.B"]


def test_duplicate_timestamps_keep_the_last_quote(fake_yahoo, tmp_path):
    """Yahoo sometimes repeats a date; the later row is the correction."""
    series = make_series("AAA", "2024-01-01", 20)
    corrected = pd.concat([series, series.iloc[[3]] * 2]).sort_index(kind="stable")
    fake_yahoo.series["AAA"] = corrected

    prices = load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path)

    assert prices.index.is_unique
    assert prices["AAA"].loc[series.index[3]] == pytest.approx(series.iloc[3] * 2)


def test_unwritable_cache_warns_but_returns_prices(fake_yahoo, tmp_path, monkeypatch):
    fake_yahoo.series["AAA"] = make_series("AAA", "2024-01-01", 20)

    def refuse(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(data.Path, "mkdir", refuse)

    with pytest.warns(UserWarning, match="Could not cache"):
        prices = load_prices("AAA", "2024-01-01", "2024-01-26", cache_dir=tmp_path / "sub")

    assert len(prices) == 20
