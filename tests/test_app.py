"""Smoke tests for the Streamlit app.

`AppTest` runs app.py the way Streamlit does and lets the test inspect
what landed on the page. It will not tell you the layout is sensible --
that is settled by opening it -- but it does catch the failure that
matters most in a GUI: an exception thrown halfway through a render,
leaving the user with a stack trace instead of a number.

Nothing here touches the network. The download is replaced, and so is
the on-disk cache: the repository's `data/` folder holds real prices
for the app's own default tickers, so without stubbing the cache these
tests would quietly read those instead of the fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import data
from src.money import USD_PLN

# Absolute: AppTest resolves a relative path against the file that calls
# it, which is this one, not the repository root.
APP = str(Path(__file__).resolve().parent.parent / "app.py")
_SESSIONS = 900


def synthetic(ticker: str, seed: int) -> pd.Series:
    """A plausible price path: positive, drifting, with fat-ish tails."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=_SESSIONS, name="date")

    if ticker == USD_PLN:
        # Exchange rates wander; they do not compound like equities.
        values = 4.0 + np.cumsum(rng.normal(0, 0.01, _SESSIONS))
        return pd.Series(np.clip(values, 2.5, 6.0), index=index, name=ticker)

    shocks = rng.standard_t(4, _SESSIONS) * 0.015 + 0.0004
    return pd.Series(100 * np.exp(np.cumsum(shocks)), index=index, name=ticker)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Cut every route to the network and to the real price cache."""
    monkeypatch.setattr(
        data, "_fetch_one", lambda ticker, start, end: synthetic(ticker, abs(hash(ticker)) % 10_000)
    )
    monkeypatch.setattr(data, "_read_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(data, "_write_cache", lambda *args, **kwargs: None)

    # Streamlit's cache outlives a single test, so a stubbed result from
    # one test would be served to the next.
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def run_app(**widget_values) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    for key, value in widget_values.items():
        app.session_state[key] = value
    app.run()
    return app


def test_the_app_renders_without_blowing_up():
    app = run_app()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_every_section_is_present():
    app = run_app()

    assert len(app.tabs) == 5
    assert app.get("metric"), "the header metrics should always render"


def test_headline_numbers_are_money_not_fractions():
    """A VaR of 0.056 on the page would be useless to size a position."""
    app = run_app()
    values = [m.value for m in app.get("metric")]

    assert any("$" in str(v) for v in values), f"no dollar amount among {values[:8]}"


def test_switching_to_zloty_changes_the_numbers():
    """PLN restates prices before measuring, so it is not a relabelling."""
    in_usd = run_app()
    usd_values = [m.value for m in in_usd.get("metric")]

    st.cache_data.clear()
    in_pln = run_app(report_currency="PLN")
    pln_values = [m.value for m in in_pln.get("metric")]

    assert not in_pln.exception, [str(e.value) for e in in_pln.exception]
    assert usd_values != pln_values


def test_a_lower_confidence_level_lowers_the_risk_figure():
    app = run_app()
    app.get("select_slider")[0].set_value(0.90).run()

    assert not app.exception, [str(e.value) for e in app.exception]


def test_an_empty_portfolio_asks_for_input_instead_of_crashing():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.session_state["holdings"] = {"edited_rows": {}, "added_rows": [], "deleted_rows": [0, 1, 2, 3]}
    app.run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.info, "an empty portfolio should prompt, not fail"


def test_weights_that_do_not_sum_to_one_hundred_are_handled():
    """The library refuses them; the GUI rescales and says so."""
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.session_state["holdings"] = {
        "edited_rows": {0: {"Weight %": 10.0}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()

    assert not app.exception, [str(e.value) for e in app.exception]
    captions = " ".join(str(c.value) for c in app.caption)
    assert "rescaled" in captions


def test_a_single_holding_still_works():
    """Correlations need two assets; the rest of the app must not care."""
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.session_state["holdings"] = {
        "edited_rows": {},
        "added_rows": [],
        "deleted_rows": [1, 2, 3],
    }
    app.run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.info, "the correlation tab should explain why it is empty"
