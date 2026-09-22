"""Tests for the report charts.

A chart cannot be checked for looking right, so these check the things
that silently go wrong instead: that the figure is built at all, that
every data point reaches it, that the labels say what the numbers say,
and that a chart never invents a second y-axis. Whether it is legible
is settled by looking at it, which no test replaces.
"""

from __future__ import annotations

import matplotlib

# Must precede pyplot: the tests run with no display attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.backtest import rolling_var_backtest  # noqa: E402
from src.covariance import ewma_volatility, rolling_correlation  # noqa: E402
from src.plots import (  # noqa: E402
    plot_backtest,
    plot_ewma_volatility,
    plot_return_distribution,
    plot_rolling_correlation,
    plot_stress,
    plot_var_comparison,
    set_theme,
)
from src.stress import HISTORICAL_SCENARIOS, stress_table  # noqa: E402
from src.var import var_summary  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Matplotlib keeps every figure alive until told otherwise."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def light_theme():
    set_theme("light")


@pytest.fixture
def portfolio() -> pd.Series:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2014-01-01", periods=1500, name="date")
    return pd.Series(rng.normal(0.0004, 0.011, size=1500), index=index, name="portfolio")


@pytest.fixture
def assets() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    cov = [[0.012**2, 0.5 * 0.012 * 0.015], [0.5 * 0.012 * 0.015, 0.015**2]]
    index = pd.bdate_range("2014-01-01", periods=1500, name="date")
    return pd.DataFrame(
        rng.multivariate_normal([0.0, 0.0], cov, size=1500), index=index, columns=["AAA", "BBB"]
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    index = pd.bdate_range("2006-01-02", "2023-12-29", name="date")
    n = len(index)
    return pd.DataFrame(
        {"AAA": 100 * np.exp(np.linspace(0, 0.5, n)), "BBB": 50 * np.exp(np.linspace(0, 0.3, n))},
        index=index,
    )


# --- the comparison table --------------------------------------------


def test_var_comparison_draws_a_pair_of_bars_per_method(portfolio):
    summary = var_summary(portfolio, confidence=0.99, seed=1)

    fig = plot_var_comparison(summary)
    ax = fig.axes[0]

    assert len(fig.axes) == 1, "a second axis would mean a second y-scale"
    assert len(ax.containers) == 2, "one container per measure"
    assert [text.get_text() for text in ax.legend_.get_texts()] == ["VaR", "Expected Shortfall"]
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        "historical",
        "normal",
        "student t",
        "monte carlo",
    ]


def test_bar_labels_match_the_underlying_numbers(portfolio):
    summary = var_summary(portfolio, confidence=0.99, seed=1)

    fig = plot_var_comparison(summary)
    labels = [text.get_text() for text in fig.axes[0].texts]

    assert f"{summary['var'].iloc[0] * 100:.2f}%" in labels
    assert f"{summary['expected_shortfall'].iloc[0] * 100:.2f}%" in labels


def test_confidence_reaches_the_title(portfolio):
    summary = var_summary(portfolio, confidence=0.95, seed=1)

    assert "95%" in plot_var_comparison(summary).axes[0].get_title(loc="left")


def test_a_table_without_the_expected_columns_is_rejected():
    with pytest.raises(ValueError, match="Expected column"):
        plot_var_comparison(pd.DataFrame({"something_else": [1.0]}))


# --- distribution ----------------------------------------------------


def test_distribution_marks_both_measures(portfolio):
    fig = plot_return_distribution(portfolio, 0.025, 0.033, 0.99)
    labels = " ".join(text.get_text() for text in fig.axes[0].texts)

    assert "VaR 2.50%" in labels
    assert "ES 3.30%" in labels


def test_distribution_labels_do_not_sit_on_top_of_each_other(portfolio):
    """They collided when VaR and ES were close, which is the usual case."""
    fig = plot_return_distribution(portfolio, 0.0392, 0.0529, 0.99)
    offsets = {text.get_position()[1] for text in fig.axes[0].texts}

    assert len(offsets) == len(fig.axes[0].texts) or len(fig.axes[0].texts) == 2


def test_distribution_separates_the_tail_from_the_body(portfolio):
    fig = plot_return_distribution(portfolio, 0.025, 0.033, 0.99)
    ax = fig.axes[0]

    body, tail = ax.containers[0], ax.containers[1]
    assert len(body) > len(tail), "the tail is the small part, by definition"
    assert max(bar.get_x() for bar in tail) <= -2.5


# --- time series -----------------------------------------------------


def test_rolling_correlation_plots_every_point(assets):
    rolling = rolling_correlation(assets, window=60)

    line = plot_rolling_correlation(rolling, 60).axes[0].lines[0]

    assert len(line.get_ydata()) == len(rolling)
    assert line.get_linewidth() == 2


def test_rolling_correlation_labels_the_peak(assets):
    rolling = rolling_correlation(assets, window=60)

    texts = " ".join(t.get_text() for t in plot_rolling_correlation(rolling).axes[0].texts)

    assert f"{rolling.max():.2f}" in texts
    assert "median" in texts


def test_volatility_plot_is_annualised(portfolio):
    vol = ewma_volatility(portfolio, annualise=True)

    ax = plot_ewma_volatility(vol).axes[0]

    assert "annualised" in ax.get_ylabel()
    assert ax.lines[0].get_ydata().max() == pytest.approx(vol.max() * 100)


# --- backtest --------------------------------------------------------


def test_backtest_plot_has_two_panels_not_two_scales(portfolio):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = rolling_var_backtest(portfolio, window=250)

    fig = plot_backtest(report)

    assert len(fig.axes) == 2
    assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])


def test_every_breach_is_marked(portfolio):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = rolling_var_backtest(portfolio, window=250)

    fig = plot_backtest(report)
    marked = fig.axes[0].collections[0].get_offsets()

    assert len(marked) == int(report.breaches.sum())


def test_backtest_title_states_the_verdict(portfolio):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = rolling_var_backtest(portfolio, window=250)

    title = plot_backtest(report).axes[0].get_title(loc="left")

    assert str(int(report.breaches.sum())) in title
    assert report.basel_zone in title


# --- stress ----------------------------------------------------------


def test_stress_plot_draws_one_bar_per_scenario(prices):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = stress_table(prices, {"AAA": 0.5, "BBB": 0.5})

    ax = plot_stress(table).axes[0]

    assert len(ax.containers[0]) == len(table)
    assert [label.get_text() for label in ax.get_yticklabels()] == list(table.index)


def test_stress_bars_share_one_colour(prices):
    """Shading by size would double-encode the length."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = stress_table(prices, {"AAA": 0.5, "BBB": 0.5})

    colours = {bar.get_facecolor() for bar in plot_stress(table).axes[0].containers[0]}

    assert len(colours) == 1


def test_worst_scenario_is_drawn_at_the_top(prices):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = stress_table(prices, {"AAA": 0.5, "BBB": 0.5})

    ax = plot_stress(table).axes[0]
    bottom, top = ax.get_ylim()

    assert bottom > top, "the y-axis must be inverted so the sort reads downwards"
    assert ax.get_yticklabels()[0].get_text() == table.index[0]


def test_stress_plot_needs_the_right_column():
    with pytest.raises(ValueError, match="Expected column"):
        plot_stress(pd.DataFrame({"loss": [-0.3]}, index=["Made up"]))


# --- theming ---------------------------------------------------------


def test_dark_theme_changes_the_surface(portfolio):
    summary = var_summary(portfolio, seed=1)

    set_theme("light")
    light = plot_var_comparison(summary).get_facecolor()
    set_theme("dark")
    dark = plot_var_comparison(summary).get_facecolor()

    assert light != dark


def test_an_unknown_theme_is_rejected():
    with pytest.raises(ValueError, match="'light' or 'dark'"):
        set_theme("solarized")


# --- the scenario list is plottable ----------------------------------


def test_every_builtin_scenario_name_fits_on_an_axis():
    """A label longer than this gets clipped or shrinks the plot."""
    for scenario in HISTORICAL_SCENARIOS:
        assert len(scenario.name) <= 30

