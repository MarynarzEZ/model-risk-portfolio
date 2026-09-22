"""Charts for the risk report.

Every function takes an already-computed result -- a summary table, a
backtest report, a volatility series -- and returns a matplotlib Figure.
Nothing here estimates anything. Keeping the two apart means a chart
cannot quietly disagree with the number in the table beside it, and the
estimators stay testable without a display.

Design rules, applied throughout rather than per chart:

* one colour per entity, assigned from a fixed slot order, never cycled
  and never chosen by rank -- a bar does not change colour because it
  moved up the sort;
* status red is reserved for things that *are* bad (a VaR breach) and is
  never used as "series 3";
* no second y-axis, ever. Two measures on different scales get two
  panels, because the alignment of two scales is arbitrary and invents a
  relationship the data does not contain;
* hairline grid one shade off the surface, solid not dashed, and no
  frame -- the data should be the darkest thing on the page;
* a legend whenever there are two or more series, plus direct labels on
  the few marks that carry the point, so identity is never colour alone.
"""

from __future__ import annotations

from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# The validated default palette, used unchanged. The slot order is the
# colourblind-safety mechanism, not decoration: it was chosen so that
# adjacent slots stay distinguishable under simulated colour vision
# deficiency. No chart here uses more than three series, which is the
# range that clears every gate for scattered marks too. If you swap
# these hexes for your own, re-run the palette validator -- picking
# colours that "look different enough" is exactly what the check exists
# to replace.
_LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    "critical": "#d03b3b",
    "warning": "#fab219",
    "good": "#0ca30c",
}

_DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    # The same three hues re-stepped for the dark surface, not an
    # automatic inversion of the light values.
    "series": ("#3987e5", "#d95926", "#199e70"),
    "critical": "#d03b3b",
    "warning": "#fab219",
    "good": "#0ca30c",
}

_theme = dict(_LIGHT)


def set_theme(mode: Literal["light", "dark"] = "light") -> None:
    """Switch the palette. Call before building figures."""
    global _theme
    if mode not in {"light", "dark"}:
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
    _theme = dict(_LIGHT if mode == "light" else _DARK)


def plot_var_comparison(summary: pd.DataFrame) -> Figure:
    """VaR against Expected Shortfall, one pair of bars per method.

    The chart the whole package builds towards. Read the *spread*, not
    any one bar: the distance between the historical and the normal bar
    is the cost of assuming normality, and the ES column widens it
    because averaging the tail is more sensitive to its shape than
    reading one point on it.

    Horizontal bars because the category labels are words, and words
    read better along the axis than rotated under it.
    """
    _require_columns(summary, ["var", "expected_shortfall"])

    methods = list(summary.index)
    positions = np.arange(len(methods))
    height = 0.36

    # The extra inch of height is the band the legend sits in, below the
    # axis label. Sizing the figure to the plot alone would let
    # tight_layout crop it.
    fig, ax = _figure(figsize=(8.0, 0.85 * len(methods) + 2.6))

    for offset, (column, label, colour) in enumerate(
        (
            ("var", "VaR", _theme["series"][0]),
            ("expected_shortfall", "Expected Shortfall", _theme["series"][1]),
        )
    ):
        values = summary[column].to_numpy() * 100
        # The 0.02 gap keeps the two bars in a pair from touching; a
        # border drawn round each bar would do the same job louder.
        bars = ax.barh(
            positions + (0.5 - offset) * (height + 0.02),
            values,
            height=height,
            color=colour,
            label=label,
            zorder=3,
        )
        ax.bar_label(
            bars,
            fmt="%.2f%%",
            padding=4,
            color=_theme["secondary"],
            fontsize=9,
        )

    ax.set_yticks(positions, [m.replace("_", " ") for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("loss, % of portfolio value")
    confidence = summary.attrs.get("confidence")
    ax.set_xlim(0, float(summary["expected_shortfall"].max()) * 118)
    _style(
        ax,
        grid_axis="x",
        title="Daily VaR and Expected Shortfall by method"
        + (f"  ·  {confidence:.0%} confidence" if confidence else ""),
    )
    # Below the axis. Inside the plot it eventually lands on a bar's
    # value label -- which corner is free depends on the numbers, so no
    # fixed inside position is safe and `best` moves it between runs --
    # and on the title line it runs into the title.
    ax.legend(ncols=2, bbox_to_anchor=(0.5, -0.22), **_legend_kwargs("upper center"))
    return fig


def plot_return_distribution(
    returns: pd.Series, var: float, expected_shortfall: float, confidence: float = 0.99
) -> Figure:
    """What VaR and ES actually point at, drawn on the distribution.

    The single most useful picture for anyone who has to sign off on a
    risk number without having estimated one: VaR is a line on the
    x-axis, ES is the average of everything to the left of it. The bars
    beyond VaR are shaded, because that region -- not the line -- is what
    the capital is being held against.
    """
    sample = returns.dropna().to_numpy() * 100
    var_pct, es_pct = var * 100, expected_shortfall * 100

    fig, ax = _figure(figsize=(9.0, 4.6))

    counts, edges = np.histogram(sample, bins=120)
    centres = (edges[:-1] + edges[1:]) / 2
    in_tail = centres <= -var_pct

    ax.bar(
        centres[~in_tail],
        counts[~in_tail],
        width=np.diff(edges)[0] * 0.9,
        color=_theme["series"][0],
        label="daily returns",
        zorder=3,
    )
    ax.bar(
        centres[in_tail],
        counts[in_tail],
        width=np.diff(edges)[0] * 0.9,
        color=_theme["critical"],
        label=f"worst {1 - confidence:.0%} of days",
        zorder=3,
    )

    # The tail bars are genuinely tiny -- that is the shape of the data --
    # so the region is shaded as well, or the thing the chart is about
    # would be a few pixels tall.
    ax.axvspan(
        float(sample.min()) * 1.05,
        -var_pct,
        color=_theme["critical"],
        alpha=0.07,
        zorder=1,
    )

    # Stagger the two labels: at 99% on real data VaR and ES sit close
    # enough together that stacking them at the same height overprints
    # one on the other.
    for value, label, style, drop in (
        (-var_pct, f"VaR {var_pct:.2f}%", "-", -14),
        (-es_pct, f"ES {es_pct:.2f}%", ":", -30),
    ):
        ax.axvline(value, color=_theme["text"], linewidth=1.4, linestyle=style, zorder=4)
        ax.annotate(
            label,
            xy=(value, ax.get_ylim()[1]),
            xytext=(-6, drop),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=_theme["text"],
        )

    ax.set_xlabel("daily return, %")
    ax.set_ylabel("days")
    _style(ax, grid_axis="y", title=f"Where the {confidence:.0%} risk measures sit")
    ax.legend(**_legend_kwargs())
    return fig


def plot_rolling_correlation(rolling: pd.Series, window: int = 60) -> Figure:
    """Average pairwise correlation over time.

    The plot that undermines the comfortable version of
    diversification. One series, so no legend -- the title names it --
    and one annotation on the peak, because the date the correlations
    converged is the whole message and a number on every point would
    bury it.
    """
    fig, ax = _figure(figsize=(9.5, 4.0))

    ax.plot(rolling.index, rolling.to_numpy(), color=_theme["series"][0], linewidth=2, zorder=3)
    _annotate_extreme(ax, rolling, rolling.idxmax(), "{:.2f}", above=True)

    ax.axhline(float(rolling.median()), color=_theme["muted"], linewidth=1, zorder=2)
    # Anchored to the right edge: the left of a long correlation series
    # is as dense as the rest, and the label was being crossed by the
    # line it describes.
    ax.annotate(
        f"median {rolling.median():.2f}",
        xy=(rolling.index[-1], rolling.median()),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=_theme["muted"],
    )

    ax.set_ylabel("average pairwise correlation")
    _style(
        ax,
        grid_axis="y",
        title=f"Diversification disappears when it is needed  ·  {window}-day window",
    )
    return fig


def plot_ewma_volatility(volatility: pd.Series, annualised: bool = True) -> Figure:
    """Conditional volatility over time.

    Volatility clustering, drawn: calm stretches interrupted by spikes
    that decay over weeks. This is the behaviour that makes square-root
    -of-time scaling wrong and makes VaR exceedances arrive in clusters.
    """
    fig, ax = _figure(figsize=(9.5, 4.0))
    scale = 100 if annualised else 100

    ax.plot(
        volatility.index,
        volatility.to_numpy() * scale,
        color=_theme["series"][0],
        linewidth=1.8,
        zorder=3,
    )
    _annotate_extreme(ax, volatility * scale, volatility.idxmax(), "{:.0f}%", above=True)

    ax.set_ylabel("annualised volatility, %" if annualised else "daily volatility, %")
    _style(ax, grid_axis="y", title="Volatility clusters: quiet years, then weeks that undo them")
    return fig


def plot_backtest(report, max_points: int = 0) -> Figure:
    """Daily returns against the VaR that was forecast for that day.

    Two panels sharing an x-axis rather than two y-scales on one: the
    top shows the returns and the VaR line with every breach marked, the
    bottom counts breaches per quarter so clustering is visible as a
    shape rather than inferred from a p-value.

    Every mark below the line is a day the model was wrong. A good model
    gets about one in a hundred, scattered. A bad one gets them in
    clumps, which is what the bottom panel is for.
    """
    returns = report.returns
    var = report.var
    breaches = report.breaches

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.4),
        height_ratios=(3, 1),
        sharex=True,
        facecolor=_theme["surface"],
    )
    for ax in (top, bottom):
        ax.set_facecolor(_theme["surface"])

    top.plot(
        returns.index,
        returns.to_numpy() * 100,
        color=_theme["muted"],
        linewidth=0.6,
        label="daily return",
        zorder=2,
    )
    top.plot(
        var.index,
        -var.to_numpy() * 100,
        color=_theme["series"][0],
        linewidth=2,
        label=f"{report.confidence:.0%} VaR forecast",
        zorder=3,
    )
    top.scatter(
        returns.index[breaches],
        returns[breaches].to_numpy() * 100,
        s=44,
        color=_theme["critical"],
        edgecolor=_theme["surface"],
        linewidth=1.2,
        label=f"breach ({int(breaches.sum())})",
        zorder=4,
    )

    top.set_ylabel("daily return, %")
    _style(
        top,
        grid_axis="y",
        title=f"Out-of-sample backtest  ·  {int(breaches.sum())} breaches in "
        f"{len(breaches)} days ({report.exceedance_rate:.2%}, "
        f"{1 - report.confidence:.2%} expected)  ·  Basel {report.basel_zone}",
    )
    top.legend(**_legend_kwargs())

    per_quarter = breaches.resample("QE").sum()
    bottom.bar(
        per_quarter.index,
        per_quarter.to_numpy(),
        width=75,
        color=_theme["critical"],
        zorder=3,
    )
    bottom.set_ylabel("breaches\nper quarter")
    _style(bottom, grid_axis="y")
    return fig


def plot_stress(table: pd.DataFrame) -> Figure:
    """Scenario losses, worst first.

    One colour for every bar. Shading them by size would double-encode
    the length the chart already shows and burn the only free channel on
    nothing. Historical scenarios are annotated with how long they took,
    because a 30% fall over 17 months and a 30% fall over 23 days are
    not the same problem.
    """
    _require_columns(table, ["portfolio_return"])

    values = table["portfolio_return"].to_numpy() * 100
    labels = list(table.index)
    positions = np.arange(len(labels))

    fig, ax = _figure(figsize=(9.0, 0.55 * len(labels) + 2.2))

    bars = ax.barh(positions, values, height=0.6, color=_theme["series"][0], zorder=3)
    ax.bar_label(bars, fmt="%.1f%%", padding=-42, color=_theme["surface"], fontsize=9)

    if "n_days" in table.columns:
        for position, days in zip(positions, table["n_days"]):
            if pd.notna(days):
                ax.annotate(
                    f"{int(days)} days",
                    xy=(0, position),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=_theme["muted"],
                )

    ax.set_yticks(positions, labels)
    # `stress_table` sorts worst first; barh would put that at the
    # bottom, which reads backwards against the table it came from.
    ax.invert_yaxis()
    ax.set_xlabel("portfolio return, %")
    ax.axvline(0, color=_theme["axis"], linewidth=1, zorder=2)
    _style(ax, grid_axis="x", title="Stress scenarios at current weights")
    return fig


# --- shared chrome ---------------------------------------------------


def _figure(figsize: tuple[float, float]) -> tuple[Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=_theme["surface"])
    ax.set_facecolor(_theme["surface"])
    return fig, ax


def _style(ax: mpl.axes.Axes, grid_axis: str = "y", title: str | None = None) -> None:
    """Recessive chrome: hairline grid, no frame, muted labels.

    The title is set here rather than by the caller because a left
    aligned title is a different Text object from the centred one, and
    styling `ax.title` after `set_title(loc="left")` silently colours an
    empty string -- which looks fine in light mode and leaves black text
    on a black surface in dark mode.
    """
    if title is not None:
        ax.set_title(title, loc="left", color=_theme["text"], fontsize=12)

    ax.grid(axis=grid_axis, color=_theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_theme["axis"])
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(colors=_theme["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(_theme["secondary"])
    ax.yaxis.label.set_color(_theme["secondary"])
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_color(_theme["secondary"])


def _legend_kwargs(loc: str = "best") -> dict:
    return {
        "frameon": False,
        "loc": loc,
        "fontsize": 9,
        "labelcolor": _theme["secondary"],
    }


def _annotate_extreme(
    ax: mpl.axes.Axes, series: pd.Series, at, fmt: str, above: bool = True
) -> None:
    """Direct-label one point: the one the chart is about."""
    value = float(series.loc[at])
    ax.annotate(
        f"{fmt.format(value)}\n{pd.Timestamp(at):%b %Y}",
        xy=(at, value),
        xytext=(0, 12 if above else -28),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=_theme["text"],
    )
    ax.scatter(
        [at],
        [value],
        s=36,
        color=_theme["series"][0],
        edgecolor=_theme["surface"],
        linewidth=1.5,
        zorder=4,
    )


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Expected column(s) {', '.join(missing)}. Pass the table the "
            f"corresponding module produces, unmodified."
        )
