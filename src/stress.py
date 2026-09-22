"""Stress testing.

VaR describes the middle of the distribution's bad end, estimated from a
sample that by definition contains no event more extreme than the ones
already in it. Stress testing asks the other question: never mind the
probability, what does *this specific thing* cost?

Two kinds live here, and they answer different questions:

* historical -- replay an episode that really happened, at today's
  weights. Its authority is that nobody has to defend the assumptions:
  the market did this. Its limit is that the next crisis will not be a
  rerun of the last one.
* hypothetical -- impose a shock nobody has lived through. No
  probability is attached and none should be; the number is an answer to
  "what would it take to breach the limit", not a forecast.

Returns are buy-and-hold over the scenario window: the portfolio is set
to `weights` on day one and left alone. Rebalancing daily through a
crash assumes someone was willing and able to keep buying the falling
asset every morning, which is exactly what does not happen when funding
dries up. Buy-and-hold also makes the per-asset contributions add up
exactly to the portfolio's loss, so the attribution is honest.

Losses are reported as signed returns here, not as positive losses:
-0.34 means the portfolio lost 34%. The `var` convention is inverted on
purpose -- a stress result is a return, and some scenarios are positive
for some holdings.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import pandas as pd

from src.data import DataError


@dataclass(frozen=True)
class HistoricalScenario:
    """A window of real market history, replayed at today's weights."""

    name: str
    start: str
    end: str
    note: str = ""

    def __post_init__(self) -> None:
        if pd.Timestamp(self.end) <= pd.Timestamp(self.start):
            raise ValueError(f"{self.name}: end must be after start")


@dataclass(frozen=True)
class HypotheticalScenario:
    """An instantaneous shock, one return per asset."""

    name: str
    shocks: Mapping[str, float]
    note: str = ""

    def __post_init__(self) -> None:
        if not self.shocks:
            raise ValueError(f"{self.name}: a scenario with no shocks says nothing")
        # Frozen protects the attribute binding, not the dict it points
        # at. A scenario that quietly changed between two runs of a
        # report would be very hard to notice.
        object.__setattr__(
            self, "shocks", MappingProxyType({str(k).upper(): float(v) for k, v in self.shocks.items()})
        )


@dataclass(frozen=True)
class StressResult:
    """What a scenario did to the portfolio."""

    name: str
    portfolio_return: float
    contributions: pd.Series = field(repr=False)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    n_days: int | None = None

    @property
    def worst_contributor(self) -> str:
        """The asset that did most of the damage, weight included.

        Not the same as the worst-performing asset: a 40% fall in a 2%
        position matters less than a 12% fall in a 50% one, and it is
        the second that a risk committee needs to hear about.
        """
        return str(self.contributions.idxmin())


# Dates are the widely cited peak-to-trough windows for each episode,
# taken from S&P 500 closing levels. They are round enough to be
# defensible and specific enough to reproduce; shift them by a few days
# and the numbers move by a few percent, which is worth knowing before
# quoting any of this to three decimal places.
HISTORICAL_SCENARIOS: tuple[HistoricalScenario, ...] = (
    HistoricalScenario(
        "Global financial crisis",
        "2007-10-09",
        "2009-03-09",
        "S&P 500 peak to trough, -57% over 17 months",
    ),
    HistoricalScenario(
        "Lehman collapse",
        "2008-09-08",
        "2008-11-20",
        "The acute phase: funding markets froze in weeks, not months",
    ),
    HistoricalScenario(
        "Taper tantrum",
        "2013-05-22",
        "2013-06-24",
        "A rates shock, not a growth shock -- hits long duration hardest",
    ),
    HistoricalScenario(
        "China devaluation",
        "2015-08-10",
        "2015-08-25",
        "Two weeks; tests how fast a portfolio can bleed, not how far",
    ),
    HistoricalScenario(
        "Volmageddon",
        "2018-01-26",
        "2018-02-08",
        "Short-volatility unwind; equities fell with no macro trigger",
    ),
    HistoricalScenario(
        "COVID-19 crash",
        "2020-02-19",
        "2020-03-23",
        "Fastest 30% drawdown on record, 23 trading days",
    ),
    HistoricalScenario(
        "2022 rate shock",
        "2022-01-03",
        "2022-10-12",
        "Equities and bonds fell together, so the usual hedge did not work",
    ),
)


def run_historical(
    prices: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    scenario: HistoricalScenario,
) -> StressResult:
    """Replay a historical window at today's weights.

    `prices` must span the scenario. That is a real constraint, not a
    technicality: stress testing the 2008 crisis needs 2008 prices, so
    the price history has to reach much further back than the one or two
    years used to estimate VaR. A ticker that listed after the scenario
    simply cannot be tested against it, and this raises rather than
    filling the gap with a zero, which would silently report the
    position as having been immune.
    """
    frame = _check_prices(prices)
    start = pd.Timestamp(scenario.start)
    end = pd.Timestamp(scenario.end)

    _require_coverage(frame, scenario, start, end)

    window = frame.loc[start:end]
    if len(window) < 2:
        raise DataError(
            f"{scenario.name}: only {len(window)} trading day(s) of prices fall inside "
            f"{start.date()}..{end.date()}"
        )

    aligned = _weights_for(window.columns, weights)
    asset_returns = window.iloc[-1] / window.iloc[0] - 1.0
    contributions = (aligned * asset_returns).rename(scenario.name)

    return StressResult(
        name=scenario.name,
        portfolio_return=float(contributions.sum()),
        contributions=contributions,
        start=window.index[0],
        end=window.index[-1],
        n_days=len(window),
    )


def run_hypothetical(
    weights: Mapping[str, float] | pd.Series,
    scenario: HypotheticalScenario,
) -> StressResult:
    """Apply an instantaneous shock. No price history needed.

    Every asset must be given a shock. Defaulting the unnamed ones to
    zero would be convenient and would quietly turn a forgotten ticker
    into a position that survives the crisis untouched.
    """
    aligned = _as_named_weights(weights)

    missing = [name for name in aligned.index if name not in scenario.shocks]
    if missing:
        raise ValueError(
            f"{scenario.name}: no shock given for {', '.join(missing)}. Every holding "
            f"needs one -- an unnamed asset would be treated as unaffected."
        )

    shocks = pd.Series({name: scenario.shocks[name] for name in aligned.index})
    contributions = (aligned * shocks).rename(scenario.name)

    return StressResult(
        name=scenario.name,
        portfolio_return=float(contributions.sum()),
        contributions=contributions,
    )


def uniform_shock(name: str, assets: Iterable[str], shock: float) -> HypotheticalScenario:
    """The same move for every holding, e.g. 'equities fall 20%'.

    Crude by design. It is the right shape for an equity-only book,
    where in a real panic the dispersion between names is small next to
    the size of the common move.
    """
    return HypotheticalScenario(name, {str(a).upper(): float(shock) for a in assets})


def stress_table(
    prices: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    scenarios: Sequence[HistoricalScenario | HypotheticalScenario] = HISTORICAL_SCENARIOS,
    *,
    skip_uncovered: bool = True,
) -> pd.DataFrame:
    """Run every scenario and collect the results, worst first.

    `skip_uncovered=True` drops historical scenarios the price history
    does not reach, with a warning naming each one. The default is to
    skip because a table of five crises is still useful when one of the
    tickers only listed in 2015; set it to False when the report has to
    state that a scenario was not tested rather than omit it.
    """
    rows = []
    skipped = []

    for scenario in scenarios:
        try:
            if isinstance(scenario, HistoricalScenario):
                result = run_historical(prices, weights, scenario)
            elif isinstance(scenario, HypotheticalScenario):
                result = run_hypothetical(weights, scenario)
            else:
                raise TypeError(f"Not a scenario: {scenario!r}")
        except DataError as exc:
            if not skip_uncovered:
                raise
            skipped.append(f"{scenario.name} ({exc})")
            continue

        rows.append(
            {
                "scenario": result.name,
                "portfolio_return": result.portfolio_return,
                "worst_contributor": result.worst_contributor,
                "worst_contribution": float(result.contributions.min()),
                "start": result.start,
                "end": result.end,
                "n_days": result.n_days,
            }
        )

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} scenario(s) the price history does not cover: "
            f"{'; '.join(skipped)}",
            stacklevel=2,
        )
    if not rows:
        raise DataError(
            "No scenario could be run. The price history does not reach back far "
            "enough for any of them -- load a longer sample."
        )

    return pd.DataFrame(rows).sort_values("portfolio_return").set_index("scenario")


# --- internals -------------------------------------------------------


def _check_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Stress tests run on prices, not returns: the level is the point."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Stress tests take a DataFrame of prices, one column per asset")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Prices must be indexed by date")
    if prices.empty:
        raise DataError("No prices given")
    if (prices <= 0).any(axis=None):
        raise DataError("Prices must be strictly positive")
    return prices.sort_index()


def _require_coverage(
    frame: pd.DataFrame,
    scenario: HistoricalScenario,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    """Refuse to report a partial window as if it were the whole thing."""
    first, last = frame.index[0], frame.index[-1]
    if first > start or last < end:
        raise DataError(
            f"{scenario.name} runs {start.date()}..{end.date()} but the prices only "
            f"cover {first.date()}..{last.date()}. Reload with an earlier start date; "
            f"a truncated window would understate the scenario."
        )


def _weights_for(
    columns: pd.Index, weights: Mapping[str, float] | pd.Series
) -> pd.Series:
    """Reuse the validation from `returns`, keeping one weight contract."""
    from src.returns import _align_weights

    return _align_weights(columns, weights)


def _as_named_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    """A hypothetical scenario has no columns to match against."""
    if isinstance(weights, pd.Series):
        weights = weights.to_dict()
    if not isinstance(weights, Mapping):
        raise TypeError(
            "A hypothetical scenario needs weights keyed by ticker, since there are no "
            "price columns to match a bare sequence against."
        )

    named = pd.Series({str(k).upper(): float(v) for k, v in weights.items()}, dtype="float64")
    total = float(named.sum())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Weights must sum to 1, got {total:.6f}")
    return named
