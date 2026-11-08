"""Portfolio risk desk -- the GUI.

Everything here is presentation. The numbers come from `src/`, which
knows nothing about Streamlit and is tested without it. That split is
the point: a chart in this file cannot disagree with the model, because
it has no arithmetic of its own to disagree with.

Run it with `start.bat`, or `streamlit run app.py`.
"""

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from src.backtest import rolling_var_backtest
from src.covariance import correlation, ewma_volatility, rolling_correlation
from src.cvar import historical_cvar
from src.data import DataError, load_prices
from src.money import (
    USD_PLN,
    build_portfolio,
    convert_prices,
    format_money,
    normalise_weights,
    warn_if_stale,
)
from src.plots import (
    plot_backtest,
    plot_ewma_volatility,
    plot_return_distribution,
    plot_rolling_correlation,
    plot_stress,
    plot_var_comparison,
    thin_for_plotting,
)
from src.returns import portfolio_returns, simple_returns
from src.stress import stress_table, uniform_shock
from src.var import historical_var, normal_var, var_summary

st.set_page_config(page_title="Portfolio Risk Desk", page_icon="▦", layout="wide")

DEFAULT_HOLDINGS = pd.DataFrame(
    {
        "Ticker": ["NVDA", "META", "MU", "NOW"],
        "Weight %": [30.0, 30.0, 20.0, 20.0],
    }
)



# --- data, cached ----------------------------------------------------
#
# Streamlit re-runs this whole file on every widget change. Without
# caching, moving the confidence slider would re-download nine years of
# prices. The cache key is the arguments, so changing the tickers or the
# dates still refetches -- which is what the user means by changing them.


@st.cache_data(show_spinner="Fetching prices from Yahoo Finance...")
def fetch_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Stock prices plus the exchange rate, on one shared calendar.

    The two are loaded separately and joined here rather than requested
    in one call. Currencies trade through US market holidays, so the FX
    calendar is a superset of the equity one: asking for both at once
    makes `load_prices` report that it dropped a few per cent of dates,
    which sounds like a data problem and is not one. There is no stock
    return to measure on a day the exchange was shut, so those dates
    carry nothing and are dropped here deliberately and silently.

    Warnings from the stock load are passed back untouched -- a genuine
    calendar mismatch *between stocks* still deserves to be shown.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stocks = load_prices(tickers, start, end)
    fx = load_prices(USD_PLN, start, end)

    return stocks.join(fx, how="inner"), [str(w.message) for w in caught]


@st.cache_data(show_spinner="Running the backtest...")
def run_backtest(returns: pd.Series, window: int, confidence: float, method: str):
    estimator = {"historical": historical_var, "normal": normal_var}[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rolling_var_backtest(returns, estimator, window=window, confidence=confidence)



# --- sidebar: everything the user controls ---------------------------

st.sidebar.title("▦ Portfolio Risk Desk")

st.sidebar.subheader("Holdings")
st.sidebar.caption("Edit the cells; add rows with the button below the table.")

holdings = st.sidebar.data_editor(
    DEFAULT_HOLDINGS,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance symbol, e.g. NVDA"),
        "Weight %": st.column_config.NumberColumn(
            "Weight %", min_value=0.0, step=1.0, format="%.1f"
        ),
    },
    key="holdings",
)

st.sidebar.subheader("Portfolio size")
size_currency = st.sidebar.radio(
    "Amount is in", ["USD", "PLN"], horizontal=True, key="size_currency"
)
total_value = st.sidebar.number_input(
    f"Portfolio value ({size_currency})",
    min_value=1_000.0,
    value=100_000.0,
    step=10_000.0,
    format="%.0f",
    key="total_value",
)

st.sidebar.subheader("Period and parameters")
col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("From", pd.Timestamp("2016-01-01"), format="YYYY-MM-DD")
end_date = col_b.date_input("To", pd.Timestamp.today().normalize(), format="YYYY-MM-DD")

confidence = st.sidebar.select_slider(
    "Confidence level",
    options=[0.90, 0.95, 0.975, 0.99, 0.995],
    value=0.99,
    format_func=lambda v: f"{v:.1%}",
    key="confidence",
)

report_currency = st.sidebar.radio(
    "Measure risk in",
    ["USD", "PLN"],
    horizontal=True,
    key="report_currency",
    help=(
        "PLN restates prices in zloty before anything is measured, so the figures "
        "include USD/PLN risk. USD measures the equity risk on its own."
    ),
)

backtest_window = st.sidebar.slider("Backtest window (days)", 100, 750, 250, step=50)
ewma_lambda = st.sidebar.slider("EWMA lambda", 0.80, 0.99, 0.94, step=0.01)
correlation_window = st.sidebar.slider("Correlation window (days)", 20, 250, 60, step=10)


# --- validate the inputs before touching the network -----------------

clean = holdings.dropna(subset=["Ticker"]).copy()
clean["Ticker"] = clean["Ticker"].astype(str).str.strip().str.upper()
clean = clean[clean["Ticker"] != ""]

if clean.empty:
    st.info("Add at least one holding in the panel on the left.")
    st.stop()

if clean["Ticker"].duplicated().any():
    repeated = ", ".join(sorted(clean.loc[clean["Ticker"].duplicated(), "Ticker"].unique()))
    st.error(f"{repeated} appears more than once. Combine the weights into a single row.")
    st.stop()

raw_weights = dict(zip(clean["Ticker"], clean["Weight %"].fillna(0.0)))
if sum(raw_weights.values()) <= 0:
    st.error("The weights add up to zero. Enter some positive numbers.")
    st.stop()

if start_date >= end_date:
    st.error("The start date must come before the end date.")
    st.stop()

tickers = tuple(clean["Ticker"])

try:
    prices_usd, load_warnings = fetch_prices(tickers, str(start_date), str(end_date))
except DataError as exc:
    st.error(f"Could not load prices: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 -- the GUI must not die on a bad ticker
    st.error(f"Something went wrong while loading prices: {exc}")
    st.stop()

fx_rate = float(prices_usd[USD_PLN].iloc[-1])
fx_date = prices_usd.index[-1]

portfolio = build_portfolio(raw_weights, total_value, size_currency, fx_rate=fx_rate)
_, weight_sum = normalise_weights(raw_weights)

# The risk currency decides which price series everything is built on.
prices = (
    convert_prices(prices_usd) if report_currency == "PLN" else prices_usd.drop(columns=[USD_PLN])
)
returns = simple_returns(prices)
portfolio_series = portfolio_returns(returns, portfolio.weights)


def money(fraction: float) -> str:
    return format_money(portfolio.amount(fraction, report_currency), report_currency)


# --- header ----------------------------------------------------------

st.title("Portfolio Risk Desk")

head = st.columns(4)
head[0].metric(
    "Portfolio value",
    format_money(
        portfolio.total_usd if report_currency == "USD" else portfolio.total_pln, report_currency
    ),
)
head[1].metric("USD/PLN", f"{fx_rate:.4f}", help=f"As of {fx_date:%Y-%m-%d}")
head[2].metric("Trading days", f"{len(prices):,}")
head[3].metric("Period", f"{prices.index.min():%Y-%m-%d} → {prices.index.max():%Y-%m-%d}")

if abs(weight_sum - 100.0) > 0.01:
    st.caption(
        f"⚖️ Weights added up to {weight_sum:.2f}% and were rescaled to 100%. "
        f"The proportions between holdings are unchanged."
    )

if report_currency == "PLN":
    st.caption(
        "💱 Prices are restated in zloty **before** any risk is measured, so every "
        "figure below includes USD/PLN risk. Switch to USD for the equity risk alone."
    )

for message in load_warnings:
    st.warning(message)

stale = warn_if_stale(fx_date)
if stale:
    st.warning(f"💱 {stale}")

tabs = st.tabs(["Portfolio", "VaR & ES", "Correlation", "Stress tests", "Backtest"])


# --- tab 1: the portfolio itself -------------------------------------

with tabs[0]:
    st.subheader("Positions")

    positions = portfolio.positions.copy()
    positions.index.name = "Ticker"
    last_prices = prices_usd.drop(columns=[USD_PLN]).iloc[-1]
    positions["Price USD"] = last_prices
    positions["Shares (approx.)"] = (positions["value_usd"] / last_prices).round(0)

    st.dataframe(
        positions.rename(
            columns={"weight": "Weight", "value_usd": "Value USD", "value_pln": "Value PLN"}
        ),
        width="stretch",
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.2f%%", help="After rescaling"),
            "Value USD": st.column_config.NumberColumn(format="$%.0f"),
            "Value PLN": st.column_config.NumberColumn(format="%.0f zł"),
            "Price USD": st.column_config.NumberColumn(format="$%.2f"),
            "Shares (approx.)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "Share counts are indicative: rounded, and ignoring commission and "
        "fractional shares."
    )

    st.subheader("Cumulative return")
    growth = (1 + returns).cumprod()
    growth = growth / growth.iloc[0]
    chart_data, frequency = thin_for_plotting(growth)

    st.line_chart(chart_data, height=320)

    detail = "" if frequency == "daily" else f" Drawn at {frequency} resolution so the chart pans smoothly."
    st.caption(f"Prices in {report_currency}, rebased to 1 at the start of the period.{detail}")


# --- tab 2: VaR and ES -----------------------------------------------

with tabs[1]:
    st.subheader(f"What one bad day costs · {confidence:.1%} confidence")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = var_summary(portfolio_series, confidence=confidence, seed=1)
    thin_tail = [str(w.message) for w in caught if "tail" in str(w.message)]

    headline = st.columns(4)
    headline[0].metric(
        "VaR, historical",
        money(summary.loc["historical", "var"]),
        f"{summary.loc['historical', 'var']:.2%}",
        delta_color="off",
    )
    headline[1].metric(
        "ES, historical",
        money(summary.loc["historical", "expected_shortfall"]),
        f"{summary.loc['historical', 'expected_shortfall']:.2%}",
        delta_color="off",
    )
    headline[2].metric(
        "VaR, normal",
        money(summary.loc["normal", "var"]),
        f"{summary.loc['normal', 'var']:.2%}",
        delta_color="off",
    )
    headline[3].metric(
        "VaR, Student's t",
        money(summary.loc["student_t", "var"]),
        f"{summary.loc['student_t', 'var']:.2%}",
        delta_color="off",
    )

    if thin_tail:
        st.warning(
            f"At {confidence:.1%} over {len(portfolio_series)} observations only "
            f"{len(portfolio_series) * (1 - confidence):.1f} days fall in the tail, so the "
            f"historical figure rests on a handful of points. Lengthen the period or "
            f"lower the confidence level."
        )

    money_table = summary.copy()
    money_table["VaR"] = money_table["var"].map(money)
    money_table["ES"] = money_table["expected_shortfall"].map(money)
    money_table["VaR %"] = money_table["var"].map("{:.2%}".format)
    money_table["ES %"] = money_table["expected_shortfall"].map("{:.2%}".format)
    money_table["ES / VaR"] = money_table["es_var_ratio"].map("{:.2f}".format)
    money_table.index = money_table.index.str.replace("_", " ")
    money_table.index.name = "Method"

    st.dataframe(money_table[["VaR %", "VaR", "ES %", "ES", "ES / VaR"]], width="stretch")

    st.markdown(
        f"**VaR** is a threshold: the loss you will not exceed on {confidence:.0%} of days. "
        f"**ES** is the average loss on the other {1 - confidence:.0%} — it says how bad "
        "it gets once it is bad. The spread between the methods measures how much of "
        "the answer comes from the data and how much from the distribution assumed."
    )

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_var_comparison(summary), width="stretch")
    with right:
        st.pyplot(
            plot_return_distribution(
                portfolio_series,
                historical_var(portfolio_series, confidence),
                historical_cvar(portfolio_series, confidence),
                confidence,
            ),
            width="stretch",
        )


# --- tab 3: correlation and volatility -------------------------------

with tabs[2]:
    st.subheader("How the holdings move together")

    if len(tickers) < 2:
        st.info("Correlation needs at least two holdings.")
    else:
        corr = correlation(returns)
        st.dataframe(
            corr.style.background_gradient(cmap="Blues", vmin=-1, vmax=1).format("{:.3f}"),
            width="stretch",
        )

        rolling = rolling_correlation(returns, window=correlation_window)
        stats = st.columns(3)
        stats[0].metric("Correlation now", f"{rolling.iloc[-1]:.2f}")
        stats[1].metric("Median", f"{rolling.median():.2f}")
        stats[2].metric(
            "Peak", f"{rolling.max():.2f}", f"{rolling.idxmax():%Y-%m-%d}", delta_color="off"
        )

        st.pyplot(plot_rolling_correlation(rolling, correlation_window), width="stretch")
        st.caption(
            "Correlations rise in a sell-off: diversification weakens exactly when it "
            "is needed."
        )

    st.subheader("Conditional volatility")
    vol = ewma_volatility(portfolio_series, lam=ewma_lambda, annualise=True)
    vol_stats = st.columns(3)
    vol_stats[0].metric("Volatility now", f"{vol.iloc[-1]:.1%}")
    vol_stats[1].metric("Median", f"{vol.median():.1%}")
    vol_stats[2].metric("Peak", f"{vol.max():.1%}", f"{vol.idxmax():%Y-%m-%d}", delta_color="off")
    st.pyplot(plot_ewma_volatility(vol), width="stretch")


# --- tab 4: stress tests ---------------------------------------------

with tabs[3]:
    st.subheader("What a repeat of each crisis would cost")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            table = stress_table(prices, portfolio.weights)
        except DataError as exc:
            table = None
            st.error(str(exc))
        skipped = [str(w.message) for w in caught]

    if table is not None:
        shown = table.copy()
        shown["Loss"] = shown["portfolio_return"].map(money)
        shown["Loss %"] = shown["portfolio_return"].map("{:.2%}".format)
        shown = shown.rename(
            columns={"worst_contributor": "Worst contributor", "n_days": "Trading days"}
        )
        shown.index.name = "Scenario"
        st.dataframe(
            shown[["Loss %", "Loss", "Worst contributor", "Trading days"]], width="stretch"
        )
        st.pyplot(plot_stress(table), width="stretch")

    for message in skipped:
        st.info(
            f"ℹ️ {message}\n\nMove the start date back to cover more scenarios — the "
            f"sample has to span the whole crisis window."
        )

    st.divider()
    st.subheader("Your own shock")
    st.caption(
        "No probability is attached to this and none should be. It answers "
        "\"what would it cost\", not \"will it happen\"."
    )

    shock = st.slider("Every holding falls by", -80, 0, -30, step=5, format="%d%%")
    custom = uniform_shock(f"Equities {shock}%", portfolio.weights.index, shock / 100)
    loss = stress_table(prices, portfolio.weights, [custom])["portfolio_return"].iloc[0]

    shock_cols = st.columns(2)
    shock_cols[0].metric("Portfolio loss", money(loss), f"{loss:.1%}", delta_color="off")
    shock_cols[1].metric(
        "What is left",
        money(1 + loss) if loss > -1 else money(0.0),
        help="Portfolio value after the shock",
    )


# --- tab 5: backtest -------------------------------------------------

with tabs[4]:
    st.subheader("Does the model actually work")

    method = st.radio(
        "VaR method to test", ["historical", "normal"], horizontal=True, key="method"
    )

    if len(portfolio_series) <= backtest_window + 50:
        st.warning(
            f"Not enough data: {len(portfolio_series)} days against a {backtest_window}-day "
            f"window. Lengthen the period or shorten the window on the left."
        )
    else:
        report = run_backtest(portfolio_series, backtest_window, confidence, method)

        zone_colour = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[report.basel_zone]
        cols = st.columns(4)
        cols[0].metric("Breaches", f"{int(report.breaches.sum())} / {len(report.breaches)}")
        cols[1].metric(
            "Rate",
            f"{report.exceedance_rate:.2%}",
            f"{1 - confidence:.2%} expected",
            delta_color="off",
        )
        cols[2].metric("Basel zone", f"{zone_colour} {report.basel_zone}")
        cols[3].metric("Worst breach", money(-report.returns.min()))

        verdict = report.summary()
        verdict.index = [
            "Kupiec — is the number of breaches right?",
            "Christoffersen — are they spread out in time?",
            "Joint — both at once",
        ]
        verdict.index.name = "Test"
        st.dataframe(
            verdict.rename(
                columns={
                    "statistic": "Statistic",
                    "p_value": "p",
                    "rejects_at_5pct": "Rejects at 5%",
                }
            ),
            width="stretch",
            column_config={"p": st.column_config.NumberColumn(format="%.4f")},
        )

        if report.independence.rejects(0.05) and not report.kupiec.rejects(0.05):
            st.warning(
                "The model gets the **number** of breaches right but not their "
                "**timing** — they arrive in clusters. That is the worse failure: the "
                "model is correct on average and blind during the weeks that mattered."
            )
        elif report.kupiec.rejects(0.05):
            st.error(
                "The model breaches its VaR significantly more often (or less often) "
                "than it promises. Try the other method and compare."
            )
        else:
            st.success("Neither test rejects the model at the 5% level.")

        st.pyplot(plot_backtest(report), width="stretch")
        st.caption(
            f"Each VaR point is estimated from the {backtest_window} days before it, never "
            f"from the future. The lower panel counts breaches per quarter, so clustering "
            f"is visible rather than inferred."
        )
