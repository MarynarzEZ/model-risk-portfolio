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
from src.cvar import historical_cvar, normal_cvar, student_t_cvar
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
)
from src.returns import portfolio_returns, simple_returns
from src.stress import HISTORICAL_SCENARIOS, stress_table, uniform_shock
from src.var import historical_var, normal_var, student_t_var, var_summary

st.set_page_config(page_title="Portfolio Risk Desk", page_icon="▦", layout="wide")

DEFAULT_HOLDINGS = pd.DataFrame(
    {
        "Ticker": ["NVDA", "META", "MU", "NOW"],
        "Waga %": [30.0, 30.0, 20.0, 20.0],
    }
)


# --- data, cached ----------------------------------------------------
#
# Streamlit re-runs this whole file on every widget change. Without
# caching, moving the confidence slider would re-download nine years of
# prices. The cache key is the arguments, so changing the tickers or the
# dates still refetches -- which is what the user means by changing them.


@st.cache_data(show_spinner="Pobieram ceny z Yahoo Finance...")
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


@st.cache_data(show_spinner="Liczę backtest...")
def run_backtest(returns: pd.Series, window: int, confidence: float, method: str):
    estimator = {"historyczna": historical_var, "normalna": normal_var}[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rolling_var_backtest(returns, estimator, window=window, confidence=confidence)


# --- sidebar: everything the user controls ---------------------------

st.sidebar.title("▦ Portfolio Risk Desk")

st.sidebar.subheader("Skład portfela")
st.sidebar.caption("Edytuj komórki, dodawaj wiersze przyciskiem na dole tabeli.")

holdings = st.sidebar.data_editor(
    DEFAULT_HOLDINGS,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Symbol z Yahoo Finance, np. NVDA"),
        "Waga %": st.column_config.NumberColumn("Waga %", min_value=0.0, step=1.0, format="%.1f"),
    },
    key="holdings",
)

st.sidebar.subheader("Wielkość portfela")
size_currency = st.sidebar.radio("Waluta wpłaty", ["USD", "PLN"], horizontal=True)
total_value = st.sidebar.number_input(
    f"Wartość portfela ({size_currency})",
    min_value=1_000.0,
    value=100_000.0,
    step=10_000.0,
    format="%.0f",
)

st.sidebar.subheader("Okres i parametry")
col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Od", pd.Timestamp("2016-01-01"), format="YYYY-MM-DD")
end_date = col_b.date_input("Do", pd.Timestamp.today().normalize(), format="YYYY-MM-DD")

confidence = st.sidebar.select_slider(
    "Poziom ufności",
    options=[0.90, 0.95, 0.975, 0.99, 0.995],
    value=0.99,
    format_func=lambda v: f"{v:.1%}",
)

report_currency = st.sidebar.radio(
    "Waluta raportu",
    ["USD", "PLN"],
    horizontal=True,
    help=(
        "PLN przelicza ceny na złote przed liczeniem ryzyka, więc wynik zawiera "
        "ryzyko kursowe. USD mierzy samo ryzyko akcji."
    ),
)

backtest_window = st.sidebar.slider("Okno backtestu (dni)", 100, 750, 250, step=50)
ewma_lambda = st.sidebar.slider("EWMA λ", 0.80, 0.99, 0.94, step=0.01)
correlation_window = st.sidebar.slider("Okno korelacji (dni)", 20, 250, 60, step=10)


# --- validate the inputs before touching the network -----------------

clean = holdings.dropna(subset=["Ticker"]).copy()
clean["Ticker"] = clean["Ticker"].astype(str).str.strip().str.upper()
clean = clean[clean["Ticker"] != ""]

if clean.empty:
    st.info("Wpisz co najmniej jedną spółkę w panelu po lewej.")
    st.stop()

if clean["Ticker"].duplicated().any():
    repeated = ", ".join(sorted(clean.loc[clean["Ticker"].duplicated(), "Ticker"].unique()))
    st.error(f"Ten sam ticker występuje więcej niż raz: {repeated}. Zsumuj wagi w jednym wierszu.")
    st.stop()

raw_weights = dict(zip(clean["Ticker"], clean["Waga %"].fillna(0.0)))
if sum(raw_weights.values()) <= 0:
    st.error("Wagi sumują się do zera — wpisz jakieś dodatnie liczby.")
    st.stop()

if start_date >= end_date:
    st.error("Data początkowa musi być wcześniejsza niż końcowa.")
    st.stop()

tickers = tuple(clean["Ticker"])

try:
    prices_usd, load_warnings = fetch_prices(tickers, str(start_date), str(end_date))
except DataError as exc:
    st.error(f"Nie udało się pobrać danych: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 -- the GUI must not die on a bad ticker
    st.error(f"Błąd przy pobieraniu danych: {exc}")
    st.stop()

fx_rate = float(prices_usd[USD_PLN].iloc[-1])
fx_date = prices_usd.index[-1]

portfolio = build_portfolio(raw_weights, total_value, size_currency, fx_rate=fx_rate)
_, weight_sum = normalise_weights(raw_weights)

# The risk currency decides which price series everything is built on.
prices = convert_prices(prices_usd) if report_currency == "PLN" else prices_usd.drop(columns=[USD_PLN])
returns = simple_returns(prices)
portfolio_series = portfolio_returns(returns, portfolio.weights)


def money(fraction: float) -> str:
    return format_money(portfolio.amount(fraction, report_currency), report_currency)


# --- header ----------------------------------------------------------

st.title("Portfolio Risk Desk")

head = st.columns(4)
head[0].metric(
    "Wartość portfela",
    format_money(
        portfolio.total_usd if report_currency == "USD" else portfolio.total_pln, report_currency
    ),
)
head[1].metric("Kurs USD/PLN", f"{fx_rate:.4f}", help=f"Z dnia {fx_date:%Y-%m-%d}")
head[2].metric("Dni sesyjnych", f"{len(prices):,}".replace(",", " "))
head[3].metric("Okres", f"{prices.index.min():%Y-%m-%d} → {prices.index.max():%Y-%m-%d}")

if abs(weight_sum - 100.0) > 0.01:
    st.caption(
        f"⚖️ Wagi sumowały się do {weight_sum:.2f}%, zostały przeskalowane do 100%. "
        f"Proporcje między spółkami są zachowane."
    )

if report_currency == "PLN":
    st.caption(
        "💱 Ceny przeliczone na złote **przed** liczeniem ryzyka, więc wszystkie miary "
        "poniżej zawierają ryzyko kursowe USD/PLN. Przełącz na USD, żeby zobaczyć "
        "samo ryzyko akcji."
    )

for message in load_warnings:
    st.warning(message)

stale = warn_if_stale(fx_date)
if stale:
    st.warning(f"💱 {stale}")

tabs = st.tabs(["Portfel", "VaR i ES", "Korelacje", "Stress testy", "Backtest"])


# --- tab 1: the portfolio itself -------------------------------------

with tabs[0]:
    st.subheader("Pozycje")

    positions = portfolio.positions.copy()
    positions.index.name = "Ticker"
    last_prices = prices_usd.drop(columns=[USD_PLN]).iloc[-1]
    positions["Cena USD"] = last_prices
    positions["Sztuk (ok.)"] = (positions["value_usd"] / last_prices).round(0)

    st.dataframe(
        positions.rename(
            columns={"weight": "Waga", "value_usd": "Wartość USD", "value_pln": "Wartość PLN"}
        ),
        width="stretch",
        column_config={
            "Waga": st.column_config.NumberColumn(format="%.2f%%", help="Po normalizacji"),
            "Wartość USD": st.column_config.NumberColumn(format="$%.0f"),
            "Wartość PLN": st.column_config.NumberColumn(format="%.0f zł"),
            "Cena USD": st.column_config.NumberColumn(format="$%.2f"),
            "Sztuk (ok.)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "Liczba sztuk jest orientacyjna — zaokrąglona i bez uwzględnienia prowizji "
        "czy niepełnych akcji."
    )

    st.subheader("Skumulowany zwrot")
    growth = (1 + simple_returns(prices)).cumprod()
    growth = growth / growth.iloc[0]
    st.line_chart(growth, height=320)
    st.caption(f"Ceny w {report_currency}, znormalizowane do 1 na starcie okresu.")


# --- tab 2: VaR and ES -----------------------------------------------

with tabs[1]:
    st.subheader(f"Ile można stracić w jeden dzień · {confidence:.1%}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = var_summary(portfolio_series, confidence=confidence, seed=1)
    thin_tail = [str(w.message) for w in caught if "tail" in str(w.message)]

    headline = st.columns(4)
    headline[0].metric("VaR historyczny", money(summary.loc["historical", "var"]),
                       f"{summary.loc['historical', 'var']:.2%}", delta_color="off")
    headline[1].metric("ES historyczny", money(summary.loc["historical", "expected_shortfall"]),
                       f"{summary.loc['historical', 'expected_shortfall']:.2%}", delta_color="off")
    headline[2].metric("VaR normalny", money(summary.loc["normal", "var"]),
                       f"{summary.loc['normal', 'var']:.2%}", delta_color="off")
    headline[3].metric("VaR Student t", money(summary.loc["student_t", "var"]),
                       f"{summary.loc['student_t', 'var']:.2%}", delta_color="off")

    if thin_tail:
        st.warning(
            f"Przy {confidence:.1%} i {len(portfolio_series)} obserwacjach w ogonie jest "
            f"tylko {len(portfolio_series) * (1 - confidence):.1f} dni — wynik historyczny "
            f"opiera się na kilku punktach. Wydłuż okres albo obniż poziom ufności."
        )

    money_table = summary.copy()
    money_table["VaR"] = money_table["var"].map(money)
    money_table["ES"] = money_table["expected_shortfall"].map(money)
    money_table["VaR %"] = money_table["var"].map("{:.2%}".format)
    money_table["ES %"] = money_table["expected_shortfall"].map("{:.2%}".format)
    money_table["ES / VaR"] = money_table["es_var_ratio"].map("{:.2f}".format)
    money_table.index.name = "Metoda"

    st.dataframe(
        money_table[["VaR %", "VaR", "ES %", "ES", "ES / VaR"]], width="stretch"
    )

    st.markdown(
        "**VaR** to próg: strata, której nie przekroczysz w "
        f"{confidence:.0%} dni. **ES** to średnia strata w tych pozostałych "
        f"{1 - confidence:.0%} dni — mówi, jak źle jest, gdy już jest źle. "
        "Rozrzut między metodami pokazuje, ile wyniku pochodzi z danych, "
        "a ile z założonego rozkładu."
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
    st.subheader("Struktura zależności")

    if len(tickers) < 2:
        st.info("Korelacje wymagają co najmniej dwóch spółek.")
    else:
        corr = correlation(returns)
        st.dataframe(
            corr.style.background_gradient(cmap="Blues", vmin=-1, vmax=1).format("{:.3f}"),
            width="stretch",
        )

        rolling = rolling_correlation(returns, window=correlation_window)
        stats = st.columns(3)
        stats[0].metric("Korelacja teraz", f"{rolling.iloc[-1]:.2f}")
        stats[1].metric("Mediana", f"{rolling.median():.2f}")
        stats[2].metric("Szczyt", f"{rolling.max():.2f}", f"{rolling.idxmax():%Y-%m-%d}",
                        delta_color="off")

        st.pyplot(plot_rolling_correlation(rolling, correlation_window), width="stretch")
        st.caption(
            "Korelacje rosną w czasie paniki — dywersyfikacja słabnie dokładnie wtedy, "
            "kiedy jest potrzebna."
        )

    st.subheader("Zmienność warunkowa")
    vol = ewma_volatility(portfolio_series, lam=ewma_lambda, annualise=True)
    vol_stats = st.columns(3)
    vol_stats[0].metric("Zmienność teraz", f"{vol.iloc[-1]:.1%}")
    vol_stats[1].metric("Mediana", f"{vol.median():.1%}")
    vol_stats[2].metric("Szczyt", f"{vol.max():.1%}", f"{vol.idxmax():%Y-%m-%d}", delta_color="off")
    st.pyplot(plot_ewma_volatility(vol), width="stretch")


# --- tab 4: stress tests ---------------------------------------------

with tabs[3]:
    st.subheader("Co kosztowałby powtórzony kryzys")

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
        shown["Strata"] = shown["portfolio_return"].map(money)
        shown["Strata %"] = shown["portfolio_return"].map("{:.2%}".format)
        shown = shown.rename(
            columns={"worst_contributor": "Najgorszy wkład", "n_days": "Dni sesyjnych"}
        )
        st.dataframe(
            shown[["Strata %", "Strata", "Najgorszy wkład", "Dni sesyjnych"]],
            width="stretch",
        )
        st.pyplot(plot_stress(table), width="stretch")

    for message in skipped:
        st.info(
            f"ℹ️ {message}\n\nCofnij datę początkową, żeby objąć więcej scenariuszy — "
            f"próbka musi obejmować całe okno kryzysu."
        )

    st.divider()
    st.subheader("Własny wstrząs")
    st.caption("Nie ma przypisanego prawdopodobieństwa i nie powinien mieć. "
               "Odpowiada na pytanie „ile by to kosztowało”, nie „czy to się zdarzy”.")

    shock = st.slider("Wszystkie spółki spadają o", -80, 0, -30, step=5, format="%d%%")
    custom = uniform_shock(f"Akcje {shock}%", portfolio.weights.index, shock / 100)
    loss = stress_table(prices, portfolio.weights, [custom])["portfolio_return"].iloc[0]

    shock_cols = st.columns(2)
    shock_cols[0].metric("Strata portfela", money(loss), f"{loss:.1%}", delta_color="off")
    shock_cols[1].metric(
        "Zostaje",
        money(1 + loss) if loss > -1 else money(0.0),
        help="Wartość portfela po wstrząsie",
    )


# --- tab 5: backtest -------------------------------------------------

with tabs[4]:
    st.subheader("Czy ten model w ogóle działa")

    method = st.radio(
        "Metoda VaR do sprawdzenia", ["historyczna", "normalna"], horizontal=True
    )

    if len(portfolio_series) <= backtest_window + 50:
        st.warning(
            f"Za mało danych: {len(portfolio_series)} dni przy oknie {backtest_window}. "
            f"Wydłuż okres albo zmniejsz okno w panelu po lewej."
        )
    else:
        report = run_backtest(portfolio_series, backtest_window, confidence, method)

        zone_colour = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[report.basel_zone]
        cols = st.columns(4)
        cols[0].metric("Przebicia", f"{int(report.breaches.sum())} / {len(report.breaches)}")
        cols[1].metric(
            "Odsetek", f"{report.exceedance_rate:.2%}", f"oczekiwane {1 - confidence:.2%}",
            delta_color="off",
        )
        cols[2].metric("Strefa Basel", f"{zone_colour} {report.basel_zone}")
        cols[3].metric("Największe przebicie", money(-report.returns.min()))

        verdict = report.summary()
        verdict.index = [
            "Kupiec — czy liczba przebić się zgadza",
            "Christoffersen — czy przebicia są rozrzucone w czasie",
            "Łącznie — oba naraz",
        ]
        st.dataframe(
            verdict.rename(
                columns={
                    "statistic": "Statystyka",
                    "p_value": "p",
                    "rejects_at_5pct": "Odrzuca model (5%)",
                }
            ),
            width="stretch",
            column_config={"p": st.column_config.NumberColumn(format="%.4f")},
        )

        if report.independence.rejects(0.05) and not report.kupiec.rejects(0.05):
            st.warning(
                "Model trafia w **liczbę** przebić, ale nie w ich **rozkład w czasie** — "
                "przychodzą seriami. To gorsza awaria niż zła liczba: model ma rację "
                "średnio, a myli się dokładnie w tygodniach, które mają znaczenie."
            )
        elif report.kupiec.rejects(0.05):
            st.error(
                "Model przebija VaR istotnie częściej (lub rzadziej), niż obiecuje. "
                "Zobacz, czy metoda historyczna wypada lepiej niż normalna."
            )
        else:
            st.success("Żaden z testów nie odrzuca modelu na poziomie 5%.")

        st.pyplot(plot_backtest(report), width="stretch")
        st.caption(
            f"Każdy punkt VaR liczony z {backtest_window} poprzednich dni, nigdy "
            f"z przyszłości. Dolny panel liczy przebicia kwartalnie — skupiska widać gołym okiem."
        )
