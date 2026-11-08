# Portfolio Risk Desk

A desktop risk tool for an equity portfolio. Type in your holdings and
their weights, say how much money is in it, and get Value at Risk,
Expected Shortfall, correlation structure, stress scenarios and a
backtest that checks whether any of it holds up — in dollars and in
zloty.

Prices come from Yahoo Finance at runtime. Nothing is committed.

## Running it

Double-click **`start.bat`**. It opens in your browser.

First time only, create the environment:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

If you prefer a terminal: `streamlit run app.py`.

**After editing anything in `src/`, restart the app.** Streamlit reloads
`app.py` when it changes, but a module that is already imported can stay
in memory as the version that was there at startup. Adding a function to
`src/` and then importing it from `app.py` is the case that bites:
`app.py` reloads, the module does not, and you get an ImportError for a
function that is plainly sitting in the file. Closing the console window
and launching again clears it. Editing `app.py` alone needs no restart.

## What you can change

Everything lives in the left-hand panel; the whole page recomputes as
you change it.

| | |
|---|---|
| **Holdings** | ticker and weight per row, add and remove rows freely |
| **Portfolio value** | entered in dollars or zloty, converted at the live rate |
| **Period** | the estimation sample |
| **Confidence** | 90% to 99.5%; 99% is the Basel default, 97.5% is FRTB's for ES |
| **Report currency** | USD or PLN — see below, the difference is not cosmetic |
| **Backtest window** | how much history each VaR forecast is estimated from |
| **EWMA λ** | how fast the volatility estimate forgets |

Weights do not have to add up to 100. They are rescaled, and the app
says so when it does.

## Dollars or zloty is a real choice

A Polish investor holding US stocks carries two risks: the stocks can
fall, and the dollar can fall against the zloty. Converting a dollar
VaR into zloty at today's rate measures only the first one and prints
it in zloty, which understates the exposure.

So **PLN** does not relabel the dollar figure. It restates the prices
themselves in zloty before anything is measured, which puts the FX move
inside the return series where every estimator picks it up. **USD**
measures the equity risk on its own. Both are right; they answer
different questions, and the app says which one it is answering.

## What it measures

- **VaR** — the loss you will not exceed on a given share of days.
  Four estimators: historical, normal, Student's t, Monte Carlo.
- **Expected Shortfall** — the average loss on the days VaR *is*
  exceeded. VaR is a threshold; ES reads the shape of the tail past it.
- **Correlation** — Pearson matrix, plus rolling average pairwise
  correlation, which shows diversification weakening in a sell-off.
- **EWMA volatility** — conditional volatility that reacts in days
  rather than in a year.
- **Stress tests** — seven real crisis windows replayed at your current
  weights, plus a shock you set yourself. A scenario the price history
  does not cover is reported as skipped, not quietly truncated.
- **Backtest** — Kupiec on the exceedance rate, Christoffersen on their
  timing, Basel's traffic light. Each day's VaR is estimated from the
  days before it, so every comparison is out of sample.

The four VaR estimators disagree on purpose. The spread between them
measures how much of the answer comes from the data and how much from
the distribution that was assumed.

## Layout

```
app.py             the GUI; presentation only, no arithmetic
start.bat          double-click launcher
src/data.py        price download and on-disk cache
src/returns.py     simple / log returns, portfolio aggregation
src/money.py       position sizing, USD/PLN, price conversion
src/var.py         four VaR estimators and the summary table
src/cvar.py        Expected Shortfall for each of the four
src/covariance.py  sample and EWMA covariance, rolling correlation
src/stress.py      historical and hypothetical scenarios
src/backtest.py    Kupiec, Christoffersen, Basel traffic light
src/plots.py       the charts
tests/             228 tests, none of which touch the network
data/              cached prices (not versioned)
```

`app.py` holds no arithmetic of its own. Every number on the screen is
computed in `src/`, which knows nothing about Streamlit and is tested
without it — so a chart cannot disagree with the table beside it.

## Conventions

VaR and Expected Shortfall are **positive losses**: `0.0577` means a
loss of 5.77%. Stress results are signed returns instead, because some
scenarios are positive for some holdings. Everything takes *simple*
returns, since only those add up across assets.

Prices are kept only for dates on which every holding traded.
Forward-filling a missing price would invent a zero return, which
biases volatility down and correlations towards zero.

## Tests

```bash
.venv\Scripts\python -m pytest
```

228 tests. None of them reach the network: the one function that talks
to Yahoo is replaced in the fixtures, and so is the price cache. The
GUI is covered too — Streamlit's `AppTest` renders `app.py` in-process
and the tests assert that no tab throws, that the headline figures are
money rather than fractions, and that an empty or mis-weighted
portfolio prompts the user instead of crashing.
