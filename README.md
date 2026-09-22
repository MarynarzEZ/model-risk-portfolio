# Portfolio Risk Model

Market risk measurement for equity portfolios: Value at Risk, Expected
Shortfall, correlation structure and stress testing.

## Status
Complete: measurement modules, charts, and a report notebook that runs
end to end.

## Scope
- VaR: historical, parametric (normal and Student's t), Monte Carlo
- CVaR / Expected Shortfall, in closed form where one exists
- Covariance and correlation analysis (Pearson, EWMA, rolling)
- Stress testing against historical and hypothetical scenarios
- VaR backtesting (Kupiec POF, Christoffersen independence and
  conditional coverage, Basel traffic light)

### Conventions
VaR and Expected Shortfall are returned as **positive losses**: `0.0392`
means a loss of 3.92% of portfolio value. Stress results are signed
returns instead, because some scenarios are positive for some holdings.
Everything takes *simple* returns, since only those add up across assets.

## Stack
Python 3.13, NumPy, pandas, SciPy, statsmodels, yfinance, matplotlib,
seaborn.

## Getting started

```bash
git clone https://github.com/MarynarzEZ/portfolio-risk-model.git
cd portfolio-risk-model
python -m venv .venv
.venv\Scripts\activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

```python
from src.data import load_prices
from src.returns import portfolio_returns, simple_returns
from src.var import var_summary

weights = {"NVDA": 0.3, "META": 0.3, "MU": 0.2, "NOW": 0.2}
prices = load_prices(list(weights), "2016-01-01", "2024-12-31")
portfolio = portfolio_returns(simple_returns(prices), weights)

print(var_summary(portfolio, confidence=0.99))
```

```
                 var  expected_shortfall  es_var_ratio
method
historical   0.05606             0.07255       1.29415
normal       0.04899             0.05637       1.15066
student_t    0.05795             0.08306       1.43345
monte_carlo  0.04907             0.05623       1.14580
```

The spread is the point. The normal assumption prices the 99% loss at
4.90% and the data says 5.61% — and the gap widens in the Expected
Shortfall column, because that averages the tail rather than sampling
one point on it. The `es_var_ratio` of 1.15 under normality is the
theoretical value for a normal distribution at 99%; that it comes out
at 1.29 historically and 1.43 under a fitted Student's t is the fat
tail being measured rather than assumed away.

Run the tests with `pytest` from the repository root. None of them hit
the network.

## Repository layout
```
src/data.py        price download and on-disk cache
src/returns.py     simple / log returns, portfolio aggregation
src/var.py         four VaR estimators, scaling, summary table
src/cvar.py        Expected Shortfall for each of the four
src/covariance.py  sample and EWMA covariance, rolling correlation
src/stress.py      historical and hypothetical scenarios
src/backtest.py    Kupiec, Christoffersen, Basel traffic light
src/plots.py       report charts
report.ipynb       the whole thing, end to end
tests/             unit tests
data/              cached price data (not versioned)
```

## The report

`report.ipynb` runs everything above and draws it. All the inputs —
tickers, weights, date range, confidence level, EWMA decay, backtest
window — are collected in its first cell; change them there and re-run.

The notebook is committed with its outputs cleared, so the diffs stay
readable. Run it with `jupyter lab report.ipynb`; prices come from the
local cache after the first run, so it takes seconds.

## Data
Daily adjusted closes from Yahoo Finance via `yfinance`, cached under
`data/` so a result computed today can be reproduced tomorrow. Nothing
is committed to the repository: prices are downloaded at runtime.

Prices are restricted to dates on which every requested ticker traded.
Forward-filling a missing price instead would invent a zero return,
which biases volatility down and correlations towards zero.

Stress testing needs a much longer history than VaR estimation does:
testing the 2008 crisis requires 2008 prices. `stress.run_historical`
refuses a window the price history does not cover rather than reporting
a truncated one.

## Does any of it work?

`rolling_var_backtest` re-estimates VaR on a trailing 250-day window and
tests each day out of sample. On the portfolio above at 99%:

| model | exceedances | rate | Basel zone | Kupiec p | Christoffersen p |
|---|---|---|---|---|---|
| historical | 26 / 2013 | 1.29% | green | 0.21 | **0.045** |
| normal | 41 / 2013 | 2.04% | red | **0.0000** | 0.27 |

Both models fail, and they fail differently — which is why the two
tests are separate. The normal model breaches twice as often as it
promises and Kupiec rejects it outright. The historical model gets the
*rate* right and would pass every test a supervisor applies, but
Christoffersen rejects its *timing*: the breaches arrive in clusters,
because an equally weighted window only notices a volatility spike once
the spike is already inside it.

That second failure is the one worth caring about. A model that is
correct on average and blind during the week that mattered is not a
safe model, and nothing but the independence test would have caught it.
