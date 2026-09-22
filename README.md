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

weights = {"AAPL": 0.4, "MSFT": 0.4, "KO": 0.2}
prices = load_prices(list(weights), "2006-01-01", "2024-12-31")
portfolio = portfolio_returns(simple_returns(prices), weights)

print(var_summary(portfolio, confidence=0.99))
```

```
                 var  expected_shortfall  es_var_ratio
method
historical   0.03918             0.05291       1.35044
normal       0.03248             0.03734       1.14961
student_t    0.03873             0.05859       1.51271
monte_carlo  0.03253             0.03724       1.14478
```

The spread is the point. The normal assumption prices the 99% loss at
3.25% and the data says 3.92% — and the gap widens in the Expected
Shortfall column, because that averages the tail rather than sampling
one point on it.

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
tests each day out of sample. On AAPL/MSFT/KO 2006-2024 at 99%:

| model | exceedances | rate | Basel zone | Kupiec p |
|---|---|---|---|---|
| historical | 61 / 4530 | 1.35% | yellow | 0.026 |
| normal | 107 / 4530 | 2.36% | red | 0.0000 |

The normal model breaches more than twice as often as it promises, and
the test rejects it outright. That is the entire argument for not
quoting a parametric VaR without a backtest beside it.
