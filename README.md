# Portfolio Risk Model

Market risk measurement for equity portfolios: Value at Risk, Expected
Shortfall, correlation structure and stress testing.

## Status
Work in progress. Data layer done (price loading with an on-disk cache,
simple and log returns, portfolio aggregation); VaR is next.

## Planned scope
- VaR: historical, parametric (normal and Student's t), Monte Carlo
- CVaR / Expected Shortfall
- Covariance and correlation analysis (Pearson, EWMA, rolling)
- Stress testing against historical and hypothetical scenarios
- VaR backtesting (Kupiec POF, Christoffersen independence test)

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

prices = load_prices(["AAPL", "MSFT", "KO"], "2023-01-01", "2024-12-31")
returns = portfolio_returns(
    simple_returns(prices), {"AAPL": 0.4, "MSFT": 0.4, "KO": 0.2}
)
```

Run the tests with `pytest` from the repository root. None of them hit
the network.

## Repository layout
```
src/data.py      price download and on-disk cache
src/returns.py   simple / log returns, portfolio aggregation
tests/           unit tests
data/            cached price data (not versioned)
```

## Data
Daily adjusted closes from Yahoo Finance via `yfinance`, cached under
`data/` so a result computed today can be reproduced tomorrow. Nothing
is committed to the repository: prices are downloaded at runtime.

Prices are restricted to dates on which every requested ticker traded.
Forward-filling a missing price instead would invent a zero return,
which biases volatility down and correlations towards zero.
