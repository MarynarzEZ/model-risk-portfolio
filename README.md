# Portfolio Risk Model

Market risk measurement for equity portfolios: Value at Risk, Expected
Shortfall, correlation structure and stress testing.

## Status
Work in progress. Project setup complete; data layer is next.

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

## Repository layout
```
src/       risk model modules (data loading, returns, VaR, CVaR, ...)
tests/     unit tests
data/      downloaded price data (not versioned)
```

## Data
Daily equity prices from Yahoo Finance via `yfinance`. Nothing is
committed to the repository: prices are downloaded at runtime.
