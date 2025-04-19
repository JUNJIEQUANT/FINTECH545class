# Portfolio Analysis and Risk Attribution Toolkit

This repository contains a comprehensive set of financial analysis tools for portfolio performance attribution, risk modeling, and portfolio optimization. The code is organized into five interconnected parts that build upon each other to provide a complete framework for portfolio analysis.

## Dependencies

The code requires the following Python libraries:
- pandas
- numpy
- scikit-learn
- scipy
- tqdm
- matplotlib (for visualization if needed)

Install dependencies using:
```
pip install pandas numpy scikit-learn scipy tqdm matplotlib
```

## Required Data Files

The analysis requires three input files:
- `initial_portfolio.csv`: Contains the holdings information for three portfolios (A, B, C)
- `DailyPrices.csv`: Historical daily prices for all securities with dates
- `rf.csv`: Daily risk-free rates

## Part 1: CAPM Attribution Analysis

This part implements a CAPM-based performance attribution model:

- Splits data into estimation period (through 2023) and holding period (2024+)
- Uses SPY as the market index for CAPM regressions
- Calculates the initial portfolio weights based on market values
- Tracks daily weights as they evolve through the holding period
- Computes systematic (market-related) and idiosyncratic return components
- Implements the Carino linking method for multi-period attribution
- Analyzes both return and risk attribution for each portfolio

The output shows how much of each portfolio's performance and risk can be attributed to market exposure versus security selection.

## Part 2: Maximum Sharpe Ratio Portfolio Optimization

Building on the CAPM parameters from Part 1:

- Assumes zero alpha for all stocks
- Uses the historical average of SPY excess return as the expected market premium
- Constructs the covariance matrix under the CAPM framework
- Optimizes each portfolio for maximum Sharpe ratio, allowing short positions
- Simulates the optimized portfolios over the holding period
- Performs attribution analysis on the optimized portfolios
- Compares original versus optimized portfolio performance

This part demonstrates how modern portfolio theory can be applied to improve risk-adjusted returns.

## Part 3: Distribution Investigation

This part (documentation only) investigates alternative distributions for modeling financial returns:

- Normal Inverse Gaussian (NIG) distribution
- Skew Normal distribution
- Analysis of fat tails and skewness in financial data

## Part 4: Advanced Risk Modeling

This section implements sophisticated risk modeling techniques:

- Fits four distribution types to each stock's returns:
  - Normal
  - Generalized T
  - Normal Inverse Gaussian (NIG)
  - Skew Normal
- Selects the best distribution for each stock using AICc criterion
- Forces zero mean for return distributions
- Implements Gaussian Copula simulation with fitted marginal distributions
- Calculates 1-day Value-at-Risk (VaR) and Expected Shortfall (ES) at 95% confidence
- Compares results between copula approach and multivariate normal assumption

This provides a more accurate risk assessment by capturing non-normal behavior in returns.

## Part 5: Risk Parity Portfolio Construction

The final part creates risk-balanced portfolios:

- Constructs risk parity portfolios using Expected Shortfall as the risk metric
- Uses Spearman rank correlation and fitted distributions from Part 4
- Implements an optimization routine to equalize risk contributions across assets
- Simulates risk parity portfolios over the holding period
- Performs attribution analysis on the risk parity portfolios
- Compares results across all portfolio construction methods

This demonstrates an alternative approach to portfolio construction focused on risk diversification rather than return optimization.

## Usage Instructions

1. Place the three required data files in the same directory as the script
2. Run the script in its entirety to execute all five parts sequentially
3. Analysis outputs will be printed to the console
4. Modify the confidence level (CONF_LEVEL) or simulation parameters if needed

## Key Output Explanation

The code generates several key outputs:

- CAPM attribution tables for original portfolios showing return and volatility contributions
- Maximum Sharpe ratio portfolio weights and comparative attribution analysis
- Best-fit distribution models and parameters for each stock
- 1-day VaR and ES risk measures in both percentage and dollar terms
- Risk parity portfolio weights and attribution results

## Notes

- All portfolio simulations use a buy-and-hold approach with daily rebalancing due to price changes
- Return attribution uses the Carino multi-period linking method to ensure additivity
- The code handles missing data and numerical stability issues in correlation/covariance matrices
- For risk simulations, a 95% confidence level is used by default

This toolkit provides a comprehensive framework for portfolio analysis, combining modern portfolio theory, risk modeling, and performance attribution.