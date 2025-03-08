# Financial Analysis Project

This repository contains scripts for financial risk assessment, options pricing, and portfolio analysis.

## Prerequisites

The following Python packages are required:
- pandas
- numpy
- scipy
- matplotlib

Install the required packages using:
```bash
pip install pandas numpy scipy matplotlib
```

## Dataset

The code requires a file named `DailyPrices.csv` with the following structure:
- Date column (which will be used as index)
- Price columns for different stocks (including SPY, AAPL, EQIX)

Place this file in the same directory as the script.

## Usage

To run the analysis, simply execute:
```bash
python Junjie-P2.py
```

## Features

The script performs various financial analyses:

1. **Return Calculations**
   - Arithmetic and logarithmic returns
   - Demeaning and standard deviation calculations

2. **Portfolio Valuation**
   - Current portfolio value calculation

3. **Risk Metrics**
   - Value at Risk (VaR) using multiple methods:
     - Normal distribution with exponentially weighted covariance
     - T distribution using a Gaussian Copula
     - Historical simulation
     - Delta-Normal approximation
     - Monte Carlo simulation
   - Expected Shortfall (ES) calculation

4. **Options Analysis**
   - Black-Scholes-Merton option pricing
   - Implied volatility calculation
   - Greeks calculation (Delta, Vega, Theta)
   - Put-Call parity verification

5. **Visualization**
   - Portfolio value vs stock price analysis
   - Comparison of risk measures across different methods

## Output

The script outputs results to the console and generates visualizations comparing different risk assessment methods.