# README

## Overview
This project showcases several statistical and time-series concepts through Python scripts, including:
1. Descriptive statistics and distribution fitting (Normal vs. T).
2. Covariance matrix manipulation and nearest positive semi-definite (PSD) approximation.
3. Conditional distributions using multivariate normal assumptions.
4. Simulation and analysis of AR/MA processes.
5. Exponentially-weighted covariance matrix and PCA analysis for returns data.
6. Covariance matrix simulation methods (Cholesky vs. PCA) and performance comparisons.

Each **Problem (1 through 6)** is subdivided into steps (a, b, c, etc.) with inline comments describing what the code is doing.

---

## Files and Data

### Code File
- **`main.py`** (or any Python file containing the code blocks provided)

### Data Files
- **`problem1.csv`**: Used in Problem 1 for descriptive statistics and distribution fitting.  
- **`problem2.csv`**: Used in Problem 2 to demonstrate covariance matrices and PSD adjustments.  
- **`problem3.csv`**: Used in Problem 3 for conditional distribution examples.  
- **`problem4.csv`**: Used in Problem 4 for time-series analysis (ARIMA, etc.).  
- **`problem6.csv`**: Used in Problem 6 for covariance simulation (Cholesky vs. PCA).  
- **`DailyReturn.csv`**: Used in Problem 5 to compute the exponentially-weighted covariance and PCA.  

Make sure these CSV files are in the same directory as the code file **or** that the code is updated with the correct file paths.

---

## Requirements

You will need a Python 3 environment with the following packages installed:

1. **pandas**  
2. **numpy**  
3. **scipy**  
4. **matplotlib**  
5. **statsmodels**  
6. **time**  

--- 

# Detailed Steps
## P1

1. Import and describe the data (problem1.csv):
    - Reads in a single-column CSV into pandas.DataFrame.
    - Calculates mean, variance, skewness, and excess kurtosis using numpy and scipy.stats.

2. Compare Normal and T distribution fits:
    - Uses scipy.stats.norm.fit and scipy.stats.t.fit to estimate parameters.
    - Computes log-likelihood and AIC for each fit, printing comparison results.

3. Plot distributions:
    - Plots a histogram of the data overlayed with the fitted Normal and T-distribution pdfs.

## P2
1. Covariance Matrix:
    - Reads a multi-column CSV (problem2.csv) and computes the pairwise covariance matrix.

2. Check Positive Semi-Definiteness:
    - Computes eigenvalues of the covariance matrix. Checks if all eigenvalues ≥ 0.

3. Nearest Correlation (and Covariance) Matrix:
    - Demonstrates two methods for finding the nearest PSD (or correlation) matrix:
        - An iterative algorithm that projects to PSD and sets diagonal to 1.
        - A Rebonato-Jäckel-based approach.
    - Constructs the corresponding nearest PSD covariance matrix from the nearest correlation matrix.

4. Handling Missing Data:
    - Demonstrates dropping missing values (.dropna()) and recomputing the covariance matrix.

## p3
1. Read bivariate data (problem3.csv) and compute sample mean vector and covariance.
2. Conditional mean and variance of a bivariate Normal:
    - Method 1: Direct formula using partitioned covariance matrices.
    - Method 2: Linear regression approach to express conditional mean.
3. Monte Carlo simulation to verify conditional distribution:
    - Uses numpy.random to draw from the bivariate normal distribution after applying the Cholesky decomposition.

## p4
1. Simulate MA(1), MA(2), and MA(3) processes:
    - A custom function, simulate_ma_process(n, theta), generates MA(q) data.
    - Plots the ACF and PACF for each simulated series.

2. Simulate AR(1), AR(2), AR(3) processes:
    - A custom function, simulate_ar_process(n, phi), generates AR(p) data.
    - Again plots the ACF and PACF.

3. Analyze a real time series from problem4.csv:
    - Plots the ACF and PACF to identify potential AR/MA orders.

4. Model selection:
-  Loops over different AR(p) and MA(q) orders, fitting via statsmodels.tsa.arima.model.ARIMA and computing the AIC.

## P5
1. Exponential Weighted Covariance:
    - Reads in DailyReturn.csv (asset returns).
    - Implements a function, exp_weighted_cov(returns, lambda_val), to compute the RiskMetrics-style EWMA covariance.

2. Compare EWMAs for different lambdas:
    - Applies PCA to each covariance matrix and plots the cumulative explained variance to show how choice of λ affects the covariance structure.

## p6
1. Cholesky-based simulation:
    - Reads in problem6.csv as a covariance matrix.
    - Ensures it is PSD via the nearest correlation matrix routine.
    - Performs Cholesky decomposition and simulates correlated samples.

2. PCA-based simulation:
    - Uses eigendecomposition of the same PSD covariance matrix.
    - Selects the principal components to capture ≥ 75% variance (example threshold).
    - Simulates correlated samples from these principal components.

3. Performance Comparison:
    - Records runtime for both methods.
    - Compares the sample covariance from the simulations with the original PSD matrix, computing the Frobenius norm differences.

4. Cumulative Variance Comparison:
    - Plots how well the sample covariance from each method reproduces the eigenvalue spectrum of the original PSD matrix.

5. Speed Conclusion:
Prints which method (Cholesky or PCA) is faster based on the timed results.