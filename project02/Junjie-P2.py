import pandas as pd
import numpy as np
from scipy.stats import t,norm
import scipy.stats as st
from scipy.optimize import brentq
import math
import matplotlib.pyplot as plt
#cd D:\545\project02
df = pd.read_csv('DailyPrices.csv', parse_dates=['Date'], index_col='Date')

stocks = ['SPY', 'AAPL', 'EQIX']

# ------------------------------
# 1A. Arithmetic Returns
# ------------------------------
arithmetic_returns = df[stocks].pct_change().dropna()

arithmetic_returns_demeaned = arithmetic_returns - arithmetic_returns.mean()

print("Arithmetic Returns (demeaned) - Last 5 Rows:")
print(arithmetic_returns_demeaned.tail())

std_arithmetic = arithmetic_returns_demeaned.std()
print("\nStandard Deviations (Arithmetic Returns):")
print(std_arithmetic)

# ------------------------------
# 1B. Log Returns
# ------------------------------
log_returns = np.log(df[stocks] / df[stocks].shift(1)).dropna()

log_returns_demeaned = log_returns - log_returns.mean()

print("\nLog Returns (demeaned) - Last 5 Rows:")
print(log_returns_demeaned.tail())

std_log = log_returns_demeaned.std()
print("\nStandard Deviations (Log Returns):")
print(std_log)

# ------------------------------
# 2A.
# ------------------------------

target_date = '2025-01-03'

prices_on_date = df.loc[target_date]
date_used = target_date

spy_price = prices_on_date['SPY']
aapl_price = prices_on_date['AAPL']
eqix_price = prices_on_date['EQIX']

spy_shares = 100
aapl_shares = 200
eqix_shares = 150

portfolio_value = (spy_shares * spy_price) + (aapl_shares * aapl_price) + (eqix_shares * eqix_price)

print(f"Portfolio Value as of {date_used}: ${portfolio_value:,.2f}")


# ------------------------------
# 2Ba.
# ------------------------------

shares_per_stock = {
    'SPY': spy_shares,
    'AAPL': aapl_shares,
    'EQIX': eqix_shares
}
current_stock_prices = {
    'SPY': spy_price,
    'AAPL': aapl_price,
    'EQIX': eqix_price
}
stock_symbols = list(shares_per_stock.keys())
position_values = np.array([shares_per_stock[symbol] * current_stock_prices[symbol] for symbol in stock_symbols])
portfolio_weights = position_values / portfolio_value

def calculate_ewcov(returns, lambda_val=0.97):
    n = returns.shape[0]
    

    weights = (1 - lambda_val) * lambda_val ** np.arange(n-1, -1, -1)
    
    weights /= weights.sum()
    
    weighted_mean = returns.mul(weights, axis=0).sum()
    
    centered_returns = returns - weighted_mean

    sqrt_weights = np.sqrt(weights)
    weighted_centered = centered_returns.mul(sqrt_weights, axis=0)
    
    cov_matrix = np.dot(weighted_centered.T, weighted_centered)
    
    cov_matrix_df = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
    return cov_matrix_df

price_data = df[stock_symbols].copy()
return_data = price_data / price_data.shift(1) - 1.0
return_data.dropna(inplace=True)

zero_mean_returns = return_data.copy()

decay_factor_value = 0.97
exp_weighted_cov = calculate_ewcov(zero_mean_returns, lambda_val=decay_factor_value)
exp_weighted_cov_values = exp_weighted_cov.values

confidence_level = 0.05
z_score_alpha = norm.ppf(confidence_level) 
norm_density_z_score = norm.pdf(z_score_alpha)

portfolio_variance = portfolio_weights @ exp_weighted_cov_values @ portfolio_weights
portfolio_volatility = np.sqrt(portfolio_variance)

portfolio_var = -z_score_alpha * portfolio_volatility * portfolio_value
portfolio_es = portfolio_volatility * (norm_density_z_score / confidence_level) * portfolio_value

stock_var_values = {}
stock_es_values = {}

for i, symbol in enumerate(stock_symbols):
    stock_volatility = np.sqrt(exp_weighted_cov.iloc[i, i])
    stock_position_value = shares_per_stock[symbol] * current_stock_prices[symbol]
    stock_var_values[symbol] = -z_score_alpha * stock_volatility * stock_position_value
    stock_es_values[symbol] = stock_volatility * (norm_density_z_score / confidence_level) * stock_position_value

print("\nIndividual Stock VaR (5%):")
for symbol in stock_symbols:
    print(f"{symbol}: ${stock_var_values[symbol]:,.2f}")

print("\nIndividual Stock Expected Shortfall (5%):")
for symbol in stock_symbols:
    print(f"{symbol}: ${stock_es_values[symbol]:,.2f}")

print("\nPortfolio VaR (5%): ${:,.2f}".format(portfolio_var))
print("Portfolio Expected Shortfall (5%): ${:,.2f}".format(portfolio_es))


# ------------------------------
# 2Bb.
# ------------------------------

df_filtered = df[df['Date'] == target_date]

spy_shares = 100
aapl_shares = 200
eqix_shares = 150
shares_dict = {
    "SPY": spy_shares,
    "AAPL": aapl_shares,
    "EQIX": eqix_shares
}
stock_symbols = list(shares_dict.keys())

spy_price = df_filtered['SPY'].values[0]
aapl_price = df_filtered['AAPL'].values[0]
eqix_price = df_filtered['EQIX'].values[0]
portfolio_value = (spy_shares * spy_price) + (aapl_shares * aapl_price) + (eqix_shares * eqix_price)

print(f"A) Current Portfolio Value on {target_date}: ${portfolio_value:,.2f}")

def calculate_var_es_t_copula(price_data, stock_symbols, shares_dict, alpha=0.05):
    price_data = price_data.copy()
    
    price_data.sort_values("Date", inplace=True)
    
    for stock in stock_symbols:
        price_data[stock] = price_data[stock].pct_change()
    price_data.dropna(inplace=True)
    
    arithmetic_return_remove_mean = {}
    for stock in stock_symbols:
        arr = price_data[stock].values
        arr_no_mean = arr - np.mean(arr)
        arithmetic_return_remove_mean[stock] = arr_no_mean
    
    t_params = {}
    for stock in stock_symbols:
        data = arithmetic_return_remove_mean[stock]
        df_, loc_, scale_ = st.t.fit(data, method="MLE")
        t_params[stock] = (df_, loc_, scale_)

    U = pd.DataFrame()
    for stock in stock_symbols:
        df_, loc_, scale_ = t_params[stock]
        data = arithmetic_return_remove_mean[stock]
        U[stock] = st.t.cdf(data, df_, loc_, scale_)
    
    Z = U.apply(lambda x: st.norm.ppf(x))
    
    R_spearman = Z.corr(method='spearman')
    
    n_samples = 10000
    np.random.seed(42)
    copula_sim = st.multivariate_normal.rvs(
        mean=np.zeros(len(stock_symbols)),
        cov=R_spearman,
        size=n_samples
    )
    
    sim_returns_t = np.zeros_like(copula_sim)
    for i, stock in enumerate(stock_symbols):
        df_, loc_, scale_ = t_params[stock]
        sim_u = st.norm.cdf(copula_sim[:, i])
        sim_returns_t[:, i] = st.t.ppf(sim_u, df_, loc_, scale_)
    
    latest_prices = {}
    for stock in stock_symbols:
        latest_prices[stock] = float(df_filtered[stock].values[0])
    
    sim_pnl_t = np.zeros(n_samples)
    for i, stock in enumerate(stock_symbols):
        shares = shares_dict[stock]
        current_price = latest_prices[stock]
        sim_pnl_t += shares * current_price * sim_returns_t[:, i]
    
    var_value = np.percentile(sim_pnl_t, alpha * 100)
    
    tail_mask = (sim_pnl_t <= var_value)
    if np.any(tail_mask):
        es_value = sim_pnl_t[tail_mask].mean()
    else:
        es_value = var_value
    
    portfolio_value = 0.0
    for stock in stock_symbols:
        portfolio_value += shares_dict[stock] * latest_prices[stock]
    
    stock_var = {}
    stock_es = {}
    for i, stock in enumerate(stock_symbols):
        stock_pnl = shares_dict[stock] * latest_prices[stock] * sim_returns_t[:, i]
        stock_var[stock] = -np.percentile(stock_pnl, alpha * 100)
        stock_tail_mask = (stock_pnl <= np.percentile(stock_pnl, alpha * 100))
        if np.any(stock_tail_mask):
            stock_es[stock] = -stock_pnl[stock_tail_mask].mean()
        else:
            stock_es[stock] = -np.percentile(stock_pnl, alpha * 100)
    
    return -var_value, -es_value, portfolio_value, stock_var, stock_es, sim_pnl_t

var_value, es_value, portfolio_value_check, stock_var, stock_es, sim_pnl = calculate_var_es_t_copula(
    df, stock_symbols, shares_dict, alpha=0.05
)


print(f"\nIndividual Stock VaR (5%):")
for stock in stock_symbols:
    print(f"{stock}: ${stock_var[stock]:,.2f}")
print(f"\nIndividual Stock Expected Shortfall (5%):")
for stock in stock_symbols:
    print(f"{stock}: ${stock_es[stock]:,.2f}")
print(f"\nPortfolio VaR (5%): ${var_value:,.2f}")
print(f"Portfolio Expected Shortfall (5%): ${es_value:,.2f}")

# ------------------------------
# 2Bc.
# ------------------------------
latest_prices = df[stocks].iloc[-1]
arithmetic_returns_demeaned 

portfolio_holdings = {symbol: shares_per_stock[symbol] * latest_prices[symbol] for symbol in stocks}
total_value = sum(portfolio_holdings.values())

position_array = np.array([portfolio_holdings[symbol] for symbol in stocks])

historical_scenarios = arithmetic_returns_demeaned.values @ position_array

portfolio_var = -np.percentile(historical_scenarios, 0.05 * 100)

extreme_losses = historical_scenarios[historical_scenarios <= -portfolio_var]
portfolio_es = -np.mean(extreme_losses) if len(extreme_losses) > 0 else portfolio_var

individual_var = {}
individual_es = {}

for i, symbol in enumerate(stocks):
    stock_dollar_returns = arithmetic_returns_demeaned[symbol].values * portfolio_holdings[symbol]
    
    stock_var = -np.percentile(stock_dollar_returns, 0.05 * 100)
    individual_var[symbol] = stock_var
    
    stock_extreme_losses = stock_dollar_returns[stock_dollar_returns <= -stock_var]
    stock_es = -np.mean(stock_extreme_losses) if len(stock_extreme_losses) > 0 else stock_var
    individual_es[symbol] = stock_es

print(f"Portfolio Value: ${total_value:,.2f}")
print(f"\n{int(0.05 * 100)}% Historical VaR: ${portfolio_var:,.2f}")
print(f"{int(0.05 * 100)}% Historical ES: ${portfolio_es:,.2f}")

# ------------------------------
# 3A.
# ------------------------------
def black_scholes_call(S, K, T, r, sigma):

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def black_scholes_put(S, K, T, r, sigma):

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price


def calculate_implied_volatility(S, K, T, r, market_price, option_type='call'):
    def objective(sigma):
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, sigma) - market_price
        else:
            return black_scholes_put(S, K, T, r, sigma) - market_price
            
    vol_lower, vol_upper = 1e-6, 5.0
    return brentq(objective, vol_lower, vol_upper)
        



S = 31         
K = 30         
T = 0.25        
r = 0.10        
market_price = 3.00  

implied_vol = calculate_implied_volatility(S, K, T, r, market_price)


print(f"\nResults:")
print(f"Implied Volatility: {implied_vol:.4f} ({implied_vol*100:.2f}%)")


# ------------------------------
# 3B.
# ------------------------------
def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    delta = norm.cdf(d1)
    
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100
    
    theta_annual = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta = theta_annual / 365
    
    return delta, vega, theta, call_price


delta, vega, theta, original_price = calculate_greeks(S, K, T, r, implied_vol)

new_vol = implied_vol + 0.01 
_, _, _, new_price = calculate_greeks(S, K, T, r, new_vol)

actual_price_change = new_price - original_price

expected_price_change = vega * 0.01 

print(f"\nGreeks:")
print(f"Delta: {delta:.4f}")
print(f"Vega: ${vega:.4f} per 1 percentage point change in volatility")
print(f"Theta: ${theta:.4f} per year")
print(f"Theta (daily): ${theta/365:.4f} per day")

print(f"\nPrice Change Analysis:")
print(f"Original Option Price: ${original_price:.4f}")
print(f"Original Volatility: {implied_vol:.4f} ({implied_vol*100:.2f}%)")
print(f"New Volatility (1 percentage point increase): {new_vol:.4f} ({new_vol*100:.2f}%)")
print(f"New Option Price: ${new_price:.4f}")
print(f"Actual Price Change: ${actual_price_change:.4f}")
print(f"Expected Price Change (using Vega): ${expected_price_change:.4f}")
print(f"Difference: ${actual_price_change - expected_price_change:.8f}")

epsilon = 0.0001
_, _, _, price_up = calculate_greeks(S, K, T, r, implied_vol + epsilon)
_, _, _, price_down = calculate_greeks(S, K, T, r, implied_vol - epsilon)
estimated_vega = (price_up - price_down) / (2 * epsilon)

print(f"\nVerification:")
print(f"Estimated Vega using finite difference: ${estimated_vega:.4f} per 1 percentage point change")
print(f"Comparison to analytical Vega: ${vega:.4f}")

# ------------------------------
# 3C.
# ------------------------------

put_price = black_scholes_put(S, K, T, r, implied_vol)
print(f"Put Price: ${put_price:.2f}")

def check_put_call_parity(call_price, put_price, S, K, T, r, tolerance=1e-10):
    left_side = call_price + K * np.exp(-r * T)
    right_side = put_price + S
    difference = abs(left_side - right_side)
    is_parity_holding = difference < tolerance
    return left_side, right_side, difference, is_parity_holding

left_side, right_side, difference, is_parity_holding = check_put_call_parity(market_price, put_price, S, K, T, r)
print(f"\nPut-Call Parity Check:")
print(f"Left side (C + K * e^(-r * T)): ${left_side:.6f}")
print(f"Right side (P + S): ${right_side:.6f}")
print(f"Difference: ${difference:.10f}")
print(f"Does put-call parity hold? {is_parity_holding}")

put_price_from_parity = market_price - S + K * np.exp(-r * T)
print(f"\nPut Price from Parity: ${put_price_from_parity:.2f}")
print(f"Difference between BSM and Parity: ${abs(put_price - put_price_from_parity):.10f}")

# ------------------------------
# 3Dd.
# ------------------------------
def call_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def put_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def call_theta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta_annual = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return theta_annual

def put_theta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta_annual = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta_annual

trading_days_per_year = 255
holding_days = 20
alpha = 0.05
stock_annual_vol = 0.25
sigma_daily = stock_annual_vol / math.sqrt(trading_days_per_year)
sigma_20d = sigma_daily * math.sqrt(holding_days)

call_market_price = 3.0
sigma_iv = implied_vol
call_0 = black_scholes_call(S, K, T, r, sigma_iv)
put_0 = black_scholes_put(S, K, T, r, sigma_iv)
portfolio_0 = call_0 + put_0 + S

delta_call = call_delta(S, K, T, r, sigma_iv)
delta_put = put_delta(S, K, T, r, sigma_iv)
delta_stock = 1.0
portfolio_delta = delta_call + delta_put + delta_stock

theta_call = call_theta(S, K, T, r, sigma_iv)
theta_put = put_theta(S, K, T, r, sigma_iv)
portfolio_theta = theta_call + theta_put

dt = holding_days / trading_days_per_year
mean_pnl = portfolio_theta * dt
std_pnl = abs(portfolio_delta) * S * sigma_20d

z_alpha = norm.ppf(alpha)
var_delta_normal = -(mean_pnl + z_alpha * std_pnl)
phi_z = norm.pdf(z_alpha)
es_delta_normal = -(mean_pnl - std_pnl * phi_z / alpha)

print("\n=== Delta-Normal Approximation (with Theta) ===")
print(f"Portfolio Delta = {portfolio_delta:.4f}")
print(f"Portfolio Theta = ${portfolio_theta:.4f}")
print(f"VaR (95%) = ${var_delta_normal:.4f}")
print(f"ES (95%) = ${es_delta_normal:.4f}")

# ------------------------------
# 3Dd.
# ------------------------------

n_sims = 100000
np.random.seed(42)
mu = 0 
rands = np.random.normal(mu, sigma_20d, n_sims)
S_20d = S * np.exp(rands)

T_new = T - (holding_days / trading_days_per_year)
call_20d = [black_scholes_call(s_i, K, T_new, r, sigma_iv) for s_i in S_20d]
put_20d = [black_scholes_put(s_i, K, T_new, r, sigma_iv) for s_i in S_20d]

portfolio_20d = np.array(call_20d) + np.array(put_20d) + S_20d

pnl = portfolio_20d - portfolio_0

pnl_sorted = np.sort(pnl)
idx_var = int(alpha * n_sims)
var_mc = -pnl_sorted[idx_var]
es_mc = -pnl_sorted[:idx_var].mean()

print("\n=== (e) Monte Carlo Simulation ===")
print(f"n = {n_sims}")
print(f"Mean of PnL    = {pnl.mean():.4f}")
print(f"VaR (95%)      = {var_mc:.4f}")
print(f"ES (95%)       = {es_mc:.4f}")

# ------------------------------
# 3E.
# ------------------------------
S_range = np.linspace(S * 0.7, S * 1.3, 100)

portfolio_exact = []
for S_i in S_range:
    call_i = black_scholes_call(S_i, K, T, r, sigma_iv)
    put_i = black_scholes_put(S_i, K, T, r, sigma_iv)
    portfolio_i = call_i + put_i + S_i
    portfolio_exact.append(portfolio_i)

portfolio_approx = []
for S_i in S_range:
    portfolio_i = portfolio_0 + portfolio_delta * (S_i - S)
    portfolio_approx.append(portfolio_i)

fig, axs = plt.subplots(figsize=(8, 6))

axs.plot(S_range, portfolio_exact, 'b-', linewidth=2, label='Exact Pricing')
axs.plot(S_range, portfolio_approx, 'r--', linewidth=2, label='Delta-Normal Approximation')
axs.axvline(x=S, color='gray', linestyle='--', alpha=0.7, label='Current Stock Price')
axs.set_xlabel('Stock Price ($)')
axs.set_ylabel('Portfolio Value ($)')
axs.set_title('Portfolio Value vs Stock Price: Method Comparison')
axs.grid(True)
axs.legend()


methods = ['Delta-Normal', 'Monte Carlo']
var_values = [var_delta_normal, var_mc]
es_values = [es_delta_normal, es_mc]

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(methods))
width = 0.35

ax.bar(x - width/2, var_values, width, label='VaR (95%)')
ax.bar(x + width/2, es_values, width, label='ES (95%)')

for i, v in enumerate(var_values):
    ax.text(i - width/2, v + 0.1, f'${v:.2f}', ha='center')
    
for i, v in enumerate(es_values):
    ax.text(i + width/2, v + 0.1, f'${v:.2f}', ha='center')

ax.set_ylabel('Value ($)')
ax.set_title('Comparison of Risk Measures')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
plt.tight_layout()
plt.show()

# Print comparison of methods
print("\n=== Comparison of Methods ===")
print("Delta-Normal Approximation vs Monte Carlo Simulation")
print("\nKey Differences:")
print("1. Linear vs Non-linear: Delta-Normal assumes a linear relationship between stock price")
print("   and portfolio value, while Monte Carlo captures the non-linear nature of options.")
print("2. Distribution assumptions: Delta-Normal assumes the P&L is normally distributed,")
print("   while Monte Carlo builds the full distribution through simulation.")
print("3. Accuracy: Monte Carlo is generally more accurate, especially for portfolios with")
print("   significant convexity (gamma) or longer holding periods.")
print("4. Computational efficiency: Delta-Normal is faster but less accurate for option portfolios.")
print("\nRisk Measure Comparison:")
print(f"VaR (95%): Delta-Normal = ${var_delta_normal:.2f}, Monte Carlo = ${var_mc:.2f}")
print(f"ES (95%): Delta-Normal = ${es_delta_normal:.2f}, Monte Carlo = ${es_mc:.2f}")