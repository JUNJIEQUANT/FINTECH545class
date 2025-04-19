import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize
import warnings
import time
from scipy.stats import norm
from tqdm import tqdm      
import inspect
portf_df = pd.read_csv("initial_portfolio.csv")
price_df = pd.read_csv("DailyPrices.csv", parse_dates=["Date"])
rfr_df = pd.read_csv("rf.csv", parse_dates=["Date"])

price_df.set_index("Date", inplace=True)
rfr_df.set_index("Date", inplace=True)

ret_df = price_df.pct_change().dropna()
ret_df = ret_df.join(rfr_df, how="left")
ret_df["rf"] = ret_df["rf"].ffill()  
est_data = ret_df[ret_df.index.year <= 2023]
hold_data = ret_df[ret_df.index.year > 2023]

last_date_2023 = est_data.index[-1]
last_price_2023 = price_df.loc[last_date_2023]
mkt_idx = "SPY"
mkt_exc_est = est_data[mkt_idx] - est_data["rf"]
mkt_exc_hold = hold_data[mkt_idx] - hold_data["rf"]

portf_ids = portf_df["Portfolio"].unique()
init_wts = {}        
day_wts_dict = {}    
portf_rets_dict = {}

# Calculate market's cumulative return for the holding period
mkt_cum_ret = (1 + hold_data[mkt_idx]).prod() - 1

capm_params = {}

for sym in portf_df["Symbol"].unique():
    if sym in est_data.columns and sym != mkt_idx:
        stk_exc = est_data[sym] - est_data["rf"]
        
        valid_mask = (~stk_exc.isna()) & (~mkt_exc_est.isna())
        X = mkt_exc_est[valid_mask].values.reshape(-1, 1)
        y = stk_exc[valid_mask].values
        
        if len(y) > 1:  # Need at least 2 points for regression
            reg_model = LinearRegression().fit(X, y)
            alpha_val = float(reg_model.intercept_)
            beta_val = float(reg_model.coef_[0])
        else:
            alpha_val, beta_val = 0.0, 0.0
            
        capm_params[sym] = (alpha_val, beta_val)

# Calculate portfolio weights and returns during holding period
for p_id in portf_ids:
    sub_df = portf_df[portf_df["Portfolio"] == p_id]
    sym_list = sub_df["Symbol"].tolist()
    qty_arr = sub_df["Holding"].values
    
    # Calculate initial weights
    mkt_val_arr = np.zeros(len(sym_list))
    for i, sym in enumerate(sym_list):
        if sym in last_price_2023:
            mkt_val_arr[i] = qty_arr[i] * last_price_2023[sym]
    
    tot_val = mkt_val_arr.sum()
    if tot_val <= 0:
        wts_arr = np.ones(len(sym_list)) / len(sym_list) if len(sym_list) > 0 else np.array([])
    else:
        wts_arr = mkt_val_arr / tot_val
    
    # Store initial weights
    init_wts[p_id] = dict(zip(sym_list, wts_arr))
    
    # Track daily weights and returns during holding period
    curr_wts = wts_arr.copy()
    portf_ret_list = []
    day_wts_list = []
    
    for dt in hold_data.index:
        day_ret_arr = np.zeros(len(sym_list))
        for i, sym in enumerate(sym_list):
            if sym in hold_data.columns:
                day_ret_arr[i] = hold_data.loc[dt, sym]
        
        # Calculate portfolio return
        portf_ret = np.sum(curr_wts * day_ret_arr)
        portf_ret_list.append(portf_ret)
        day_wts_list.append(curr_wts.copy())
        
        # Update weights for next period
        new_vals = curr_wts * (1 + day_ret_arr)
        new_tot = np.sum(new_vals)
        
        if new_tot <= 0:
            curr_wts = np.zeros(len(sym_list))
        else:
            curr_wts = new_vals / new_tot
    
    day_wts_df = pd.DataFrame(day_wts_list, index=hold_data.index, columns=sym_list)
    portf_ret_series = pd.Series(portf_ret_list, index=hold_data.index)
    
    day_wts_dict[str(p_id)] = day_wts_df
    portf_rets_dict[str(p_id)] = portf_ret_series

# Calculate total portfolio weights and returns
tot_init_val = 0
for p_id in portf_ids:
    sub_df = portf_df[portf_df["Portfolio"] == p_id]
    sym_list = sub_df["Symbol"].tolist()
    qty_arr = sub_df["Holding"].values
    
    for i, sym in enumerate(sym_list):
        if sym in last_price_2023:
            tot_init_val += qty_arr[i] * last_price_2023[sym]

# Calculate portfolio weights in total portfolio
agg_wts = {}
for p_id in portf_ids:
    sub_df = portf_df[portf_df["Portfolio"] == p_id]
    sym_list = sub_df["Symbol"].tolist()
    qty_arr = sub_df["Holding"].values
    
    p_val = 0
    for i, sym in enumerate(sym_list):
        if sym in last_price_2023:
            p_val += qty_arr[i] * last_price_2023[sym]
    
    # Portfolio weight in total portfolio
    if tot_init_val > 0:
        agg_wts[str(p_id)] = p_val / tot_init_val
    else:
        agg_wts[str(p_id)] = 0

# Calculate total portfolio return series
agg_rets = pd.Series(0.0, index=hold_data.index)
for p_id in portf_ids:
    if str(p_id) in portf_rets_dict and str(p_id) in agg_wts:
        wt_rets = portf_rets_dict[str(p_id)] * agg_wts[str(p_id)]
        agg_rets += wt_rets

portf_rets_dict["Total"] = agg_rets

# Attribution analysis
attribution_results = {}

for p_id in list(portf_ids) + ["Total"]:
    p_str = str(p_id)
    
    if p_str not in portf_rets_dict:
        continue
        
    portf_rets = portf_rets_dict[p_str]
    
    # Calculate systematic and idiosyncratic components - initialize with float dtype
    sys_comp = pd.Series(0.0, index=portf_rets.index)
    
    if p_id != "Total":
        sub_df = portf_df[portf_df["Portfolio"] == p_id]
        sym_list = sub_df["Symbol"].tolist()
        
        for dt in portf_rets.index:
            # Get market excess return for this date
            if dt in mkt_exc_hold.index:
                r_m = mkt_exc_hold.loc[dt]
                
                # Get weights for this date
                w_arr = day_wts_dict[p_str].loc[dt].values
                
                # Calculate systematic component
                sys_t = 0.0
                for i, sym in enumerate(sym_list):
                    if sym in capm_params:
                        _, beta = capm_params[sym]
                        sys_t += w_arr[i] * beta * r_m
                
                sys_comp.loc[dt] = sys_t
    else:
        # For total portfolio, aggregate from individual portfolios
        for pid in portf_ids:
            pid_str = str(pid)
            if pid_str in portf_rets_dict and pid_str in agg_wts:
                sub_df = portf_df[portf_df["Portfolio"] == pid]
                sym_list = sub_df["Symbol"].tolist()
                
                for dt in portf_rets.index:
                    if dt in mkt_exc_hold.index:
                        r_m = mkt_exc_hold.loc[dt]
                        
                        # Get weights for this date
                        if dt in day_wts_dict[pid_str].index:
                            w_arr = day_wts_dict[pid_str].loc[dt].values
                            
                            # Calculate systematic component for this portfolio
                            sys_t = 0.0
                            for i, sym in enumerate(sym_list):
                                if sym in capm_params:
                                    _, beta = capm_params[sym]
                                    sys_t += w_arr[i] * beta * r_m
                            
                            # Weight by portfolio weight in total portfolio
                            sys_comp.loc[dt] += sys_t * agg_wts[pid_str]
    
    # Calculate idiosyncratic component (total - systematic)
    idio_comp = portf_rets - sys_comp
    
    # Calculate cumulative returns using Carino method
    tot_ret = (1 + portf_rets).prod() - 1
    
    # Carino attribution factor
    k_factor = 1.0  # Default
    if not np.isclose(tot_ret, 0):
        k_factor = np.log(1 + tot_ret) / tot_ret
    
    # Daily k factors
    daily_k = pd.Series(1.0, index=portf_rets.index)
    non_zero = ~np.isclose(portf_rets, 0)
    daily_k[non_zero] = np.log1p(portf_rets[non_zero]) / (portf_rets[non_zero] * k_factor)
    
    # Attribution results
    sys_ret = (sys_comp * daily_k).sum()
    idio_ret = (idio_comp * daily_k).sum()
    
    # Calculate alpha (as per Option 2, where rf is in idiosyncratic bucket)
    mkt_excess_cum_ret = (1 + mkt_exc_hold).prod() - 1
    beta_weighted_mkt = sys_ret  # This is already the beta-weighted market excess return
    alpha_ret = tot_ret - beta_weighted_mkt - hold_data["rf"].sum()
    
    # Risk attribution (volatility)
    tot_vol = portf_rets.std(ddof=0) 
    
    # Calculate covariance contributions
    if len(portf_rets) > 1:
        cov_sys = np.cov(sys_comp, portf_rets, ddof=0)[0,1]
        cov_idio = np.cov(idio_comp, portf_rets, ddof=0)[0,1]
        
        # Risk contributions
        if not np.isclose(tot_vol, 0):
            sys_ri = cov_sys / tot_vol
            idio_ri = cov_idio / tot_vol
        else:
            sys_ri = idio_ri = 0.0
    else:
        sys_ri = idio_ri = 0.0
    
    attribution_results[p_str] = {
        "tot_ret": tot_ret,
        "sys_ret": sys_ret,
        "idio_ret": idio_ret,
        "tot_vol": tot_vol,
        "sys_ri": sys_ri,
        "idio_ri": idio_ri
    }

# Output results
for p_id in list(portf_ids) + ["Total"]:
    p_str = str(p_id)
    
    if p_str not in attribution_results:
        continue
    
    res = attribution_results[p_str]
    print(f"\nPortfolio {p_str} Attribution:")
    print("-" * 70)
    print(f"Total Return: {res['tot_ret']:.6f}")
    print("-" * 70)
    headers = ["Metric", "Systematic", "Idiosyncratic", "Portfolio"]
    print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<10}")
    print("-" * 70)
    print(f"{'Return Attr.':<15} {res['sys_ret']:<12.6f} {res['idio_ret']:<12.6f}  {res['tot_ret']:<10.6f}")
    print(f"{'Vol. Attr.':<15} {res['sys_ri']:<12.6f} {res['idio_ri']:<12.6f}  {res['tot_vol']:<10.6f}")
    print("-" * 70)
    

###############################################################################
# Part 2
###############################################################################
exp_rf = est_data["rf"].mean()
exp_mkt_excess = mkt_exc_est.mean()
mkt_var = mkt_exc_est.var(ddof=0)

capm_zero_alpha = {}
idio_var = {}

for sym in portf_df["Symbol"].unique():
    if sym in est_data.columns and sym != mkt_idx:
        stk_exc = est_data[sym] - est_data["rf"]
        valid_mask = (~stk_exc.isna()) & (~mkt_exc_est.isna())
        
        X = mkt_exc_est[valid_mask].values.reshape(-1, 1)
        y = stk_exc[valid_mask].values
        
        if len(y) > 1:
            reg_model = LinearRegression().fit(X, y)
            beta_val = float(reg_model.coef_[0])
            
            y_pred = reg_model.predict(X)
            residuals = y - y_pred
            idio_variance = np.var(residuals, ddof=0)
        else:
            beta_val = 0.0
            idio_variance = 0.0
            
        capm_zero_alpha[sym] = beta_val
        idio_var[sym] = idio_variance

# Create maximum Sharpe ratio portfolios
opt_weights = {}
max_sharpe_values = {}

for p_id in portf_ids:
    sub_df = portf_df[portf_df["Portfolio"] == p_id]
    sym_list = sub_df["Symbol"].tolist()
    valid_syms = [sym for sym in sym_list if sym in capm_zero_alpha]
    n_assets = len(valid_syms)
    
    if n_assets == 0:
        print(f"No valid symbols for Portfolio {p_id}, skipping optimization")
        continue
    
    # Get betas and idiosyncratic variances
    betas = np.array([capm_zero_alpha[sym] for sym in valid_syms])
    stock_idio_var = np.array([idio_var[sym] for sym in valid_syms])
    
    # Calculate expected returns under CAPM (with zero alpha)
    exp_returns = exp_rf + betas * exp_mkt_excess
    
    # Construct covariance matrix under CAPM
    beta_matrix = np.outer(betas, betas) * mkt_var
    idio_matrix = np.diag(stock_idio_var)
    cov_matrix = beta_matrix + idio_matrix
    
    # Objective function: negative Sharpe ratio
    def neg_sharpe(weights):
        portfolio_return = np.sum(weights * exp_returns)
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_var)
        
        if portfolio_std < 1e-8:
            return 1e10  # Large penalty for near-zero volatility
        
        sharpe = (portfolio_return - exp_rf) / portfolio_std
        return -sharpe
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    # Bounds: allow short selling within limits (-1 to 1)
    bounds = tuple([(-1.0, 1.0) for _ in range(n_assets)])
    
    # Initial guess: equal weights
    init_guess = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, 
                     constraints=constraints, options={'ftol': 1e-9, 'disp': False})
    
    if result.success:
        w_opt = result.x
        # Normalize weights to ensure they sum to 1
        w_opt = w_opt / np.sum(w_opt)
        
        # Calculate achieved Sharpe ratio
        sharpe_ratio = -result.fun
        max_sharpe_values[str(p_id)] = sharpe_ratio
        
        opt_weights[str(p_id)] = dict(zip(valid_syms, w_opt))
    else:
        print(f"Optimization failed for Portfolio {p_id}: {result.message}")
        opt_weights[str(p_id)] = dict(zip(valid_syms, init_guess))
        max_sharpe_values[str(p_id)] = -neg_sharpe(init_guess)



# Store results for optimal portfolios
opt_day_wts_dict = {}
opt_portf_rets_dict = {}
opt_attribution_results = {}

# Simulate each optimal portfolio
for p_id in portf_ids:
    p_str = str(p_id)
    
    if p_str not in opt_weights:
        continue
    
    weights_dict = opt_weights[p_str]
    sym_list = list(weights_dict.keys())
    weight_arr = np.array(list(weights_dict.values()))
    
    # Track daily weights and returns during holding period
    curr_wts = weight_arr.copy()
    portf_ret_list = []
    day_wts_list = []
    
    for dt in hold_data.index:
        day_ret_arr = np.zeros(len(sym_list))
        for i, sym in enumerate(sym_list):
            if sym in hold_data.columns:
                day_ret_arr[i] = hold_data.loc[dt, sym]
        
        # Calculate portfolio return 
        portf_ret = np.sum(curr_wts * day_ret_arr)
        portf_ret_list.append(portf_ret)
        day_wts_list.append(curr_wts.copy())
        
        # Update weights for next period
        new_vals = curr_wts * (1 + day_ret_arr)
        new_tot = np.sum(new_vals)
        
        if new_tot <= 0:
            curr_wts = np.zeros(len(sym_list))
        else:
            curr_wts = new_vals / new_tot
    
    day_wts_df = pd.DataFrame(day_wts_list, index=hold_data.index, columns=sym_list)
    portf_ret_series = pd.Series(portf_ret_list, index=hold_data.index)
    
    opt_day_wts_dict[p_str] = day_wts_df
    opt_portf_rets_dict[p_str] = portf_ret_series

# Calculate total optimal portfolio return series
opt_agg_wts = {p_str: 1/len(portf_ids) for p_str in opt_weights.keys()}  # Equal weight the sub-portfolios
opt_agg_rets = pd.Series(0.0, index=hold_data.index)

for p_id in portf_ids:
    p_str = str(p_id)
    if p_str in opt_portf_rets_dict and p_str in opt_agg_wts:
        wt_rets = opt_portf_rets_dict[p_str] * opt_agg_wts[p_str]
        opt_agg_rets += wt_rets

opt_portf_rets_dict["Total"] = opt_agg_rets

# Attribution analysis for optimal portfolios
for p_id in list(portf_ids) + ["Total"]:
    p_str = str(p_id)
    
    if p_str not in opt_portf_rets_dict:
        continue
        
    portf_rets = opt_portf_rets_dict[p_str]
    
    # Calculate systematic and idiosyncratic components
    sys_comp = pd.Series(0.0, index=portf_rets.index)
    
    if p_id != "Total":
        sym_list = list(opt_weights[p_str].keys())
        
        for dt in portf_rets.index:
            if dt in mkt_exc_hold.index:
                r_m = mkt_exc_hold.loc[dt]
                
                if dt in opt_day_wts_dict[p_str].index:
                    w_arr = opt_day_wts_dict[p_str].loc[dt].values
                    
                    # Calculate systematic component
                    sys_t = 0.0
                    for i, sym in enumerate(sym_list):
                        if sym in capm_zero_alpha:
                            beta = capm_zero_alpha[sym]
                            sys_t += w_arr[i] * beta * r_m
                    
                    sys_comp.loc[dt] = sys_t
    else:
        # For total portfolio, aggregate from individual portfolios
        for pid in portf_ids:
            pid_str = str(pid)
            if pid_str in opt_portf_rets_dict and pid_str in opt_agg_wts:
                if pid_str in opt_weights:
                    sym_list = list(opt_weights[pid_str].keys())
                    
                    for dt in portf_rets.index:
                        if dt in mkt_exc_hold.index:
                            r_m = mkt_exc_hold.loc[dt]
                            
                            if dt in opt_day_wts_dict[pid_str].index:
                                w_arr = opt_day_wts_dict[pid_str].loc[dt].values
                                
                                # Calculate systematic component for this portfolio
                                sys_t = 0.0
                                for i, sym in enumerate(sym_list):
                                    if sym in capm_zero_alpha:
                                        beta = capm_zero_alpha[sym]
                                        sys_t += w_arr[i] * beta * r_m
                                
                                # Weight by portfolio weight in total portfolio
                                sys_comp.loc[dt] += sys_t * opt_agg_wts[pid_str]
    
    # Calculate idiosyncratic component (total - systematic)
    idio_comp = portf_rets - sys_comp
    
    # Calculate cumulative returns using Carino method
    tot_ret = (1 + portf_rets).prod() - 1
    
    # Carino attribution factor
    k_factor = 1.0  # Default
    if not np.isclose(tot_ret, 0):
        k_factor = np.log(1 + tot_ret) / tot_ret
    
    # Daily k factors
    daily_k = pd.Series(1.0, index=portf_rets.index)
    non_zero = ~np.isclose(portf_rets, 0)
    daily_k[non_zero] = np.log1p(portf_rets[non_zero]) / (portf_rets[non_zero] * k_factor)
    
    # Attribution results
    sys_ret = (sys_comp * daily_k).sum()
    idio_ret = (idio_comp * daily_k).sum()
    
    # Risk attribution (volatility)
    tot_vol = portf_rets.std() 
    
    # Calculate covariance contributions
    if len(portf_rets) > 1:
        cov_sys = np.cov(sys_comp, portf_rets, ddof=0)[0,1]
        cov_idio = np.cov(idio_comp, portf_rets, ddof=0)[0,1]
        
        # Risk contributions
        if not np.isclose(tot_vol, 0):
            sys_ri = cov_sys / tot_vol
            idio_ri = cov_idio / tot_vol
        else:
            sys_ri = idio_ri = 0.0
    else:
        sys_ri = idio_ri = 0.0
    
    opt_attribution_results[p_str] = {
        "tot_ret": tot_ret,
        "sys_ret": sys_ret,
        "idio_ret": idio_ret,
        "tot_vol": tot_vol,
        "sys_ri": sys_ri,
        "idio_ri": idio_ri
    }

print("\n--- Optimal Portfolio Attribution Results ---")

for p_id in list(portf_ids) + ["Total"]:
    p_str = str(p_id)
    
    if p_str not in opt_attribution_results:
        continue
    
    res = opt_attribution_results[p_str]
    print(f"\nOptimal Portfolio {p_str} Attribution:")
    print("-" * 50)
    print(f"Total Return: {res['tot_ret']:.6f}")
    print("-" * 50)
    headers = ["Metric", "Systematic", "Idiosyncratic", "Portfolio"]
    print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<10}")
    print("-" * 50)
    print(f"{'Return Attr.':<15} {res['sys_ret']:<12.6f} {res['idio_ret']:<12.6f}  {res['tot_ret']:<10.6f}")
    print(f"{'Vol. Attr.':<15} {res['sys_ri']:<12.6f} {res['idio_ri']:<12.6f}  {res['tot_vol']:<10.6f}")
    print("-" * 50)

# Compare original and optimal portfolios
print("\n--- Comparative Analysis: Original vs. Optimal Portfolios ---")
print("-" * 125)
print(f"{'Portfolio':<10} {'Original Return':<15} {'Optimal Return':<15} {'Return Diff':<15} {'Original Risk':<15} {'Optimal Risk':<15} {'Risk Diff':<15}")
print("-" * 125)

for p_id in portf_ids:
    p_str = str(p_id)
    
    if p_str in attribution_results and p_str in opt_attribution_results:
        orig_ret = attribution_results[p_str]["tot_ret"]
        opt_ret = opt_attribution_results[p_str]["tot_ret"]
        ret_diff = opt_ret - orig_ret
        
        orig_risk = attribution_results[p_str]["tot_vol"]
        opt_risk = opt_attribution_results[p_str]["tot_vol"]
        risk_diff = opt_risk - orig_risk
        
        print(f"{p_str:<10} {orig_ret:<15.6f} {opt_ret:<15.6f} {ret_diff:<15.6f} {orig_risk:<15.6f} {opt_risk:<15.6f}  {risk_diff:<15.6f}")

if "Total" in attribution_results and "Total" in opt_attribution_results:
    orig_ret = attribution_results["Total"]["tot_ret"]
    opt_ret = opt_attribution_results["Total"]["tot_ret"]
    ret_diff = opt_ret - orig_ret
    
    orig_risk = attribution_results["Total"]["tot_vol"]
    opt_risk = opt_attribution_results["Total"]["tot_vol"]
    risk_diff = opt_risk - orig_risk
    
    print(f"{'Total':<10} {orig_ret:<15.6f} {opt_ret:<15.6f} {ret_diff:<15.6f} {orig_risk:<15.6f} {opt_risk:<15.6f}  {risk_diff:<15.6f}")

print("-" * 125)
print("\n--- Comparative Analysis: Return Attribution ---")
print("-" * 75)
print(f"{'Portfolio':<10} {'Orig Sys.':<10} {'Opt Sys.':<10} {'Orig Idio.':<10} {'Opt Idio.':<10}")
print("-" * 75)

for p_id in portf_ids:
    p_str = str(p_id)
    
    if p_str in attribution_results and p_str in opt_attribution_results:
        orig_sys = attribution_results[p_str]["sys_ret"]
        opt_sys = opt_attribution_results[p_str]["sys_ret"]
        
        orig_idio = attribution_results[p_str]["idio_ret"]
        opt_idio = opt_attribution_results[p_str]["idio_ret"]
        
        print(f"{p_str:<10} {orig_sys:<10.6f} {opt_sys:<10.6f} {orig_idio:<10.6f} {opt_idio:<10.6f}")

if "Total" in attribution_results and "Total" in opt_attribution_results:
    orig_sys = attribution_results["Total"]["sys_ret"]
    opt_sys = opt_attribution_results["Total"]["sys_ret"]
    
    orig_idio = attribution_results["Total"]["idio_ret"]
    opt_idio = opt_attribution_results["Total"]["idio_ret"]
    
    print(f"{'Total':<10} {orig_sys:<10.6f} {opt_sys:<10.6f} {orig_idio:<10.6f} {opt_idio:<10.6f}")

print("-" * 75)


###############################################################################
# Part 4
###############################################################################

CONF_LEVEL = 0.95
TAIL_PROB = 1 - CONF_LEVEL
NUM_SIMS = 10000
np.random.seed(42)
warnings.filterwarnings('ignore')

# Load data (using existing variables from previous code)
return_data = price_df.pct_change().dropna()
return_data = return_data.join(rfr_df, how="left")
return_data["rf"] = return_data["rf"].ffill()  

# Split data for pre-holding period
train_data = return_data[return_data.index.year <= 2023]
stock_rets = train_data.drop(columns=["rf"], errors="ignore")

# Get symbols from portfolio
symbol_list = portf_df["Symbol"].unique().tolist()
available_symbols = [s for s in symbol_list if s in stock_rets.columns]

print("\n--- Fitting Distributions to Stock Returns ---")

# Define distributions to test
dist_options = {
    "Normal": stats.norm,
    "Generalized T": stats.t,
    "NIG": stats.norminvgauss,
    "Skew Normal": stats.skewnorm
}

# Function to calculate AICc
def compute_aicc(log_likelihood, sample_size, param_count):
    """Calculate AICc for model comparison"""
    aic = -2 * log_likelihood + 2 * param_count
    correction = (2 * param_count * (param_count + 1)) / (sample_size - param_count - 1)
    return aic + correction

# Store distribution fitting results
model_fits = {}
detail_results = {}

for sym in available_symbols:
    print(f"Analyzing {sym}...")
    
    # Get return data for this symbol
    sym_data = stock_rets[sym].dropna().values
    sample_size = len(sym_data)
    
    min_aicc = np.inf
    best_dist = None
    best_p = None
    fit_details = {}
    
    for dist_name, dist_func in dist_options.items():
        try:
            # Fit distribution with zero mean constraint where applicable
            if dist_name in ["Normal", "Skew Normal"]:
                params = dist_func.fit(sym_data, floc=0)  # Fix location to zero
                param_count = len(params) - 1  # One parameter is fixed
            else:
                params = dist_func.fit(sym_data)
                param_count = len(params)
            
            # Calculate log-likelihood
            log_lik = np.sum(dist_func.logpdf(sym_data, *params))
            
            # Calculate AICc
            aicc_val = compute_aicc(log_lik, sample_size, param_count)
            
            # Store results
            fit_details[dist_name] = {
                "params": params,
                "aicc": aicc_val,
                "log_lik": log_lik,
                "param_count": param_count
            }
            
            # Check if this is the best model so far
            if aicc_val < min_aicc:
                min_aicc = aicc_val
                best_dist = dist_name
                best_p = params
                
        except Exception as e:
            print(f"  Error fitting {dist_name} for {sym}: {e}")
    
    if best_dist:
        # Ensure location parameter is set to 0 for the best model
        final_p = list(best_p)
        
        # Get parameter structure for this distribution
        dist = dist_options[best_dist]
        shape_params = dist.shapes
        loc_scale_list = []
        
        # Identify which parameters are shape, loc, and scale
        if 'loc' in inspect.signature(dist._parse_args).parameters:
            loc_scale_list.append('loc')
        if 'scale' in inspect.signature(dist._parse_args).parameters:
            loc_scale_list.append('scale')
            
        # Map parameter names to indices
        param_map = {}
        current_idx = 0
        
        if shape_params:
            for p in shape_params.split(','):
                param_map[p.strip()] = current_idx
                current_idx += 1
                
        if 'loc' in loc_scale_list:
            param_map['loc'] = current_idx
            current_idx += 1
        if 'scale' in loc_scale_list:
            param_map['scale'] = current_idx
            
        # Force loc=0 for zero mean return assumption
        if 'loc' in param_map:
            final_p[param_map['loc']] = 0.0
        
        # Store best model information
        model_fits[sym] = {
            "dist_name": best_dist,
            "params": tuple(final_p),
            "dist_func": dist
        }
        
        # Print results as required by the assignment
        print(f"  Best fit: {best_dist}")
        print(f"  Parameters: {np.round(final_p, 4)}")
        
        # Store detailed results
        detail_results[sym] = fit_details

# Precompute distribution quantiles for faster simulation
print("\nPrecomputing distribution quantiles...")
ppf_cache = {}
prob_points = np.linspace(0.001, 0.999, 1000)  # Grid of probability points

for sym in model_fits:
    model_info = model_fits[sym]
    dist_func = model_info["dist_func"]
    params = model_info["params"]
    
    # Precompute quantile function values
    quantile_vals = dist_func.ppf(prob_points, *params)
    
    # Store in cache
    ppf_cache[sym] = {
        "probs": prob_points,
        "quantiles": quantile_vals
    }

# VaR and ES calculation
print(f"\n--- Computing 1-Day VaR and ES at {CONF_LEVEL*100}% Confidence Level ---")

# Prepare portfolio information
port_ids = portf_df["Portfolio"].unique()
port_stocks = {}
port_wts = {}
port_vals = {}

# Use prices at the end of pre-holding period for weights
end_prices = price_df.loc[price_df.index <= '2023-12-31'].iloc[-1]

# Calculate portfolio weights and values
for pid in port_ids:
    stocks_subset = portf_df[portf_df["Portfolio"] == pid].copy()
    
    # Filter to stocks with fitted models
    stocks_subset = stocks_subset[stocks_subset["Symbol"].isin(model_fits.keys())]
    syms = stocks_subset["Symbol"].tolist()
    shares = stocks_subset["Holding"].values
    
    market_vals = []
    valid_syms = []
    
    for s, qty in zip(syms, shares):
        if s in end_prices.index and not pd.isna(end_prices[s]):
            market_vals.append(qty * end_prices[s])
            valid_syms.append(s)
    
    market_vals = np.array(market_vals)
    total_val = market_vals.sum()
    
    # Calculate weights
    if np.isclose(total_val, 0):
        num_valid = len(valid_syms)
        weights = np.ones(num_valid) / num_valid if num_valid > 0 else np.array([])
        print(f"Warning: Portfolio {pid} has zero value. Using equal weights.")
    else:
        weights = market_vals / total_val
    
    # Store portfolio information
    port_stocks[str(pid)] = valid_syms
    port_wts[str(pid)] = pd.Series(weights, index=valid_syms)
    port_vals[str(pid)] = total_val

# Create total portfolio
total_value = sum(port_vals.values())
total_wts = {}

for pid in port_ids:
    pid_str = str(pid)
    port_pct = port_vals[pid_str] / total_value
    
    for s, w in port_wts[pid_str].items():
        if s not in total_wts:
            total_wts[s] = 0
        total_wts[s] += port_pct * w

port_stocks["Total"] = list(total_wts.keys())
port_wts["Total"] = pd.Series(total_wts)
port_vals["Total"] = total_value

# Helper function for VaR and ES calculation
def compute_var_es(simulated_returns, alpha):
    """Calculate VaR and ES from simulated returns"""
    if len(simulated_returns) == 0:
        return np.nan, np.nan
    
    sorted_rets = np.sort(simulated_returns)
    var_idx = int(alpha * len(sorted_rets))
    var_value = -sorted_rets[var_idx]
    es_value = -np.mean(sorted_rets[:var_idx+1])
    
    return var_value, es_value

# Results storage
var_es_results = []

# Calculate VaR and ES for each portfolio
for pid, syms in port_stocks.items():
    start_time = time.time()
    print(f"\nPortfolio: {pid}")
    
    if not syms:
        print("  No valid stocks in portfolio. Skipping.")
        continue
    
    # Get portfolio weights and value
    wts = port_wts[pid].reindex(syms).fillna(0).values
    port_value = port_vals[pid]
    
    # Normalize weights if needed
    if not np.isclose(np.sum(wts), 1.0):
        print(f"  Warning: Weights sum to {np.sum(wts):.4f}. Normalizing.")
        wts = wts / np.sum(wts)
    
    # Get historical returns for correlation estimation
    hist_rets = stock_rets[syms].dropna()
    n_assets = len(syms)
    
    if hist_rets.shape[0] < 2 or hist_rets.shape[1] == 0:
        print(f"  Insufficient data for portfolio {pid}. Skipping.")
        continue
    
    # 1. Gaussian Copula with fitted distributions
    try:
        print(f"  Running Gaussian Copula simulation ({NUM_SIMS} iterations)...")
        
        # Calculate Spearman rank correlation
        spearman_corr = hist_rets.corr(method='spearman').fillna(0).values
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(spearman_corr))
        if min_eig < 1e-8:
            print(f"  Adjusting correlation matrix for stability")
            spearman_corr = spearman_corr + np.eye(n_assets) * max(0, 1e-8 - min_eig)
        
        # Cholesky decomposition
        chol = np.linalg.cholesky(spearman_corr)
        
        # Generate correlated standard normal variables
        z_std = np.random.standard_normal((NUM_SIMS, n_assets))
        z_corr = z_std @ chol.T
        
        # Transform to uniform using normal CDF
        u_corr = stats.norm.cdf(z_corr)
        
        # Transform to target distributions using quantile functions
        x_sim = np.zeros_like(u_corr)
        
        for i, s in enumerate(syms):
            if s in ppf_cache:
                # Fast interpolation using cached quantiles
                cache = ppf_cache[s]
                x_sim[:, i] = np.interp(
                    u_corr[:, i],
                    cache["probs"],
                    cache["quantiles"]
                )
            elif s in model_fits:
                # Direct calculation if not cached
                model = model_fits[s]
                dist_func = model["dist_func"]
                params = model["params"]
                x_sim[:, i] = dist_func.ppf(u_corr[:, i], *params)
            else:
                # Fallback to normal distribution
                x_sim[:, i] = stats.norm.ppf(
                    u_corr[:, i], 
                    loc=0, 
                    scale=hist_rets[s].std()
                )
        
        # Handle any infinities or NaNs
        x_sim[~np.isfinite(x_sim)] = 0.0
        
        # Calculate portfolio returns
        port_rets_copula = x_sim @ wts
        
        # Calculate VaR and ES
        var_copula, es_copula = compute_var_es(port_rets_copula, TAIL_PROB)
        
        print(f"  Copula: VaR={var_copula*100:.4f}% (${var_copula*port_value:.2f}), "
              f"ES={es_copula*100:.4f}% (${es_copula*port_value:.2f})")
        
    except Exception as e:
        print(f"  Error in Gaussian Copula calculation: {e}")
        var_copula, es_copula = np.nan, np.nan
    
    # 2. Multivariate Normal
    try:
        print("  Computing multivariate normal risk measures...")
        
        # Mean vector (zero)
        mu = np.zeros(n_assets)
        
        # Covariance matrix
        cov_mat = hist_rets.cov().fillna(0).values
        
        # Ensure positive semi-definiteness
        min_eig = np.min(np.linalg.eigvalsh(cov_mat))
        if min_eig < -1e-8:
            print(f"  Warning: Covariance matrix not positive semi-definite")
            cov_mat = cov_mat + np.eye(n_assets) * abs(min_eig + 1e-8)
        
        # Portfolio variance
        port_var = wts.T @ cov_mat @ wts
        if port_var < 0:
            print(f"  Warning: Negative portfolio variance. Setting to zero.")
            port_var = 0
        
        # Portfolio standard deviation
        port_std = np.sqrt(port_var)
        
        # Calculate VaR and ES analytically for normal distribution
        z_score = stats.norm.ppf(TAIL_PROB)
        var_mvn = -(0 + z_score * port_std)  # Mean is zero
        es_mvn = -(0 - port_std * stats.norm.pdf(z_score) / TAIL_PROB)
        
        print(f"  MVN: VaR={var_mvn*100:.4f}% (${var_mvn*port_value:.2f}), "
              f"ES={es_mvn*100:.4f}% (${es_mvn*port_value:.2f})")
        
    except Exception as e:
        print(f"  Error in Multivariate Normal calculation: {e}")
        var_mvn, es_mvn = np.nan, np.nan
    
    # Record results
    var_es_results.append({
        "Portfolio": pid,
        "Value": port_value,
        "VaR_Copula_pct": var_copula * 100,
        "ES_Copula_pct": es_copula * 100,
        "VaR_MVN_pct": var_mvn * 100,
        "ES_MVN_pct": es_mvn * 100,
        "VaR_Copula_dollar": var_copula * port_value,
        "ES_Copula_dollar": es_copula * port_value,
        "VaR_MVN_dollar": var_mvn * port_value,
        "ES_MVN_dollar": es_mvn * port_value
    })
    
    elapsed = time.time() - start_time
    print(f"  Time elapsed: {elapsed:.2f} seconds")

# Final output reports
print("\n--- Best Fit Distribution Models and Parameters ---")
for sym in available_symbols:
    if sym in model_fits:
        model_info = model_fits[sym]
        print(f"{sym}: {model_info['dist_name']} - Parameters: {np.round(model_info['params'], 4)}")

print("\n--- 1-Day VaR and ES Results (95% Confidence) ---")
results_summary = pd.DataFrame(var_es_results)
print(results_summary[["Portfolio", "VaR_Copula_pct", "ES_Copula_pct", "VaR_MVN_pct", "ES_MVN_pct"]])

print("\n--- Dollar Risk Measures ---")
dollar_columns = ["Portfolio", "Value", "VaR_Copula_dollar", "ES_Copula_dollar", "VaR_MVN_dollar", "ES_MVN_dollar"]
print(results_summary[dollar_columns])

warnings.filterwarnings("default")


###############################################################################
# Part 5
###############################################################################          

CONF = 0.95              
ALPHA = 1 - CONF
NSIMS = 10_000
rng   = np.random.default_rng(42)


def simulate_copula(sym_list, corr, wts):
    """Simulate portfolio return distribution with Gaussian copula + fitted marginals."""
    n = len(sym_list)
    chol = np.linalg.cholesky(corr)
    z    = rng.standard_normal((NSIMS, n)) @ chol.T            # correlated N(0,1)
    u    = norm.cdf(z)                                         # uniform(0,1)
    x    = np.empty_like(u)
    for j, s in enumerate(sym_list):
        cache = ppf_cache[s]
        x[:, j] = np.interp(u[:, j], cache["probs"], cache["quantiles"])
    port_ret = x @ wts
    return port_ret, x


def es_and_contribs(sym_list, wts, corr):
    """Return ES and each asset's ES‑risk contribution (absolute, not pct)."""
    port_ret, asset_ret = simulate_copula(sym_list, corr, wts)

    # identify tail scenarios
    thresh   = np.quantile(port_ret, ALPHA)
    tail_idx = port_ret <= thresh
    tail_R   = asset_ret[tail_idx]               # (k, n)
    tail_p   = port_ret[tail_idx]               # (k,)

    ES  = -tail_p.mean()
    RCs = -wts * tail_R.mean(axis=0)            # raw contributions
    return ES, RCs


def risk_parity_objective(wts, sym_list, corr):
    """ squared error between each RC and the average RC """
    wts = np.maximum(wts, 1e-10)          # keep positivity
    wts = wts / wts.sum()
    ES, RCs = es_and_contribs(sym_list, wts, corr)
    target  = ES / len(RCs)
    return np.sum((RCs - target) ** 2)


def fit_risk_parity(sym_list, corr):
    n = len(sym_list)
    x0 = np.ones(n) / n
    bnds = [(0, 1) for _ in range(n)]       
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    res  = minimize(risk_parity_objective, x0, args=(sym_list, corr),
                    method='SLSQP', bounds=bnds, constraints=cons,
                    options={'ftol': 1e-10, 'disp': False})
    w = res.x / res.x.sum()
    return w

train_rets = price_df.pct_change().loc[:'2023-12-31'].dropna()    

corr_dict = {}
for pid in portf_df["Portfolio"].unique():
    syms = portf_df.loc[portf_df.Portfolio == pid, "Symbol"]
    syms = [s for s in syms if s in model_fits]                 
    corr = train_rets[syms].corr(method='spearman').fillna(0).values
    # force positive‑definite
    mineig = np.min(np.linalg.eigvalsh(corr))
    if mineig < 1e-8:
        corr += np.eye(len(syms)) * (1e-8 - mineig)
    corr_dict[str(pid)] = (syms, corr)

rp_weights = {}
for pid, (syms, corr) in corr_dict.items():
    if len(syms) == 0: continue
    wts = fit_risk_parity(syms, corr)
    rp_weights[pid] = pd.Series(wts, index=syms)

rp_day_wts = {}
rp_port_ret = {}
for pid, w0 in rp_weights.items():
    syms      = w0.index.tolist()
    curr_wts  = w0.values.copy()
    ret_list  = []
    wts_list  = []

    for dt in hold_data.index:
        r = hold_data.loc[dt, syms].values
        ret_p = np.dot(curr_wts, r)
        ret_list.append(ret_p)
        wts_list.append(curr_wts)
        val_next = curr_wts * (1 + r)
        curr_wts = val_next / val_next.sum()

    rp_day_wts[pid]  = pd.DataFrame(wts_list, index=hold_data.index, columns=syms)
    rp_port_ret[pid] = pd.Series(ret_list,  index=hold_data.index)

# equally weight the three RP sub‑portfolios to build a “Total‑RP”
tot_rp_ret = sum(rp_port_ret[pid] / 3 for pid in rp_port_ret)
rp_port_ret["Total"] = tot_rp_ret

def carino_link(r_series):
    tot = (1 + r_series).prod() - 1
    k   = np.log1p(tot) / tot if not np.isclose(tot, 0) else 1.0
    lk  = np.where(np.isclose(r_series, 0), 1,
                   np.log1p(r_series) / (r_series * k))
    return k, pd.Series(lk, index=r_series.index)

attr_rp = {}
for pid, port_ret in rp_port_ret.items():
    sys_comp = pd.Series(0.0, index=port_ret.index)

    if pid != "Total":
        sym_list = rp_day_wts[pid].columns
        for dt in port_ret.index:
            r_m = mkt_exc_hold.loc[dt]
            wts = rp_day_wts[pid].loc[dt].values
            sys_comp[dt] = sum(w * capm_params[s][1] * r_m for w, s in zip(wts, sym_list))
    else:
        for dt in port_ret.index:
            r_m = mkt_exc_hold.loc[dt]
            s  = 0.0
            for sub in ["A", "B", "C"]:
                wts = rp_day_wts[sub].loc[dt].values
                syms = rp_day_wts[sub].columns
                beta_part = sum(w * capm_params[sym][1] * r_m for w, sym in zip(wts, syms))
                s += beta_part / 3
            sys_comp[dt] = s

    idio_comp = port_ret - sys_comp
    k, lk = carino_link(port_ret)
    sys_ret  = (sys_comp * lk).sum()
    idio_ret = (idio_comp * lk).sum()
    tot_ret  = (1 + port_ret).prod() - 1
    sigma    = port_ret.std(ddof=0)
    # risk contribs (volatility)
    cov_sys  = np.cov(sys_comp,  port_ret, ddof=0)[0,1]
    cov_idio = np.cov(idio_comp, port_ret, ddof=0)[0,1]
    sys_rc   = cov_sys / sigma if sigma > 0 else 0
    idio_rc  = cov_idio / sigma if sigma > 0 else 0

    attr_rp[pid] = dict(tot_ret=tot_ret, sys_ret=sys_ret, idio_ret=idio_ret,
                        vol=sigma, sys_rc=sys_rc, idio_rc=idio_rc)
 

print("\nRisk‑Parity (ES) CAPM Attribution")
print("Portfolio   TotRet    SysRet   IdioRet   Vol    SysRC   IdioRC")
for pid, d in attr_rp.items():
    print(f"{pid:<10}  {d['tot_ret']:.4%}  {d['sys_ret']:.4%} "
          f"{d['idio_ret']:.4%}   {d['vol']:.2%}  {d['sys_rc']:.2%}   {d['idio_rc']:.2%}")

