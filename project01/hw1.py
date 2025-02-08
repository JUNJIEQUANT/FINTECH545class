import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import time
#========================================
#          1.a
#========================================
df = pd.read_csv('problem1.csv')
data = df.iloc[:, 0]
def descriptive_statistics(data):
    mean_value = np.mean(data)
    variance_value = np.var(data, ddof=1)  # sample variance (ddof=1)
    skewness_value = skew(data, bias=False)
    excess_kurtosis_value = kurtosis(data, bias=False)  # Fisher's definition -> excess kurtosis
    
    # Print results
    print(f"Mean: {mean_value:.4f}")
    print(f"Variance: {variance_value:.4f}")
    print(f"Skewness: {skewness_value:.4f}")
    print(f"Kurtosis (excess): {excess_kurtosis_value:.4f}")

descriptive_statistics(data)

#========================================
#          1.b
#========================================
def compare_normal_t_aic(data):
    """
    Fits a Normal distribution and a T-distribution to the given data,
    computes their log-likelihoods, and then calculates AIC values.

    Parameters
    ----------
    data : array-like
        The data points to fit.

    Returns
    -------
    aic_norm : float
        AIC for the Normal distribution.
    aic_t : float
        AIC for the T-distribution.
    """
    mu_norm, std_norm = norm.fit(data)
    ll_norm = np.sum(norm.logpdf(data, mu_norm, std_norm))
    aic_norm = 2 * 2 - 2 * ll_norm

    df_t, loc_t, scale_t = t.fit(data)
    ll_t = np.sum(t.logpdf(data, df_t, loc_t, scale_t))
    aic_t = 2 * 3 - 2 * ll_t

    return aic_norm, aic_t

aic_normal, aic_tdist = compare_normal_t_aic(data)
print(f"Normal Distribution: AIC = {aic_normal:.2f}")
print(f"T-Distribution: AIC = {aic_tdist:.2f}")
#========================================
#          1.c
#========================================
mu_norm, std_norm = norm.fit(data)
df_t, loc_t, scale_t = t.fit(data)
plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
x = np.linspace(data.min(), data.max(), 100)
plt.plot(x, norm.pdf(x, mu_norm, std_norm), 'r-', label='Normal Fit')
plt.plot(x, t.pdf(x, df_t, loc_t, scale_t), 'g--', label='T-Distribution Fit')
plt.legend()
plt.show()


#========================================
#          2.a
#========================================
df = pd.read_csv('problem2.csv')

def covmatrix(df):
    return df.cov()
covariance_matrix = covmatrix(df)

print("Pairwise Covariance Matrix:")
print(covariance_matrix)

#========================================
#          2.b
#========================================

eigenvalues = np.linalg.eigvals(covariance_matrix)
print("Eigenvalues:", eigenvalues)

is_psd = np.all(eigenvalues >= 0)
print("Is the matrix at least positive semidefinite?", is_psd)

#========================================
#          2.c
#========================================
cov_df = covariance_matrix  
variances = np.diag(cov_df)
std_devs = np.sqrt(variances)

D_std = np.diag(std_devs)
D_std_inv = np.diag(1.0 / std_devs)

corr_mat = D_std_inv @ cov_df @ D_std_inv

def project_to_psd(matrix):
    eigen_vals, eigen_vecs = np.linalg.eigh(matrix)
    eigen_vals = np.maximum(eigen_vals, 0)
    return eigen_vecs @ np.diag(eigen_vals) @ eigen_vecs.T

def set_unit_diagonal(matrix):
    result = matrix.copy()
    np.fill_diagonal(result, 1.0)
    return result

def matrix_infinity_norm(matrix):
    return np.max(np.sum(np.abs(matrix), axis=1))

def nearest_correlation_matrix(initial_corr, max_iterations=100, tolerance=1e-8):
    Y_prev = initial_corr.copy()
    X_prev = initial_corr.copy()
    adjustment = np.zeros_like(initial_corr)
    
    for _ in range(max_iterations):
        residual = Y_prev - adjustment
        X_curr = project_to_psd(residual)
        adjustment = X_curr - residual
        Y_curr = set_unit_diagonal(X_curr)
        
        # Compute relative changes for convergence checking
        change_X = matrix_infinity_norm(X_curr - X_prev) / (matrix_infinity_norm(X_curr) + 1e-12)
        change_Y = matrix_infinity_norm(Y_curr - Y_prev) / (matrix_infinity_norm(Y_curr) + 1e-12)
        change_diag = matrix_infinity_norm(Y_curr - X_curr) / (matrix_infinity_norm(Y_curr) + 1e-12)
        max_change = max(change_X, change_Y, change_diag)
        
        if max_change < tolerance:
            break
        
        X_prev = X_curr.copy()
        Y_prev = Y_curr.copy()
    
    return X_curr


nearest_corr = nearest_correlation_matrix(corr_mat)

nearest_cov = D_std @ nearest_corr @ D_std

nearest_cov_df = pd.DataFrame(nearest_cov)

print("Nearest PSD Covariance Matrix:")
print(nearest_cov_df)


def rebonato_jackel_near_psd(cov_matrix: np.ndarray, epsilon: float = 0.0) -> np.ndarray:

    diag_vals = np.diag(cov_matrix)
    if np.any(diag_vals <= 0):
        raise ValueError("Covariance matrix has non-positive elements on its diagonal. "
                         "All diagonal entries must be strictly positive for standardization.")

    std_devs = np.sqrt(diag_vals)
    
    D = np.diag(std_devs)
    D_inv = np.diag(1.0 / std_devs)
    
    corr_matrix = D_inv @ cov_matrix @ D_inv

    def _near_psd(A: np.ndarray, eps: float = 0.0) -> np.ndarray:
        n = A.shape[0]
        eigval, eigvec = np.linalg.eig(A)
        
        adjusted_eigval = np.maximum(eigval, eps)
        
        val = np.matrix(adjusted_eigval)
        vec = np.matrix(eigvec)
        
        T = 1.0 / (np.multiply(vec, vec) * val.T)
        T = np.matrix(np.sqrt(np.diag(np.array(T).reshape(n))))
        
        B = T * vec * np.diag(np.array(np.sqrt(val)).reshape(n))
        return B * B.T

    nearest_corr_psd = _near_psd(corr_matrix, epsilon)
    nearest_cov_psd = D @ nearest_corr_psd @ D
    nearest_cov_psd = np.asarray(nearest_cov_psd)
    return nearest_cov_psd

fixed_cov=rebonato_jackel_near_psd(covariance_matrix)
fixed_cov_df = pd.DataFrame(fixed_cov)

#========================================
#          2.d
#========================================

df_clean = df.dropna()

cov_matrix_strict = df_clean.cov()

print(cov_matrix_strict)


#========================================
#          3.a
#========================================
data = pd.read_csv('problem3.csv')

mean_vector = data.mean().values

n = len(data)
covariance_matrix = data.cov().values

#========================================
#          3.b
#========================================
mu1, mu2 = mean_vector[0], mean_vector[1]

cov_mat = data.cov()
sigma11 = cov_mat.iloc[0, 0] 
sigma12 = cov_mat.iloc[0, 1]  
sigma22 = cov_mat.iloc[1, 1] 

x1_value = 0.6

cond_mean_1 = mu2 + (sigma12 / sigma11) * (x1_value - mu1)
cond_var_1  = sigma22 - (sigma12**2 / sigma11)

print("Method 1 - Conditional Mean of X2 | X1=0.6:", cond_mean_1)
print("Method 1 - Conditional Variance of X2 | X1=0.6:", cond_var_1)


beta_1 = sigma12 / sigma11
beta_0 = mu2 - beta_1 * mu1

cond_mean_2 = beta_0 + beta_1 * x1_value
print("Method 2 - Conditional Mean of X2 | X1=0.6:", cond_mean_2)
cond_var_2 = sigma22 - beta_1 * sigma12
print("Method 2 - Conditional Variance of X2 | X1=0.6:", cond_var_2)

#========================================
#          3.c
#========================================
chol_factor = np.linalg.cholesky(cov_mat)

num_draws = 10_000
rng = np.random.default_rng(seed=42)
Z = rng.normal(size=(num_draws, 2))

x1_fixed_component = (0.6 - mu1) / chol_factor[0, 0]
X2_cond_samples = mu2 + chol_factor[1, 0] * x1_fixed_component + chol_factor[1, 1] * Z[:, 1]
sim_mean = X2_cond_samples.mean()
sim_variance = X2_cond_samples.var()

print("Simulated mean of X2 | X1=0.6:", sim_mean)
print("Simulated variance of X2 | X1=0.6:", sim_variance)

#========================================
#          4.a
#========================================

def simulate_ma_process(n, theta, seed=None):
    """
    Simulate an MA(q) process.

    The process is defined as:
        X_t = epsilon_t + theta[0]*epsilon_{t-1} + ... + theta[q-1]*epsilon_{t-q},
    where epsilon_t ~ N(0,1).

    Parameters:
        n (int): Number of observations to generate.
        theta (list or array): MA coefficients (for lags 1, 2, ..., q).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.array: Simulated MA process of length n.
    """
    if seed is not None:
        np.random.seed(seed)
    
    q = len(theta) 
    eps = np.random.normal(loc=0, scale=1, size=n+q)
    
    X = np.zeros(n)

    for t in range(n):
        X[t] = eps[t+q]
        for j in range(1, q+1):
            X[t] += theta[j-1] * eps[t+q - j]
            
    return X

def plot_acf_pacf(series, lags=20, title=""):
    """
    Plot the ACF and PACF for a given time series.

    Parameters:
        series (array-like): The time series data.
        lags (int): Number of lags to show.
        title (str): Title to add to the plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('ACF ' + title)
    
    plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title('PACF ' + title)
    
    plt.tight_layout()
    plt.show()

n = 500
theta_ma1 = [0.5]
ma1 = simulate_ma_process(n, theta_ma1, seed=123)
plot_acf_pacf(ma1, lags=20, title="MA(1) Process")

theta_ma2 = [0.5, 0.2]
ma2 = simulate_ma_process(n, theta_ma2, seed=123)
plot_acf_pacf(ma2, lags=20, title="MA(2) Process")

theta_ma3 = [0.5, 0.2, 0.1]
ma3 = simulate_ma_process(n, theta_ma3, seed=123)
plot_acf_pacf(ma3, lags=20, title="MA(3) Process")


#========================================
#          4.b
#========================================
def simulate_ar_process(n, phi, burn_in=100, seed=42):
    """
    Simulate an AR(p) process of length n.
    
    Parameters:
        n (int)        : Number of samples to return in the final series.
        phi (list/array): Coefficients [phi_1, phi_2, ..., phi_p].
        burn_in (int)  : Number of initial samples to discard to reduce dependence on initial conditions.
        seed (int)     : Random seed for reproducibility.

    Returns:
        np.array: AR(p) process of length n.
    """
    np.random.seed(seed)
    
    p = len(phi)               
    eps = np.random.normal(0, 1, size=n + burn_in + p)  
    
    X_full = np.zeros(n + burn_in + p)
    
    for t in range(p, n + burn_in + p):
        X_full[t] = eps[t] + sum(phi[j] * X_full[t - (j+1)] for j in range(p))
    X = X_full[p + burn_in : p + burn_in + n]
    return X

def plot_acf_pacf(series, lags=20, title=""):
    """
    Plot the ACF and PACF for a given time series.

    Parameters:
        series (array-like): The time series data.
        lags (int)         : Number of lags to show.
        title (str)        : Title to add to the plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('ACF ' + title)
    
    plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title('PACF ' + title)
    
    plt.tight_layout()
    plt.show()

n = 500

phi_ar1 = [0.6]  # phi_1
ar1_data = simulate_ar_process(n, phi_ar1, burn_in=100, seed=123)
plot_acf_pacf(ar1_data, lags=20, title="AR(1)")

phi_ar2 = [0.6, -0.3]  # phi_1, phi_2
ar2_data = simulate_ar_process(n, phi_ar2, burn_in=100, seed=123)
plot_acf_pacf(ar2_data, lags=20, title="AR(2)")

phi_ar3 = [0.4, 0.3, -0.2]  
ar3_data = simulate_ar_process(n, phi_ar3, burn_in=100, seed=123)
plot_acf_pacf(ar3_data, lags=20, title="AR(3)")


#========================================
#          4.c
#========================================
df = pd.read_csv("problem4.csv")
y = df['y']

def plot_acf_pacf(y, lags=20, title_prefix=""):
    """
    Plots the ACF and PACF for a given time series.
    
    Parameters:
        y (array-like): The time series data.
        lags (int)    : Maximum number of lags to display.
        title_prefix (str): A string prefix for the subplot titles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    plot_acf(y, ax=axes[0], lags=lags)
    axes[0].set_title(f"{title_prefix} ACF")
    
    plot_pacf(y, ax=axes[1], lags=lags, method='ywm')
    axes[1].set_title(f"{title_prefix} PACF")
    
    plt.tight_layout()
    plt.show()
plot_acf_pacf(y)

#========================================
#          4.d
#========================================
def get_ar_aic_table(y, max_p=10):
    """
    Computes and returns a list of tuples containing AR orders (p) and their AIC values.
    Orders range from 0 to max_p.
    """
    aic_results = []
    for p in range(0, max_p + 1):
        try:
            model = ARIMA(y, order=(p, 0, 0))
            results = model.fit(method_kwargs={"warn_convergence": False})
            aic = results.aic
        except Exception as e:
            aic = np.nan
            print(f"AR({p}) could not be fit: {e}")
        aic_results.append((p, aic))
    return aic_results

def get_ma_aic_table(y, max_q=10):
    """
    Computes and returns a list of tuples containing MA orders (q) and their AIC values.
    Orders range from 0 to max_q.
    """
    aic_results = []
    for q in range(0, max_q + 1):
        try:
            model = ARIMA(y, order=(0, 0, q))
            results = model.fit(method_kwargs={"warn_convergence": False})
            aic = results.aic
        except Exception as e:
            aic = np.nan
            print(f"MA({q}) could not be fit: {e}")
        aic_results.append((q, aic))
    return aic_results

get_ar_aic_table(y)
get_ma_aic_table(y)

#========================================
#          5.a
#========================================
def exp_weighted_cov(returns, lambda_val=0.97):
    """
    Calculate the exponentially weighted covariance matrix.
    
    Parameters:
        returns (pd.DataFrame): A DataFrame of asset returns. 
                                Each column is a different asset.
        lambda_val (float): Decay factor (commonly 0.97 as per RiskMetrics).
    
    Returns:
        pd.DataFrame: The exponentially weighted covariance matrix.
    """
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

returns = pd.read_csv('DailyReturn.csv', index_col=0)
    
cov_matrix = exp_weighted_cov(returns, lambda_val=0.97)
    
print(cov_matrix)

#========================================
#          5.b
#========================================
def exp_weighted_cov(returns, lambda_val=0.97):
    """
    Compute the exponentially weighted covariance matrix.
    
    Parameters:
        returns (pd.DataFrame): A DataFrame of asset returns, with each column representing an asset.
        lambda_val (float): Decay factor (e.g., 0.97 as recommended by RiskMetrics).
        
    Returns:
        pd.DataFrame: The exponentially weighted covariance matrix.
    """
    n = returns.shape[0]
    
    weights = (1 - lambda_val) * lambda_val ** np.arange(n-1, -1, -1)
    
    weights /= weights.sum()
    
    weighted_mean = returns.mul(weights, axis=0).sum()
    
    centered_returns = returns - weighted_mean
    
    sqrt_weights = np.sqrt(weights)
    weighted_centered = centered_returns.mul(sqrt_weights, axis=0)
    
    cov_matrix = np.dot(weighted_centered.T, weighted_centered)
    
    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

def plot_pca_cumulative_variance(returns, lambda_values):
    """
    For each lambda in lambda_values, compute the exponentially weighted covariance matrix,
    perform PCA (via eigen-decomposition) on it, and plot the cumulative explained variance
    as a function of the number of principal components.
    
    Parameters:
        returns (pd.DataFrame): The returns data (each column is an asset).
        lambda_values (iterable): List or array of lambda values (should be in (0,1)).
    """
    plt.figure(figsize=(10, 6))
    num_assets = returns.shape[1]
    
    for lambda_val in lambda_values:
        cov_matrix = exp_weighted_cov(returns, lambda_val=lambda_val)
        
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        sorted_eigenvalues = eigenvalues[::-1]
        
        total_variance = np.sum(sorted_eigenvalues)
        explained_variance_ratio = sorted_eigenvalues / total_variance
        
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        
        components = np.arange(1, num_assets + 1)
        plt.plot(components, cumulative_explained_variance, 
                 label=f'λ = {lambda_val:.2f}')
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by PCA for Different λ Values')
    plt.legend(title='Decay Factor (λ)')
    plt.show()

lambda_values = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    
plot_pca_cumulative_variance(returns, lambda_values)

#========================================
#          6.a
#========================================
start_cholesky = time.time()
cov_df = pd.read_csv('problem6.csv')
cov_df = cov_df.values

variances = np.diag(cov_df)
std_devs = np.sqrt(variances)
D_std = np.diag(std_devs)
D_std_inv = np.diag(1.0 / std_devs)

corr_mat = D_std_inv @ cov_df @ D_std_inv

nearest_corr = nearest_correlation_matrix(corr_mat)

nearest_cov = D_std @ nearest_corr @ D_std

eigvals, eigvecs = np.linalg.eigh(nearest_cov)
eigvals_fixed = np.where(eigvals < 1e-10, 1e-10, eigvals)
nearest_cov_pd = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T

L = np.linalg.cholesky(nearest_cov_pd)

num_draws = 10000

dim = cov_matrix.shape[0]

Z = np.random.randn(num_draws, dim)

simulated_draws_cholesky = Z @ L.T
df_cholesky = pd.DataFrame(simulated_draws_cholesky)
end_cholesky = time.time()
cholesky_time = end_cholesky - start_cholesky
print(f"Cholesky simulation time: {cholesky_time:.5f} seconds")
#========================================
#          6.b
#========================================
start_pca = time.time()
eigvals, eigvecs = np.linalg.eigh(nearest_cov_pd)

idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]

total_variance = np.sum(eigvals_sorted)
cum_variance = np.cumsum(eigvals_sorted)
num_components = np.searchsorted(cum_variance, 0.75 * total_variance) + 1
print(f"Using {num_components} components to capture at least 75% of the variance.")

eigvals_selected = eigvals_sorted[:num_components]
eigvecs_selected = eigvecs_sorted[:, :num_components]

n_draws = 10000
Z = np.random.randn(n_draws, num_components)

Z_scaled = Z * np.sqrt(eigvals_selected)

simulated_draws_pca = Z_scaled @ eigvecs_selected.T
df_pca = pd.DataFrame(simulated_draws_pca)
end_pca = time.time()
pca_time = end_pca - start_pca
print(f"PCA simulation time:      {pca_time:.5f} seconds")
#========================================
#          6.C
#========================================
sample_cov_chol = np.cov(df_cholesky, rowvar=False)
sample_cov_pca = np.cov(df_pca, rowvar=False)



frobenius_diff_chol = np.linalg.norm(sample_cov_chol - nearest_cov_pd, 'fro')
frobenius_diff_pca  = np.linalg.norm(sample_cov_pca - nearest_cov_pd,  'fro')
frobenius_original = np.linalg.norm(nearest_cov_pd, 'fro')

print("Frobenius norm of the original covariance matrix: {:.4f}".format(frobenius_original))
print("Frobenius norm difference (Cholesky simulation): {:.4f}".format(frobenius_diff_chol))
print("Frobenius norm difference (PCA simulation): {:.4f}".format(frobenius_diff_pca))

#========================================
#          6.D
#========================================
def cumulative_variance_explained(cov_matrix):
    eigenvals, _ = np.linalg.eigh(cov_matrix)
    eigenvals_sorted = np.sort(eigenvals)[::-1]
    total_variance = np.sum(eigenvals_sorted)
    cumulative_explained = np.cumsum(eigenvals_sorted) / total_variance
    return cumulative_explained

cum_var_input = cumulative_variance_explained(nearest_cov_pd)
cum_var_chol  = cumulative_variance_explained(sample_cov_chol)
cum_var_pca   = cumulative_variance_explained(sample_cov_pca)

num_components = len(cum_var_input)
components = np.arange(1, num_components + 1)

plt.figure(figsize=(10, 6))
plt.plot(components, cum_var_input, linestyle='-', label='Input Covariance')
plt.plot(components, cum_var_chol, linestyle='--', label='Cholesky Simulation')
plt.plot(components, cum_var_pca, linestyle='-.', label='PCA Simulation')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Eigenvalues')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#========================================
#          6.E
#========================================
print(f"Cholesky method took {cholesky_time:.5f} seconds.")
print(f"PCA method took      {pca_time:.5f} seconds.")

if cholesky_time < pca_time:
    print("Cholesky method is faster.")
else:
    print("PCA method is faster.")
