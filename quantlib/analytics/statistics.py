"""
Statistical Analysis Module

Provides statistical tools and tests for quantitative analysis including:
- Distribution analysis and fitting
- Hypothesis testing
- Correlation and regression analysis
- Time series analysis
- Monte Carlo simulation utilities
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass


@dataclass
class DistributionFitResult:
	"""Results from distribution fitting"""
	distribution: str
	parameters: Tuple
	aic: float
	bic: float
	ks_statistic: float
	p_value: float
	fitted_data: np.ndarray


class DistributionAnalyzer:
	"""Analyze and fit probability distributions to data"""
	
	DISTRIBUTIONS = {
		'normal': stats.norm,
		'lognormal': stats.lognorm,
		't': stats.t,
		'skewnorm': stats.skewnorm,
		'gamma': stats.gamma,
		'beta': stats.beta,
		'exponential': stats.expon,
		'uniform': stats.uniform
	}
	
	@staticmethod
	def fit_distribution(data: np.ndarray, 
						distribution: str = 'normal') -> DistributionFitResult:
		"""Fit a specific distribution to data"""
		if distribution not in DistributionAnalyzer.DISTRIBUTIONS:
			raise ValueError(f"Unsupported distribution: {distribution}")
		
		dist = DistributionAnalyzer.DISTRIBUTIONS[distribution]
		
		# Fit distribution
		params = dist.fit(data)
		
		# Calculate goodness of fit
		fitted_data = dist.rvs(*params, size=len(data))
		ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
		
		# Calculate information criteria
		log_likelihood = np.sum(dist.logpdf(data, *params))
		k = len(params)
		n = len(data)
		aic = 2 * k - 2 * log_likelihood
		bic = k * np.log(n) - 2 * log_likelihood
		
		return DistributionFitResult(
			distribution=distribution,
			parameters=params,
			aic=aic,
			bic=bic,
			ks_statistic=ks_stat,
			p_value=p_value,
			fitted_data=fitted_data
		)
	
	@staticmethod
	def find_best_distribution(data: np.ndarray, 
							  distributions: Optional[List[str]] = None) -> DistributionFitResult:
		"""Find the best fitting distribution from a list"""
		if distributions is None:
			distributions = list(DistributionAnalyzer.DISTRIBUTIONS.keys())
		
		results = []
		for dist_name in distributions:
			try:
				result = DistributionAnalyzer.fit_distribution(data, dist_name)
				results.append(result)
			except Exception as e:
				warnings.warn(f"Failed to fit {dist_name}: {str(e)}")
				continue
		
		if not results:
			raise ValueError("No distributions could be fitted")
		
		# Prefer BIC (penalizes complexity more). If close, prefer normal for stability
		best = min(results, key=lambda x: x.bic)
		normals = [r for r in results if r.distribution == 'normal']
		if normals:
			normal = normals[0]
			if normal.bic - best.bic <= 2.0:
				best = normal
		return best
	
	@staticmethod
	def distribution_summary(data: np.ndarray) -> Dict:
		"""Comprehensive distribution summary"""
		return {
			'mean': np.mean(data),
			'std': np.std(data),
			'skewness': stats.skew(data),
			'kurtosis': stats.kurtosis(data),
			'jarque_bera': stats.jarque_bera(data),
			'shapiro_wilk': stats.shapiro(data),
			'anderson_darling': stats.anderson(data),
			'percentiles': np.percentile(data, [1, 5, 25, 50, 75, 95, 99])
		}


class HypothesisTests:
	"""Statistical hypothesis testing utilities"""
	
	@staticmethod
	def normality_tests(data: np.ndarray) -> Dict:
		"""Comprehensive normality testing"""
		results = {}
		
		# Shapiro-Wilk test
		stat, p_val = stats.shapiro(data)
		results['shapiro_wilk'] = {'statistic': stat, 'p_value': p_val}
		
		# Jarque-Bera test
		stat, p_val = stats.jarque_bera(data)
		results['jarque_bera'] = {'statistic': stat, 'p_value': p_val}
		
		# Anderson-Darling test
		result = stats.anderson(data, dist='norm')
		significance_levels = getattr(result, 'significance_levels', getattr(result, 'significance_level', None))
		results['anderson_darling'] = {
			'statistic': result.statistic,
			'critical_values': result.critical_values,
			'significance_levels': significance_levels
		}
		
		# Kolmogorov-Smirnov test
		stat, p_val = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
		results['kolmogorov_smirnov'] = {'statistic': stat, 'p_value': p_val}
		
		return results
	
	@staticmethod
	def stationarity_tests(data: np.ndarray) -> Dict:
		"""Test for stationarity in time series"""
		try:
			from statsmodels.tsa.stattools import adfuller, kpss
			
			results = {}
			
			# Augmented Dickey-Fuller test
			adf_result = adfuller(data)
			results['adf'] = {
				'statistic': adf_result[0],
				'p_value': adf_result[1],
				'critical_values': adf_result[4]
			}
			
			# KPSS test
			kpss_result = kpss(data)
			results['kpss'] = {
				'statistic': kpss_result[0],
				'p_value': kpss_result[1],
				'critical_values': kpss_result[3]
			}
			
			return results
			
		except ImportError:
			warnings.warn("statsmodels not available for stationarity tests")
			return {}
	
	@staticmethod
	def independence_tests(x: np.ndarray, y: np.ndarray) -> Dict:
		"""Test independence between two variables"""
		results = {}
		
		# Pearson correlation test
		corr, p_val = stats.pearsonr(x, y)
		results['pearson'] = {'correlation': corr, 'p_value': p_val}
		
		# Spearman correlation test
		corr, p_val = stats.spearmanr(x, y)
		results['spearman'] = {'correlation': corr, 'p_value': p_val}
		
		# Kendall's tau
		corr, p_val = stats.kendalltau(x, y)
		results['kendall'] = {'correlation': corr, 'p_value': p_val}
		
		return results


class RegressionAnalysis:
	"""Regression analysis utilities"""
	
	@staticmethod
	def linear_regression(x: np.ndarray, y: np.ndarray) -> Dict:
		"""Perform linear regression analysis"""
		# Simple linear regression
		slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
		
		# Predictions
		y_pred = slope * x + intercept
		residuals = y - y_pred
		
		# Additional statistics
		n = len(x)
		df = n - 2
		mse = np.sum(residuals**2) / df
		rmse = np.sqrt(mse)
		
		# Confidence intervals for slope
		t_val = stats.t.ppf(0.975, df)
		slope_ci = (slope - t_val * std_err, slope + t_val * std_err)
		
		return {
			'slope': slope,
			'intercept': intercept,
			'r_squared': r_value**2,
			'p_value': p_value,
			'std_error': std_err,
			'slope_ci': slope_ci,
			'residuals': residuals,
			'predictions': y_pred,
			'mse': mse,
			'rmse': rmse,
			'durbin_watson': RegressionAnalysis._durbin_watson(residuals)
		}
	
	@staticmethod
	def _durbin_watson(residuals: np.ndarray) -> float:
		"""Calculate Durbin-Watson statistic"""
		diff = np.diff(residuals)
		return np.sum(diff**2) / np.sum(residuals**2)
	
	@staticmethod
	def rolling_regression(x: np.ndarray, y: np.ndarray, 
						  window: int) -> pd.DataFrame:
		"""Perform rolling window regression"""
		results = []
		
		for i in range(window, len(x) + 1):
			x_window = x[i-window:i]
			y_window = y[i-window:i]
			
			reg_result = RegressionAnalysis.linear_regression(x_window, y_window)
			
			results.append({
				'date_index': i - 1,
				'slope': reg_result['slope'],
				'intercept': reg_result['intercept'],
				'r_squared': reg_result['r_squared'],
				'p_value': reg_result['p_value']
			})
		
		return pd.DataFrame(results)


class TimeSeriesAnalysis:
	"""Time series analysis utilities"""
	
	@staticmethod
	def autocorrelation(data: np.ndarray, max_lags: int = 20) -> np.ndarray:
		"""Calculate autocorrelation function"""
		n = len(data)
		data = data - np.mean(data)
		autocorr = np.correlate(data, data, mode='full')
		autocorr = autocorr[n-1:]
		autocorr = autocorr / autocorr[0]
		return autocorr[:max_lags+1]
	
	@staticmethod
	def ljung_box_test(data: np.ndarray, lags: int = 10) -> Dict:
		"""Ljung-Box test for autocorrelation"""
		try:
			from statsmodels.stats.diagnostic import acorr_ljungbox
			result = acorr_ljungbox(data, lags=lags, return_df=True)
			return {
				'statistics': result['lb_stat'].values,
				'p_values': result['lb_pvalue'].values,
				'lags': list(range(1, lags + 1))
			}
		except ImportError:
			warnings.warn("statsmodels not available for Ljung-Box test")
			return {}
	
	@staticmethod
	def volatility_clustering(returns: np.ndarray, window: int = 22) -> Dict:
		"""Analyze volatility clustering"""
		# Rolling volatility
		rolling_vol = pd.Series(returns).rolling(window).std()
		
		# Volatility of volatility
		vol_of_vol = rolling_vol.rolling(window).std()
		
		# ARCH test
		squared_returns = returns**2
		arch_lm = TimeSeriesAnalysis._arch_lm_test(squared_returns)
		
		return {
			'rolling_volatility': rolling_vol.values,
			'volatility_of_volatility': vol_of_vol.values,
			'arch_lm_test': arch_lm
		}
	
	@staticmethod
	def _arch_lm_test(squared_returns: np.ndarray, lags: int = 12) -> Dict:
		"""Engle's ARCH LM test using regression on lagged squared returns."""
		n = len(squared_returns)
		if n <= lags:
			return {}
		# Regression: y_t = a0 + a1*y_{t-1} + ... + a_p*y_{t-p} + e_t, where y_t is squared returns
		y = squared_returns[lags:]
		X_cols = [np.ones(len(y))]
		for k in range(1, lags + 1):
			X_cols.append(squared_returns[lags - k: n - k])
		X = np.column_stack(X_cols)
		beta, *_ = np.linalg.lstsq(X, y, rcond=None)
		y_hat = X @ beta
		ss_res = float(np.sum((y - y_hat) ** 2))
		ss_tot = float(np.sum((y - np.mean(y)) ** 2))
		r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
		lm_stat = float(len(y) * r_squared)
		p_value = float(1 - stats.chi2.cdf(lm_stat, lags))
		return {
			'lm_statistic': lm_stat,
			'p_value': p_value,
			'r_squared': float(r_squared)
		}


class MonteCarloSimulation:
	"""Monte Carlo simulation utilities"""
	
	@staticmethod
	def geometric_brownian_motion(S0: float, mu: float, sigma: float, 
								  T: float, steps: int, 
								  simulations: int = 1000) -> np.ndarray:
		"""Simulate geometric Brownian motion paths"""
		dt = T / steps
		paths = np.zeros((simulations, steps + 1))
		paths[:, 0] = S0
		
		for i in range(1, steps + 1):
			Z = np.random.standard_normal(simulations)
			paths[:, i] = paths[:, i-1] * np.exp(
				(mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
			)
		
		return paths
	
	@staticmethod
	def var_monte_carlo(returns: np.ndarray, confidence_level: float = 0.05,
					   simulations: int = 10000) -> Dict:
		"""Monte Carlo VaR simulation"""
		# Fit distribution to returns
		mu, sigma = np.mean(returns), np.std(returns)
		
		# Generate random samples
		simulated_returns = np.random.normal(mu, sigma, simulations)
		
		# Calculate VaR
		var = np.percentile(simulated_returns, confidence_level * 100)
		
		# Expected Shortfall (CVaR)
		es = np.mean(simulated_returns[simulated_returns <= var])
		
		return {
			'var': var,
			'expected_shortfall': es,
			'simulated_returns': simulated_returns,
			'confidence_level': confidence_level
		}
	
	@staticmethod
	def bootstrap_confidence_interval(data: np.ndarray, statistic_func,
									confidence_level: float = 0.95,
									n_bootstrap: int = 1000) -> Dict:
		"""Bootstrap confidence intervals for any statistic"""
		bootstrap_stats = []
		n = len(data)
		
		for _ in range(n_bootstrap):
			# Resample with replacement
			bootstrap_sample = np.random.choice(data, size=n, replace=True)
			stat = statistic_func(bootstrap_sample)
			bootstrap_stats.append(stat)
		
		bootstrap_stats = np.array(bootstrap_stats)
		
		# Calculate confidence interval
		alpha = 1 - confidence_level
		lower = np.percentile(bootstrap_stats, (alpha/2) * 100)
		upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
		
		return {
			'statistic': statistic_func(data),
			'bootstrap_distribution': bootstrap_stats,
			'confidence_interval': (lower, upper),
			'confidence_level': confidence_level,
			'std_error': np.std(bootstrap_stats)
		}