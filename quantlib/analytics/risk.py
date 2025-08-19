"""
Risk metrics and analysis tools.

Implements Value-at-Risk, risk-adjusted returns, drawdown analysis, and stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize
import warnings

from quantlib.core.utils import Logger, Config


class VaRCalculator:
    """
    Value-at-Risk calculator with multiple methodologies.
    
    Supports historical, parametric, and Monte Carlo VaR calculations.
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.logger = Logger.get_logger("var_calculator")
        
    def historical_var(self, returns: pd.Series,
                      confidence_level: Optional[float] = None,
                      confidence_levels: Optional[List[float]] = None) -> Union[float, Dict[float, float]]:
        """
        Calculate Historical Value-at-Risk.
        
        Args:
            returns: Return series
            confidence_levels: List of confidence levels
            
        Returns:
            If confidence_level is provided, returns a single float VaR value.
            Otherwise, returns a dictionary mapping confidence levels to VaR values.
        """
        # Support both single confidence level and list API
        if confidence_level is not None:
            percentile = (confidence_level) * 100
            # Historical VaR is typically the negative percentile of losses; here returns are signed
            # Tests expect a negative number (loss), so take the percentile of the distribution
            return float(np.percentile(returns, percentile))

        confidence_levels = confidence_levels or self.confidence_levels
        
        if len(returns) < 100:
            self.logger.warning(f"Limited data for Historical VaR: {len(returns)} observations")
        
        var_results = {}
        
        for confidence in confidence_levels:
            percentile = (confidence) * 100
            var_value = np.percentile(returns, percentile)
            var_results[confidence] = var_value
            
        self.logger.info(f"Historical VaR calculated for {len(returns)} observations")
        return var_results
    
    def parametric_var(self, returns: pd.Series,
                      confidence_level: Optional[float] = None,
                      confidence_levels: Optional[List[float]] = None,
                      distribution: str = 'normal') -> Union[float, Dict[float, float]]:
        """
        Calculate Parametric Value-at-Risk.
        
        Args:
            returns: Return series
            confidence_levels: List of confidence levels
            distribution: Distribution assumption ('normal', 't', 'skewed_t')
            
        Returns:
            If confidence_level is provided, returns a single float VaR value.
            Otherwise, returns a dictionary mapping confidence levels to VaR values
        """
        # Single-value API
        if confidence_level is not None:
            mean_return = returns.mean()
            std_return = returns.std()
            if distribution == 'normal':
                z_score = stats.norm.ppf(confidence_level)
                return float(mean_return + z_score * std_return)
            elif distribution == 't':
                df, loc, scale = stats.t.fit(returns)
                t_score = stats.t.ppf(confidence_level, df, loc, scale)
                return float(t_score)
            elif distribution == 'skewed_t':
                try:
                    params = stats.skewt.fit(returns)
                    return float(stats.skewt.ppf(confidence_level, *params))
                except Exception:
                    self.logger.warning("Skewed-t fitting failed, using normal distribution")
                    z_score = stats.norm.ppf(confidence_level)
                    return float(mean_return + z_score * std_return)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

        confidence_levels = confidence_levels or self.confidence_levels
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        var_results = {}
        
        for confidence in confidence_levels:
            if distribution == 'normal':
                z_score = stats.norm.ppf(confidence)
                var_value = mean_return + z_score * std_return
                
            elif distribution == 't':
                # Fit t-distribution
                df, loc, scale = stats.t.fit(returns)
                t_score = stats.t.ppf(confidence, df, loc, scale)
                var_value = t_score
                
            elif distribution == 'skewed_t':
                # Fit skewed t-distribution
                try:
                    params = stats.skewt.fit(returns)
                    var_value = stats.skewt.ppf(confidence, *params)
                except:
                    # Fallback to normal distribution
                    self.logger.warning("Skewed-t fitting failed, using normal distribution")
                    z_score = stats.norm.ppf(confidence)
                    var_value = mean_return + z_score * std_return
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
                
            var_results[confidence] = var_value
            
        self.logger.info(f"Parametric VaR calculated using {distribution} distribution")
        return var_results
    
    def monte_carlo_var(self, returns: pd.Series,
                       confidence_level: Optional[float] = None,
                       confidence_levels: Optional[List[float]] = None,
                       simulations: int = 10000,
                       time_horizon: int = 1) -> Union[float, Dict[float, float]]:
        """
        Calculate Monte Carlo Value-at-Risk.
        
        Args:
            returns: Return series
            confidence_levels: List of confidence levels
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in periods
            
        Returns:
            If confidence_level is provided, returns a single float VaR value.
            Otherwise, returns a dictionary mapping confidence levels to VaR values
        """
        confidence_levels = confidence_levels or self.confidence_levels
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducible results
        random_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            simulations
        )
        
        # Single value API
        if confidence_level is not None:
            percentile = (confidence_level) * 100
            return float(np.percentile(random_returns, percentile))

        var_results = {}
        for confidence in confidence_levels:
            percentile = (confidence) * 100
            var_value = np.percentile(random_returns, percentile)
            var_results[confidence] = var_value
            
        self.logger.info(f"Monte Carlo VaR calculated with {simulations} simulations")
        return var_results
    
    def expected_shortfall(self, returns: pd.Series,
                          confidence_level: Optional[float] = None,
                          confidence_levels: Optional[List[float]] = None,
                          method: str = 'historical') -> Union[float, Dict[float, float]]:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Return series
            confidence_levels: List of confidence levels
            method: Calculation method ('historical', 'parametric')
            
        Returns:
            If confidence_level is provided, returns a single float ES value.
            Otherwise, returns a dictionary mapping confidence levels to ES values
        """
        confidence_levels = confidence_levels or self.confidence_levels

        def _es_at(confidence: float) -> float:
            if method == 'historical':
                percentile = (confidence) * 100
                var_threshold = np.percentile(returns, percentile)
                tail_returns = returns[returns <= var_threshold]
                return float(tail_returns.mean() if len(tail_returns) > 0 else var_threshold)
            elif method == 'parametric':
                mean_return = returns.mean()
                std_return = returns.std()
                z_score = stats.norm.ppf(confidence)
                return float(mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence))
            else:
                raise ValueError(f"Unsupported method: {method}")

        if confidence_level is not None:
            return _es_at(confidence_level)

        es_results = {}
        for confidence in confidence_levels:
            es_results[confidence] = _es_at(confidence)
        return es_results


class RiskMetrics:
    """
    Comprehensive risk metrics calculator.
    
    Provides various risk-adjusted return measures and risk statistics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.logger = Logger.get_logger("risk_metrics")
        
    def sharpe_ratio(self, returns: pd.Series, 
                    risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        rf_rate = risk_free_rate or self.risk_free_rate
        
        excess_returns = returns - rf_rate / self.trading_days
        
        if excess_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days)
    
    def sortino_ratio(self, returns: pd.Series,
                     risk_free_rate: Optional[float] = None,
                     target_return: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            target_return: Target return (if None, uses risk-free rate)
            
        Returns:
            Sortino ratio
        """
        rf_rate = risk_free_rate or self.risk_free_rate
        target = target_return or rf_rate / self.trading_days
        
        excess_returns = returns - target
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
            
        downside_deviation = downside_returns.std()
        return excess_returns.mean() / downside_deviation * np.sqrt(self.trading_days)
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Return series
            
        Returns:
            Calmar ratio
        """
        annual_return = (1 + returns).prod() ** (self.trading_days / len(returns)) - 1
        max_dd = self.calculate_max_drawdown(returns)['max_drawdown']
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
            
        return annual_return / abs(max_dd)
    
    def information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information ratio.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Information ratio
        """
        # Ensure pandas Series for alignment
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        if isinstance(benchmark_returns, np.ndarray):
            benchmark_returns = pd.Series(benchmark_returns)
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            self.logger.warning("No overlapping data for Information Ratio calculation")
            return 0.0
            
        active_returns = aligned_returns - aligned_benchmark
        
        if active_returns.std() == 0:
            return 0.0
            
        return active_returns.mean() / active_returns.std() * np.sqrt(self.trading_days)
    
    def treynor_ratio(self, returns: pd.Series, market_returns: pd.Series,
                     risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Treynor ratio.
        
        Args:
            returns: Portfolio return series
            market_returns: Market return series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Treynor ratio
        """
        rf_rate = risk_free_rate or self.risk_free_rate
        
        # Calculate beta
        beta = self.calculate_beta(returns, market_returns)
        
        if beta == 0:
            return 0.0
            
        excess_return = returns.mean() * self.trading_days - rf_rate
        return excess_return / beta
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta coefficient.
        
        Args:
            returns: Portfolio return series
            market_returns: Market return series
            
        Returns:
            Beta coefficient
        """
        # Ensure pandas Series
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        if isinstance(market_returns, np.ndarray):
            market_returns = pd.Series(market_returns)
        # Align series
        aligned_returns, aligned_market = returns.align(market_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return 0.0
            
        covariance = np.cov(aligned_returns, aligned_market)[0, 1]
        market_variance = np.var(aligned_market)
        
        if market_variance == 0:
            return 0.0
            
        return float(covariance / market_variance)

    # Alias expected by tests
    def beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        return self.calculate_beta(returns, market_returns)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Dict[str, Union[float, pd.Timestamp, int]]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Ensure returns is a pandas Series
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_date = running_max.loc[:max_dd_date].idxmax()
        
        # Find recovery date
        recovery_date = None
        recovery_days = None
        
        if max_dd_date < drawdown.index[-1]:
            post_max_dd = drawdown[max_dd_date:]
            recovery_mask = post_max_dd >= -0.001  # Allow for small rounding errors
            if recovery_mask.any():
                recovery_date = post_max_dd[recovery_mask].index[0]
                # Calculate recovery days
                try:
                    recovery_days = (recovery_date - max_dd_date).days
                except Exception:
                    # Fallback for non-datetime indices
                    recovery_days = int(recovery_date - max_dd_date)
        
        # Calculate drawdown duration
        try:
            drawdown_duration = (max_dd_date - peak_date).days
        except Exception:
            drawdown_duration = int(max_dd_date - peak_date)
            
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'peak_date': peak_date,
            'recovery_date': recovery_date,
            'recovery_days': recovery_days,
            'drawdown_duration': drawdown_duration,
            'underwater_curve': drawdown
        }

    # Simple API expected by tests operating on price levels
    def max_drawdown(self, prices: Union[pd.Series, np.ndarray, List[float]]) -> float:
        if isinstance(prices, np.ndarray) or isinstance(prices, list):
            prices = pd.Series(prices)
        running_max = prices.expanding().max()
        drawdown = prices / running_max - 1.0
        return float(drawdown.min())
    
    def value_at_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive VaR metrics.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with VaR metrics
        """
        var_calc = VaRCalculator()
        
        hist_var = var_calc.historical_var(returns)
        param_var = var_calc.parametric_var(returns)
        mc_var = var_calc.monte_carlo_var(returns)
        expected_shortfall = var_calc.expected_shortfall(returns)
        
        return {
            'historical_var_95': hist_var[0.95],
            'historical_var_99': hist_var[0.99],
            'parametric_var_95': param_var[0.95],
            'parametric_var_99': param_var[0.99],
            'monte_carlo_var_95': mc_var[0.95],
            'monte_carlo_var_99': mc_var[0.99],
            'expected_shortfall_95': expected_shortfall[0.95],
            'expected_shortfall_99': expected_shortfall[0.99],
        }


class DrawdownAnalyzer:
    """
    Specialized drawdown analysis tools.
    
    Provides detailed drawdown statistics and underwater curve analysis.
    """
    
    def __init__(self):
        self.logger = Logger.get_logger("drawdown_analyzer")
        
    def analyze_drawdowns(self, returns: pd.Series) -> Dict[str, Union[float, List, pd.DataFrame]]:
        """
        Comprehensive drawdown analysis.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with detailed drawdown analysis
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find all drawdown periods
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        
        # Calculate statistics for each drawdown period
        drawdown_stats = []
        for period in drawdown_periods:
            start_date, end_date, recovery_date = period
            period_dd = drawdown[start_date:end_date]
            
            stats_dict = {
                'start_date': start_date,
                'end_date': end_date,
                'recovery_date': recovery_date,
                'max_drawdown': period_dd.min(),
                'duration_days': (end_date - start_date).days,
                'recovery_days': (recovery_date - end_date).days if recovery_date else None,
                'total_days': (recovery_date - start_date).days if recovery_date else None
            }
            drawdown_stats.append(stats_dict)
        
        drawdown_df = pd.DataFrame(drawdown_stats)
        
        # Overall statistics
        overall_stats = {
            'total_drawdown_periods': len(drawdown_periods),
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown_df['max_drawdown'].mean() if len(drawdown_df) > 0 else 0,
            'avg_duration': drawdown_df['duration_days'].mean() if len(drawdown_df) > 0 else 0,
            'avg_recovery': drawdown_df['recovery_days'].mean() if len(drawdown_df) > 0 else 0,
            'longest_drawdown': drawdown_df['duration_days'].max() if len(drawdown_df) > 0 else 0,
            'longest_recovery': drawdown_df['recovery_days'].max() if len(drawdown_df) > 0 else 0,
            'underwater_curve': drawdown,
            'drawdown_periods': drawdown_df
        }
        
        return overall_stats

    # Methods expected by tests
    def drawdown_series(self, prices: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        if isinstance(prices, np.ndarray) or isinstance(prices, list):
            prices = pd.Series(prices)
        running_max = prices.expanding().max()
        drawdown = prices / running_max - 1.0
        return drawdown.values

    def drawdown_periods(self, prices: Union[pd.Series, np.ndarray, List[float]], min_duration: int = 1) -> List[Dict[str, Union[int, float]]]:
        drawdowns = self.drawdown_series(prices)
        prices_series = pd.Series(prices)
        periods: List[Dict[str, Union[int, float]]] = []
        in_dd = False
        start_idx = None
        for i, dd in enumerate(drawdowns):
            if not in_dd and dd < 0:
                in_dd = True
                start_idx = i
            elif in_dd and dd >= 0:
                end_idx = i
                duration = end_idx - start_idx
                if duration >= min_duration:
                    max_dd = float(np.min(drawdowns[start_idx:end_idx]))
                    periods.append({'start': start_idx, 'end': end_idx - 1, 'duration': duration, 'max_drawdown': max_dd})
                in_dd = False
        if in_dd:
            end_idx = len(drawdowns)
            duration = end_idx - start_idx
            if duration >= min_duration:
                max_dd = float(np.min(drawdowns[start_idx:end_idx]))
                periods.append({'start': start_idx, 'end': end_idx - 1, 'duration': duration, 'max_drawdown': max_dd})
        return periods

    def drawdown_statistics(self, prices: Union[pd.Series, np.ndarray, List[float]]) -> Dict[str, float]:
        series = self.drawdown_series(prices)
        periods = self.drawdown_periods(prices, min_duration=1)
        durations = [p['duration'] for p in periods]
        recoveries: List[int] = []
        # Estimate recovery time as time from trough to end of period
        for p in periods:
            recoveries.append(int(max(0, p['end'] - p['start'])))
        time_underwater_pct = 100.0 * float(np.sum(series < 0)) / len(series) if len(series) > 0 else 0.0
        return {
            'max_drawdown': float(np.min(series)) if len(series) > 0 else 0.0,
            'avg_drawdown': float(np.mean(series[series < 0])) if np.any(series < 0) else 0.0,
            'drawdown_duration_avg': float(np.mean(durations)) if durations else 0.0,
            'drawdown_duration_max': float(np.max(durations)) if durations else 0.0,
            'recovery_time_avg': float(np.mean(recoveries)) if recoveries else 0.0,
            'recovery_time_max': float(np.max(recoveries)) if recoveries else 0.0,
            'num_drawdowns': float(len(periods)),
            'time_underwater_pct': float(time_underwater_pct)
        }

    def stress_test(self, prices: Union[pd.Series, np.ndarray, List[float]],
                    scenario: str = 'market_crash',
                    shock_magnitude: float = -0.3,
                    shock_duration: int = 20) -> Dict[str, Union[np.ndarray, float, int]]:
        prices_arr = np.asarray(prices, dtype=float)
        stressed = prices_arr.copy()
        if scenario == 'market_crash':
            # Apply a one-off shock followed by recovery
            crash_point = int(len(stressed) * 0.6)
            stressed[crash_point:] *= (1.0 + shock_magnitude)
        elif scenario == 'custom':
            start = max(0, int(len(stressed) * 0.5))
            end = min(len(stressed), start + shock_duration)
            stressed[start:end] *= (1.0 + shock_magnitude)
        else:
            self.logger.warning(f"Unknown scenario {scenario}, no stress applied")
        max_dd = RiskMetrics().max_drawdown(stressed)
        # Rough recovery time: count to first non-negative drawdown after trough
        dd_series = self.drawdown_series(stressed)
        trough_idx = int(np.argmin(dd_series))
        recovery_time = 0
        for i in range(trough_idx, len(dd_series)):
            if dd_series[i] >= 0:
                recovery_time = i - trough_idx
                break
        return {
            'stressed_prices': stressed,
            'max_drawdown': float(max_dd),
            'recovery_time': int(recovery_time)
        }
    
    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[Tuple]:
        """
        Identify individual drawdown periods.
        
        Args:
            drawdown: Drawdown series
            
        Returns:
            List of tuples (start_date, trough_date, recovery_date)
        """
        periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd_value in drawdown.items():
            if not in_drawdown and dd_value < -0.001:  # Start of drawdown
                in_drawdown = True
                start_date = date
                
            elif in_drawdown and dd_value >= -0.001:  # End of drawdown
                in_drawdown = False
                
                # Find the trough (maximum drawdown) in this period
                period_dd = drawdown[start_date:date]
                trough_date = period_dd.idxmin()
                
                periods.append((start_date, trough_date, date))
        
        # Handle case where series ends in drawdown
        if in_drawdown:
            period_dd = drawdown[start_date:]
            trough_date = period_dd.idxmin()
            periods.append((start_date, trough_date, None))
        
        return periods
    
    def stress_test_scenarios(self, returns: pd.Series, 
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing under various scenarios.
        
        Args:
            returns: Return series
            scenarios: Dictionary of scenario parameters
            
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        for scenario_name, params in scenarios.items():
            # Apply scenario shock
            shocked_returns = self._apply_scenario_shock(returns, params)
            
            # Calculate metrics under stress
            risk_metrics = RiskMetrics()
            
            scenario_results = {
                'total_return': (1 + shocked_returns).prod() - 1,
                'volatility': shocked_returns.std() * np.sqrt(252),
                'sharpe_ratio': risk_metrics.sharpe_ratio(shocked_returns),
                'max_drawdown': risk_metrics.calculate_max_drawdown(shocked_returns)['max_drawdown'],
                'var_95': np.percentile(shocked_returns, 5),
                'var_99': np.percentile(shocked_returns, 1)
            }
            
            results[scenario_name] = scenario_results
            
        return results
    
    def _apply_scenario_shock(self, returns: pd.Series, 
                            params: Dict[str, float]) -> pd.Series:
        """
        Apply scenario shock to returns.
        
        Args:
            returns: Original return series
            params: Scenario parameters (mean_shock, vol_shock, etc.)
            
        Returns:
            Shocked return series
        """
        shocked_returns = returns.copy()
        
        # Apply mean shock
        if 'mean_shock' in params:
            shocked_returns += params['mean_shock']
            
        # Apply volatility shock
        if 'vol_shock' in params:
            mean_return = returns.mean()
            shocked_returns = mean_return + (shocked_returns - mean_return) * params['vol_shock']
            
        # Apply correlation shock (simplified)
        if 'correlation_shock' in params:
            # This would require market data for proper implementation
            pass
            
        return shocked_returns