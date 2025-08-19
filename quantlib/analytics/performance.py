"""
Performance analysis and attribution tools.

Implements performance metrics, factor decomposition, and attribution analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings

from quantlib.core.utils import Logger, Config
from quantlib.analytics.risk import RiskMetrics


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis toolkit.
    
    Provides detailed performance metrics and analysis capabilities.
    """
    
    def __init__(self, benchmark: str = 'SPY', risk_free_rate: float = 0.02):
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
        self.logger = Logger.get_logger("performance_analyzer")
        
    def calculate_performance_metrics(self, returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary with performance metrics
        """
        # Ensure returns is a pandas Series
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
            
        risk_metrics = RiskMetrics(self.risk_free_rate, self.trading_days)
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (self.trading_days / len(returns)) - 1
        volatility = returns.std() * np.sqrt(self.trading_days)
        
        # Risk-adjusted metrics
        sharpe = risk_metrics.sharpe_ratio(returns)
        sortino = risk_metrics.sortino_ratio(returns)
        calmar = risk_metrics.calmar_ratio(returns)
        
        # Drawdown metrics
        dd_metrics = risk_metrics.calculate_max_drawdown(returns)
        
        # VaR metrics
        var_metrics = risk_metrics.value_at_risk_metrics(returns)
        
        # Win/Loss statistics
        win_rate = (returns > 0).mean()
        loss_rate = (returns < 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        metrics = {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            
            # Risk-adjusted metrics
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Drawdown metrics
            'max_drawdown': dd_metrics['max_drawdown'],
            'recovery_days': dd_metrics['recovery_days'],
            
            # VaR metrics
            'var_95': var_metrics['historical_var_95'],
            'var_99': var_metrics['historical_var_99'],
            'expected_shortfall_95': var_metrics['expected_shortfall_95'],
            
            # Win/Loss metrics
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            
            # Distribution metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
        }
        
        # Add benchmark-relative metrics if benchmark provided
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)
            metrics.update(benchmark_metrics)
            
        return metrics

    # Methods expected by tests
    def total_return(self, values: Union[pd.Series, np.ndarray, List[float]]) -> float:
        series = pd.Series(values)
        return float(series.iloc[-1] / series.iloc[0] - 1.0)

    def annualized_return(self, returns: Union[pd.Series, np.ndarray, List[float]], periods_per_year: int = 252) -> float:
        series = pd.Series(returns)
        return float((1 + series).prod() ** (periods_per_year / len(series)) - 1)

    def volatility(self, returns: Union[pd.Series, np.ndarray, List[float]], annualized: bool = False, periods_per_year: int = 252) -> float:
        series = pd.Series(returns)
        vol = float(series.std())
        return float(vol * np.sqrt(periods_per_year)) if annualized else vol

    def sharpe_ratio(self, returns: Union[pd.Series, np.ndarray, List[float]], risk_free_rate_daily: float = 0.0, periods_per_year: int = 252) -> float:
        series = pd.Series(returns)
        excess = series - risk_free_rate_daily
        std = excess.std()
        if std == 0:
            return 0.0
        return float(excess.mean() / std * np.sqrt(periods_per_year))

    def max_drawdown(self, values: Union[pd.Series, np.ndarray, List[float]]) -> float:
        series = pd.Series(values)
        running_max = series.expanding().max()
        dd = series / running_max - 1.0
        return float(dd.min())

    def calmar_ratio(self, returns: Union[pd.Series, np.ndarray, List[float]], values: Optional[Union[pd.Series, np.ndarray, List[float]]] = None, periods_per_year: int = 252) -> float:
        series = pd.Series(returns)
        annual_return = float((1 + series).prod() ** (periods_per_year / len(series)) - 1)
        max_dd = self.max_drawdown(pd.Series(values) if values is not None else (1 + series).cumprod())
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
        return float(annual_return / abs(max_dd))

    def value_at_risk(self, returns: Union[pd.Series, np.ndarray, List[float]], confidence_level: float = 0.05) -> float:
        series = pd.Series(returns)
        return float(np.percentile(series, confidence_level * 100))

    def win_loss_ratio(self, returns: Union[pd.Series, np.ndarray, List[float]]) -> float:
        series = pd.Series(returns)
        avg_win = series[series > 0].mean() if (series > 0).any() else 0.0
        avg_loss = series[series < 0].mean() if (series < 0).any() else 0.0
        return float(abs(avg_win / avg_loss)) if avg_loss != 0 else np.inf

    def alpha(self, returns: Union[pd.Series, np.ndarray, List[float]], benchmark_returns: Union[pd.Series, np.ndarray, List[float]], risk_free_rate_daily: float = 0.0, periods_per_year: int = 252) -> float:
        r = pd.Series(returns)
        b = pd.Series(benchmark_returns)
        aligned_r, aligned_b = r.align(b, join='inner')
        if len(aligned_r) < 2:
            return 0.0
        beta = self.beta(aligned_r, aligned_b)
        alpha_daily = aligned_r.mean() - risk_free_rate_daily - beta * (aligned_b.mean() - risk_free_rate_daily)
        return float(alpha_daily * periods_per_year)

    def beta(self, returns: Union[pd.Series, np.ndarray, List[float]], benchmark_returns: Union[pd.Series, np.ndarray, List[float]]) -> float:
        r = pd.Series(returns)
        b = pd.Series(benchmark_returns)
        aligned_r, aligned_b = r.align(b, join='inner')
        if len(aligned_r) < 2:
            return 0.0
        cov = np.cov(aligned_r, aligned_b)[0, 1]
        var_b = np.var(aligned_b)
        return float(cov / var_b) if var_b != 0 else 0.0

    def tracking_error(self, returns: Union[pd.Series, np.ndarray, List[float]], benchmark_returns: Union[pd.Series, np.ndarray, List[float]], periods_per_year: int = 252) -> float:
        r = pd.Series(returns)
        b = pd.Series(benchmark_returns)
        aligned_r, aligned_b = r.align(b, join='inner')
        active = aligned_r - aligned_b
        return float(active.std() * np.sqrt(periods_per_year))

    def information_ratio(self, returns: Union[pd.Series, np.ndarray, List[float]], benchmark_returns: Union[pd.Series, np.ndarray, List[float]], periods_per_year: int = 252) -> float:
        r = pd.Series(returns)
        b = pd.Series(benchmark_returns)
        aligned_r, aligned_b = r.align(b, join='inner')
        active = aligned_r - aligned_b
        std = active.std()
        if std == 0:
            return 0.0
        return float(active.mean() / std * np.sqrt(periods_per_year))

    def comprehensive_analysis(self, returns: Union[pd.Series, np.ndarray, List[float]], values: Union[pd.Series, np.ndarray, List[float]], benchmark_returns: Union[pd.Series, np.ndarray, List[float]], risk_free_rate_daily: float = 0.0) -> Dict[str, float]:
        r = pd.Series(returns)
        v = pd.Series(values)
        b = pd.Series(benchmark_returns)
        analysis = {
            'total_return': self.total_return(v),
            'annualized_return': self.annualized_return(r),
            'volatility': self.volatility(r),
            'sharpe_ratio': self.sharpe_ratio(r, risk_free_rate_daily),
            'sortino_ratio': RiskMetrics(self.risk_free_rate, self.trading_days).sortino_ratio(r, risk_free_rate=risk_free_rate_daily * 252),
            'calmar_ratio': self.calmar_ratio(r, v),
            'max_drawdown': self.max_drawdown(v),
            'var_5': self.value_at_risk(r, 0.05),
            'var_1': self.value_at_risk(r, 0.01),
            'win_rate': float((r > 0).mean()),
            'win_loss_ratio': self.win_loss_ratio(r)
        }
        # Add benchmark-relative metrics
        analysis.update({
            'alpha': self.alpha(r, b, risk_free_rate_daily),
            'beta': self.beta(r, b),
            'tracking_error': self.tracking_error(r, b),
            'information_ratio': self.information_ratio(r, b),
        })
        # Capture ratios
        up_capture, down_capture = RiskMetrics(self.risk_free_rate, self.trading_days)._RiskMetrics__calculate_capture if False else (None, None)
        # Compute directly using aligned series
        aligned_r, aligned_b = pd.Series(returns).align(pd.Series(benchmark_returns), join='inner')
        up_mask = aligned_b > 0
        down_mask = aligned_b < 0
        up_capture = float((aligned_r[up_mask].mean() / aligned_b[up_mask].mean()) if up_mask.any() and aligned_b[up_mask].mean() != 0 else 0.0)
        down_capture = float((aligned_r[down_mask].mean() / aligned_b[down_mask].mean()) if down_mask.any() and aligned_b[down_mask].mean() != 0 else 0.0)
        analysis['up_capture'] = up_capture
        analysis['down_capture'] = down_capture
        return analysis
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate benchmark-relative performance metrics.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary with benchmark-relative metrics
        """
        risk_metrics = RiskMetrics(self.risk_free_rate, self.trading_days)
        
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            self.logger.warning("No overlapping data for benchmark comparison")
            return {}
        
        # Relative metrics
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)
        information_ratio = risk_metrics.information_ratio(aligned_returns, aligned_benchmark)
        
        # Beta and alpha
        beta = risk_metrics.calculate_beta(aligned_returns, aligned_benchmark)
        
        # Calculate alpha using CAPM
        rf_daily = self.risk_free_rate / self.trading_days
        excess_portfolio = aligned_returns - rf_daily
        excess_benchmark = aligned_benchmark - rf_daily
        
        if len(excess_portfolio) > 1:
            alpha_daily = excess_portfolio.mean() - beta * excess_benchmark.mean()
            alpha_annualized = alpha_daily * self.trading_days
        else:
            alpha_annualized = 0.0
        
        # Up/Down capture ratios
        up_capture, down_capture = self._calculate_capture_ratios(
            aligned_returns, aligned_benchmark
        )
        
        return {
            'alpha': alpha_annualized,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'active_return': active_returns.mean() * self.trading_days,
        }
    
    def _calculate_capture_ratios(self, returns: pd.Series, 
                                benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate up and down capture ratios.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Tuple of (up_capture, down_capture) ratios
        """
        # Up periods (benchmark positive)
        up_periods = benchmark_returns > 0
        if up_periods.any():
            portfolio_up = returns[up_periods].mean()
            benchmark_up = benchmark_returns[up_periods].mean()
            up_capture = portfolio_up / benchmark_up if benchmark_up != 0 else 0
        else:
            up_capture = 0
        
        # Down periods (benchmark negative)
        down_periods = benchmark_returns < 0
        if down_periods.any():
            portfolio_down = returns[down_periods].mean()
            benchmark_down = benchmark_returns[down_periods].mean()
            down_capture = portfolio_down / benchmark_down if benchmark_down != 0 else 0
        else:
            down_capture = 0
            
        return up_capture, down_capture
    
    def rolling_performance(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Return series
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(self.trading_days)
        
        # Rolling Sharpe ratio
        rolling_metrics['rolling_sharpe'] = returns.rolling(window).apply(
            lambda x: RiskMetrics(self.risk_free_rate, self.trading_days).sharpe_ratio(x)
        )
        
        # Rolling max drawdown
        rolling_metrics['rolling_max_dd'] = returns.rolling(window).apply(
            lambda x: RiskMetrics().calculate_max_drawdown(x)['max_drawdown']
        )
        
        return rolling_metrics.dropna()


class PerformanceAttribution:
    """
    Performance attribution analysis toolkit.
    
    Implements factor-based attribution and sector/style analysis.
    """
    
    def __init__(self, factors: List[str] = None):
        self.factors = factors or ['market', 'size', 'value', 'momentum']
        self.logger = Logger.get_logger("performance_attribution")
        self.trading_days = 252
        
    def factor_attribution(self, returns: pd.Series, 
                          factor_returns: pd.DataFrame) -> Dict[str, Union[float, pd.Series, Dict[str, float]]]:
        """
        Perform factor-based performance attribution.
        
        Args:
            returns: Portfolio return series
            factor_returns: DataFrame with factor return series
            
        Returns:
            Dictionary with attribution results
        """
        # Align data
        aligned_returns = pd.Series(returns)
        aligned_factors = pd.DataFrame(factor_returns)
        aligned_returns, aligned_factors = aligned_returns.align(aligned_factors, join='inner', axis=0)
        
        if len(aligned_returns) < 10:
            self.logger.warning("Insufficient data for factor attribution")
            return {}
        
        # Fit factor model using linear regression
        X = aligned_factors.values
        y = aligned_returns.values
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # Calculate factor loadings (betas)
        factor_loadings = pd.Series(model.coef_, index=aligned_factors.columns)
        alpha = model.intercept_
        
        # Calculate factor contributions
        factor_contributions = {}
        for factor in aligned_factors.columns:
            factor_contribution = factor_loadings[factor] * aligned_factors[factor]
            factor_contributions[factor] = factor_contribution
        
        # Calculate attribution statistics
        total_factor_return = sum(
            factor_loadings[factor] * aligned_factors[factor].mean() 
            for factor in aligned_factors.columns
        ) * self.trading_days
        
        # R-squared
        predicted_returns = model.predict(X)
        r_squared = stats.pearsonr(y, predicted_returns)[0] ** 2
        
        # Factor volatility contributions
        factor_vol_contributions = {}
        portfolio_var = aligned_returns.var()
        
        for factor in aligned_factors.columns:
            factor_var = aligned_factors[factor].var()
            factor_vol_contribution = (factor_loadings[factor] ** 2 * factor_var) / portfolio_var
            factor_vol_contributions[factor] = factor_vol_contribution
        
        return {
            'alpha': float(alpha * self.trading_days),
            'factor_exposures': factor_loadings.to_dict(),
            'factor_contributions': {k: float(v.mean()) for k, v in pd.DataFrame(factor_contributions).items()},
            'total_factor_return': float(total_factor_return),
            'r_squared': float(r_squared),
            'residual_return': float(aligned_returns.mean() - pd.DataFrame(factor_contributions).sum(axis=1).mean())
        }
    
    def sector_attribution(self, portfolio_weights: Union[pd.Series, np.ndarray, List[float]],
                          benchmark_weights: Union[pd.Series, np.ndarray, List[float]],
                          sector_returns: Union[pd.Series, np.ndarray, List[float]],
                          sectors: List[str]) -> Dict[str, Union[float, List[Dict[str, float]]]]:
        """
        Perform sector-based attribution analysis.
        
        Args:
            portfolio_weights: Portfolio sector weights
            sector_returns: Sector return series
            benchmark_weights: Benchmark sector weights
            
        Returns:
            Dictionary with sector attribution results
        """
        pw = pd.Series(portfolio_weights, index=sectors)
        bw = pd.Series(benchmark_weights, index=sectors)
        sr = pd.Series(sector_returns, index=sectors)
        benchmark_return = float((bw * sr).sum())
        allocation = (pw - bw) * benchmark_return
        selection = bw * (sr - benchmark_return)
        interaction = (pw - bw) * (sr - benchmark_return)
        sector_breakdown = []
        for s in sectors:
            sector_breakdown.append({
                'sector': s,
                'allocation': float(allocation[s]),
                'selection': float(selection[s]),
                'total': float(allocation[s] + selection[s] + interaction[s])
            })
        return {
            'allocation_effect': float(allocation.sum()),
            'selection_effect': float(selection.sum()),
            'interaction_effect': float(interaction.sum()),
            'total_effect': float(allocation.sum() + selection.sum() + interaction.sum()),
            'sector_breakdown': sector_breakdown
        }
    
    def style_attribution(self, returns: pd.Series,
                         benchmark_returns: pd.Series,
                         style_factors: pd.DataFrame) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Perform style-based attribution analysis.
        
        Args:
            returns: Portfolio return series
            style_factors: Style factor returns (growth, value, etc.)
            
        Returns:
            Dictionary with style attribution results
        """
        # This is a simplified implementation
        # In practice, you'd use more sophisticated style analysis
        
        attribution_results = self.factor_attribution(returns, style_factors)
        if not attribution_results:
            return {}
        style_exposures = attribution_results['factor_exposures']
        style_contributions: Dict[str, float] = {}
        for style in style_factors.columns:
            if style in style_exposures:
                style_contributions[style] = float(style_exposures[style] * style_factors[style].mean() * self.trading_days)
        active_exposures = {k: float(style_exposures.get(k, 0.0)) for k in style_factors.columns}
        attribution_summary = {
            'total_active_return': float(np.mean(returns) - np.mean(benchmark_returns)),
            'explained_active_return': float(sum(style_contributions.values())),
            'unexplained_active_return': float(np.mean(returns) - np.mean(benchmark_returns) - sum(style_contributions.values()))
        }
        return {
            'style_exposures': style_exposures,
            'style_contributions': style_contributions,
            'active_exposures': active_exposures,
            'attribution_summary': attribution_summary
        }

    def brinson_attribution(self,
                            portfolio_weights: Union[pd.Series, np.ndarray, List[float]],
                            benchmark_weights: Union[pd.Series, np.ndarray, List[float]],
                            sector_returns: Union[pd.Series, np.ndarray, List[float]],
                            benchmark_sector_returns: Union[pd.Series, np.ndarray, List[float]],
                            sectors: List[str]) -> Dict[str, float]:
        pw = pd.Series(portfolio_weights, index=sectors)
        bw = pd.Series(benchmark_weights, index=sectors)
        pr = pd.Series(sector_returns, index=sectors)
        br = pd.Series(benchmark_sector_returns, index=sectors)
        asset_allocation = float(((pw - bw) * br).sum())
        security_selection = float((bw * (pr - br)).sum())
        interaction = float(((pw - bw) * (pr - br)).sum())
        total = float(asset_allocation + security_selection + interaction)
        return {
            'asset_allocation': asset_allocation,
            'security_selection': security_selection,
            'interaction': interaction,
            'total_attribution': total,
            'sector_details': {}
        }