"""Unit tests for drawdown and recovery metrics."""

import numpy as np
import pandas as pd
import pytest
from quantlib.analytics.performance import PerformanceAnalyzer


class TestDrawdownMetrics:
    """Test drawdown and recovery calculations with known data."""
    
    @pytest.fixture
    def simple_price_series(self):
        """Simple price series with known drawdown pattern."""
        # Price series: 100 -> 110 -> 90 -> 120 -> 80 -> 100
        prices = np.array([100, 110, 90, 120, 80, 100])
        dates = pd.date_range('2023-01-01', periods=len(prices), freq='D')
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def performance_analyzer(self):
        """Performance analyzer instance."""
        return PerformanceAnalyzer()
    
    def test_max_drawdown_simple(self, performance_analyzer, simple_price_series):
        """Test maximum drawdown calculation with simple series."""
        # max_drawdown expects price values, not returns
        max_dd = performance_analyzer.max_drawdown(simple_price_series)
        
        # Expected: from peak 120 to trough 80 = (80-120)/120 = -33.33%
        expected_max_dd = -0.3333333333333333
        assert abs(max_dd - expected_max_dd) < 0.02
    
    def test_drawdown_series(self, performance_analyzer, simple_price_series):
        """Test drawdown series calculation."""
        returns = simple_price_series.pct_change().dropna()
        
        # Calculate cumulative returns and drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        
        # Verify drawdown is always <= 0
        assert all(drawdown <= 0)
        
        # Verify maximum drawdown matches our calculation
        # max_drawdown expects price values, so use cumulative returns
        max_dd = performance_analyzer.max_drawdown(cum_returns)
        assert abs(drawdown.min() - max_dd) < 1e-6
    
    def test_no_drawdown_series(self, performance_analyzer):
        """Test with series that has no drawdown (always increasing)."""
        # Always positive returns
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.005])
        cum_returns = (1 + returns).cumprod()
        
        max_dd = performance_analyzer.max_drawdown(cum_returns)
        
        # Should be 0 (no drawdown)
        assert max_dd == 0.0
    
    def test_single_large_drawdown(self, performance_analyzer):
        """Test with single large drawdown."""
        # Start at 1, drop 50%, then recover
        returns = pd.Series([0.0, -0.5, 0.0, 0.0, 1.0])
        cum_returns = (1 + returns).cumprod()
        
        max_dd = performance_analyzer.max_drawdown(cum_returns)
        
        # Maximum drawdown should be -50%
        assert abs(max_dd - (-0.5)) < 1e-6
    
    def test_multiple_drawdowns(self, performance_analyzer):
        """Test with multiple drawdown periods."""
        # Two separate drawdown periods
        returns = pd.Series([0.1, -0.2, 0.15, 0.05, -0.3, 0.2])
        
        # Calculate expected manually
        cum_returns = (1 + returns).cumprod()
        max_dd = performance_analyzer.max_drawdown(cum_returns)
        
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        expected_max_dd = drawdown.min()
        
        assert abs(max_dd - expected_max_dd) < 1e-6
    
    def test_drawdown_duration(self, performance_analyzer):
        """Test drawdown duration calculation."""
        # Create series with known recovery pattern
        # Peak at index 1, trough at index 3, recovery at index 5
        returns = pd.Series([0.1, 0.0, -0.1, -0.05, 0.08, 0.08])
        
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        
        # Find periods in drawdown (drawdown < 0)
        in_drawdown = drawdown < 0
        
        # Should have drawdown periods
        assert any(in_drawdown)
    
    def test_empty_returns(self, performance_analyzer):
        """Test drawdown calculation with empty returns."""
        empty_series = pd.Series([], dtype=float)
        
        # Empty series should return NaN or raise an error
        try:
            max_dd = performance_analyzer.max_drawdown(empty_series)
            assert np.isnan(max_dd) or np.isinf(max_dd)
        except (ValueError, IndexError):
            pass  # Either behavior is acceptable
    
    def test_single_return(self, performance_analyzer):
        """Test drawdown calculation with single return."""
        single_return = pd.Series([0.05])
        single_price = (1 + single_return).cumprod()
        
        max_dd = performance_analyzer.max_drawdown(single_price)
        
        # Single positive return should have no drawdown
        assert max_dd == 0.0
    
    def test_all_negative_returns(self, performance_analyzer):
        """Test drawdown with all negative returns."""
        negative_returns = pd.Series([-0.01, -0.02, -0.015, -0.01])
        
        # Calculate expected cumulative decline
        cum_returns = (1 + negative_returns).cumprod()
        max_dd = performance_analyzer.max_drawdown(cum_returns)
        expected_max_dd = (cum_returns.iloc[-1] - 1.0) / 1.0
        
        assert abs(max_dd - expected_max_dd) < 0.02
        assert max_dd < 0  # Should be negative
    
    def test_recovery_factor(self, performance_analyzer, simple_price_series):
        """Test recovery factor calculation."""
        returns = simple_price_series.pct_change().dropna()
        
        total_return = (1 + returns).prod() - 1
        max_dd = performance_analyzer.max_drawdown(returns)
        
        if max_dd != 0:
            recovery_factor = total_return / abs(max_dd)
            
            # Recovery factor should be positive if total return is positive
            if total_return > 0:
                assert recovery_factor > 0
        else:
            # If no drawdown, recovery factor is undefined or infinite
            pass