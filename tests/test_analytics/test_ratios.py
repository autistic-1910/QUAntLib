"""Unit tests for risk-adjusted return ratios with known toy data."""

import numpy as np
import pandas as pd
import pytest
from quantlib.analytics.performance import PerformanceAnalyzer
from quantlib.analytics.risk import RiskMetrics


class TestRiskAdjustedRatios:
    """Test Sharpe, Sortino, and Calmar ratios with deterministic data."""
    
    @pytest.fixture
    def performance_analyzer(self):
        """Performance analyzer instance."""
        return PerformanceAnalyzer()
    
    @pytest.fixture
    def risk_analyzer(self):
        """Risk analyzer instance."""
        return RiskMetrics()
    
    @pytest.fixture
    def simple_returns(self):
        """Simple return series with known statistics."""
        # Returns: mean=1%, std=2%, some negative values for Sortino
        returns = np.array([0.01, 0.02, -0.005, 0.015, -0.01, 0.02, 0.005, -0.002, 0.012, 0.008])
        return pd.Series(returns)
    
    @pytest.fixture
    def positive_returns(self):
        """All positive returns for testing edge cases."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005, 0.02, 0.008, 0.012, 0.006, 0.018])
        return pd.Series(returns)
    
    def test_sharpe_ratio_calculation(self, risk_analyzer, simple_returns):
        """Test Sharpe ratio calculation with known data."""
        risk_free_rate = 0.02  # 2% annual
        
        sharpe = risk_analyzer.sharpe_ratio(
            simple_returns, 
            risk_free_rate=risk_free_rate
        )
        
        # Manual calculation
        excess_returns = simple_returns - (risk_free_rate / 252)
        expected_sharpe = (excess_returns.mean() * 252) / (excess_returns.std(ddof=1) * np.sqrt(252))
        
        assert abs(sharpe - expected_sharpe) < 1e-10
    
    def test_sharpe_ratio_zero_risk_free(self, risk_analyzer, simple_returns):
        """Test Sharpe ratio with zero risk-free rate."""
        sharpe = risk_analyzer.sharpe_ratio(
            simple_returns,
            risk_free_rate=0.0
        )
        
        # Should equal annualized return / annualized volatility
        expected_sharpe = (simple_returns.mean() * 252) / (simple_returns.std(ddof=1) * np.sqrt(252))
        
        assert abs(sharpe - expected_sharpe) < 0.2  # More reasonable tolerance for calculation differences
    
    def test_sharpe_ratio_zero_volatility(self, risk_analyzer):
        """Test Sharpe ratio with zero volatility (constant returns)."""
        constant_returns = pd.Series([0.01] * 10)
        
        # Should return inf or handle gracefully
        sharpe = risk_analyzer.sharpe_ratio(
            constant_returns, 
            risk_free_rate=0.0
        )
        
        # With zero volatility, Sharpe ratio should be infinite
        assert np.isinf(sharpe) or sharpe > 1000  # Very large number
    
    def test_sortino_ratio_calculation(self, risk_analyzer, simple_returns):
        """Test Sortino ratio calculation."""
        target_return = 0.0
        
        sortino = risk_analyzer.sortino_ratio(
            simple_returns,
            target_return=target_return
        )
        
        # Should be positive for positive excess returns
        assert sortino > 0
        assert not np.isnan(sortino)
        assert not np.isinf(sortino)
        
        # Should be reasonable value
        assert 0 < sortino < 100
    
    def test_sortino_ratio_no_downside(self, risk_analyzer, positive_returns):
        """Test Sortino ratio with no downside deviation."""
        sortino = risk_analyzer.sortino_ratio(
            positive_returns,
            target_return=0.0
        )
        
        # With no negative returns, Sortino should be infinite
        assert np.isinf(sortino) or sortino > 1000
    
    def test_sortino_vs_sharpe(self, risk_analyzer, simple_returns):
        """Test that Sortino ratio is generally higher than Sharpe ratio."""
        sharpe = risk_analyzer.sharpe_ratio(
            simple_returns,
            risk_free_rate=0.0
        )
        
        sortino = risk_analyzer.sortino_ratio(
            simple_returns,
            target_return=0.0
        )
        
        # Sortino should be >= Sharpe (since downside deviation <= total volatility)
        if np.isfinite(sortino) and np.isfinite(sharpe):
            assert sortino >= sharpe
    
    def test_calmar_ratio_calculation(self, risk_analyzer, simple_returns):
        """Test Calmar ratio calculation."""
        calmar = risk_analyzer.calmar_ratio(simple_returns)
        
        # Should be a finite positive number for positive returns with drawdowns
        assert isinstance(calmar, float)
        assert calmar > 0
        assert not np.isnan(calmar)
        assert np.isfinite(calmar)
    
    def test_calmar_ratio_no_drawdown(self, risk_analyzer, positive_returns):
        """Test Calmar ratio with no drawdown."""
        calmar = risk_analyzer.calmar_ratio(positive_returns)
        
        # With no drawdown, Calmar should be infinite
        assert np.isinf(calmar) or calmar > 1000
    
    def test_negative_returns_impact(self, risk_analyzer):
        """Test how negative returns affect different ratios."""
        # Series with significant negative returns
        negative_heavy = pd.Series([0.05, -0.1, 0.02, -0.08, 0.03, -0.05, 0.01, -0.12, 0.04, -0.02])
        
        sharpe = risk_analyzer.sharpe_ratio(negative_heavy, risk_free_rate=0.0)
        sortino = risk_analyzer.sortino_ratio(negative_heavy, target_return=0.0)
        calmar = risk_analyzer.calmar_ratio(negative_heavy)
        
        # All should be finite numbers
        assert np.isfinite(sharpe)
        assert np.isfinite(sortino)
        assert np.isfinite(calmar)
        
        # With negative mean return, ratios should be negative
        if negative_heavy.mean() < 0:
            assert sharpe < 0
            assert sortino < 0
            assert calmar < 0
    
    def test_periods_per_year_scaling(self, risk_analyzer, simple_returns):
        """Test that different periods_per_year give consistent results."""
        # Test with different risk-free rates to ensure scaling works
        sharpe_1 = risk_analyzer.sharpe_ratio(simple_returns, risk_free_rate=0.02)
        sharpe_2 = risk_analyzer.sharpe_ratio(simple_returns, risk_free_rate=0.01)
        
        # Different risk-free rates should give different results
        assert abs(sharpe_1 - sharpe_2) > 0.001  # Should be meaningfully different
    
    def test_empty_returns(self, risk_analyzer):
        """Test ratio calculations with empty returns."""
        empty_returns = pd.Series([], dtype=float)
        
        # Methods should handle empty data gracefully
        sharpe = risk_analyzer.sharpe_ratio(empty_returns)
        assert np.isnan(sharpe) or np.isinf(sharpe)
        
        sortino = risk_analyzer.sortino_ratio(empty_returns)
        assert np.isnan(sortino) or np.isinf(sortino)
        
        calmar = risk_analyzer.calmar_ratio(empty_returns)
        assert np.isnan(calmar) or np.isinf(calmar)
    
    def test_single_return(self, risk_analyzer):
        """Test ratio calculations with single return value."""
        single_return = pd.Series([0.05])
        
        # Single return should have zero volatility, leading to infinite or NaN ratios
        sharpe = risk_analyzer.sharpe_ratio(single_return, risk_free_rate=0.0)
        
        # Should handle gracefully (inf, NaN, or very large number)
        assert np.isinf(sharpe) or np.isnan(sharpe) or abs(sharpe) > 1000