"""Unit tests for VaR calculations with deterministic data."""

import numpy as np
import pytest
from quantlib.analytics.risk import VaRCalculator


class TestVaRCalculator:
    """Test VaR calculations with known synthetic data."""
    
    @pytest.fixture
    def synthetic_returns(self):
        """Fixed synthetic return series for deterministic testing."""
        # Create a known distribution: normal with mean=0.001, std=0.02
        np.random.seed(42)  # Fixed seed for reproducibility
        returns = np.random.normal(0.001, 0.02, 1000)
        return returns
    
    @pytest.fixture
    def var_calculator(self):
        """VaR calculator instance."""
        return VaRCalculator()
    
    def test_historical_var_95(self, var_calculator, synthetic_returns):
        """Test historical VaR at 95% confidence level."""
        var_95 = var_calculator.historical_var(synthetic_returns, confidence_level=0.95)
        
        # VaR should be negative (representing a loss)
        assert var_95 < 0
        
        # Should be approximately the 5th percentile
        expected_var = np.percentile(synthetic_returns, 5)
        assert abs(var_95 - expected_var) < 1e-10
    
    def test_historical_var_99(self, var_calculator, synthetic_returns):
        """Test historical VaR at 99% confidence level."""
        var_99 = var_calculator.historical_var(synthetic_returns, confidence_level=0.99)
        var_95 = var_calculator.historical_var(synthetic_returns, confidence_level=0.95)
        
        # 99% VaR should be more negative (worse) than 95% VaR
        assert var_99 < var_95
        
        # Should be approximately the 1st percentile
        expected_var = np.percentile(synthetic_returns, 1)
        assert abs(var_99 - expected_var) < 1e-10
    
    def test_parametric_var_normal(self, var_calculator, synthetic_returns):
        """Test parametric VaR assuming normal distribution."""
        var_95 = var_calculator.parametric_var(
            synthetic_returns, 
            confidence_level=0.95, 
            distribution='normal'
        )
        
        # VaR should be negative
        assert var_95 < 0
        
        # For normal distribution, should be close to mean - 1.645*std
        mean_return = np.mean(synthetic_returns)
        std_return = np.std(synthetic_returns, ddof=1)
        expected_var = mean_return - 1.645 * std_return
        
        # Allow some tolerance due to estimation
        assert abs(var_95 - expected_var) < 0.001
    
    def test_parametric_var_t_distribution(self, var_calculator, synthetic_returns):
        """Test parametric VaR assuming t-distribution."""
        var_95 = var_calculator.parametric_var(
            synthetic_returns, 
            confidence_level=0.95, 
            distribution='t'
        )
        
        # VaR should be negative
        assert var_95 < 0
        
        # t-distribution VaR should be more conservative (more negative) than normal
        var_95_normal = var_calculator.parametric_var(
            synthetic_returns, 
            confidence_level=0.95, 
            distribution='normal'
        )
        # Allow for small numerical differences
        assert var_95 <= var_95_normal + 1e-5
    
    def test_monte_carlo_var(self, var_calculator, synthetic_returns):
        """Test Monte Carlo VaR with fixed random state."""
        var_95 = var_calculator.monte_carlo_var(
            synthetic_returns, 
            confidence_level=0.95,
            simulations=10000,
            random_state=42
        )
        
        # VaR should be negative
        assert var_95 < 0
        
        # Should be reproducible with same random state
        var_95_repeat = var_calculator.monte_carlo_var(
            synthetic_returns, 
            confidence_level=0.95,
            simulations=10000,
            random_state=42
        )
        assert abs(var_95 - var_95_repeat) < 1e-10
    
    def test_var_confidence_levels(self, var_calculator, synthetic_returns):
        """Test that VaR becomes more conservative with higher confidence."""
        var_90 = var_calculator.historical_var(synthetic_returns, confidence_level=0.90)
        var_95 = var_calculator.historical_var(synthetic_returns, confidence_level=0.95)
        var_99 = var_calculator.historical_var(synthetic_returns, confidence_level=0.99)
        
        # Higher confidence should give more negative (worse) VaR
        assert var_99 < var_95 < var_90
    
    def test_empty_returns(self, var_calculator):
        """Test VaR calculation with empty returns array."""
        empty_returns = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            var_calculator.historical_var(empty_returns, confidence_level=0.95)
    
    def test_single_return(self, var_calculator):
        """Test VaR calculation with single return value."""
        single_return = np.array([0.01])
        
        var_95 = var_calculator.historical_var(single_return, confidence_level=0.95)
        assert var_95 == 0.01
    
    def test_invalid_confidence_level(self, var_calculator, synthetic_returns):
        """Test VaR calculation with invalid confidence levels."""
        with pytest.raises(ValueError):
            var_calculator.historical_var(synthetic_returns, confidence_level=1.5)
        
        with pytest.raises(ValueError):
            var_calculator.historical_var(synthetic_returns, confidence_level=-0.1)