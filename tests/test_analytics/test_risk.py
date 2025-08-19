import pytest
import numpy as np
import pandas as pd
from quantlib.analytics.risk import VaRCalculator, RiskMetrics, DrawdownAnalyzer


class TestVaRCalculator:
    """Test VaR calculation methods"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data"""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 1000)  # Daily returns
    
    def test_historical_var(self, sample_returns):
        """Test historical VaR calculation"""
        var_calc = VaRCalculator()
        
        # Test 5% VaR
        var_5 = var_calc.historical_var(sample_returns, confidence_level=0.05)
        assert var_5 < 0  # VaR should be negative
        assert isinstance(var_5, float)
        
        # Test 1% VaR should be more negative than 5% VaR
        var_1 = var_calc.historical_var(sample_returns, confidence_level=0.01)
        assert var_1 < var_5
    
    def test_parametric_var(self, sample_returns):
        """Test parametric VaR calculation"""
        var_calc = VaRCalculator()
        
        var_param = var_calc.parametric_var(sample_returns, confidence_level=0.05)
        assert var_param < 0
        assert isinstance(var_param, float)
        
        # Test with different distribution
        var_t = var_calc.parametric_var(sample_returns, confidence_level=0.05, 
                                       distribution='t')
        assert isinstance(var_t, float)
    
    def test_monte_carlo_var(self, sample_returns):
        """Test Monte Carlo VaR calculation"""
        var_calc = VaRCalculator()
        
        var_mc = var_calc.monte_carlo_var(sample_returns, confidence_level=0.05,
                                         simulations=1000)
        assert var_mc < 0
        assert isinstance(var_mc, float)
    
    def test_expected_shortfall(self, sample_returns):
        """Test Expected Shortfall calculation"""
        var_calc = VaRCalculator()
        
        es = var_calc.expected_shortfall(sample_returns, confidence_level=0.05)
        var = var_calc.historical_var(sample_returns, confidence_level=0.05)
        
        # ES should be more negative than VaR
        assert es < var
        assert isinstance(es, float)


class TestRiskMetrics:
    """Test risk metrics calculations"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample portfolio and benchmark data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        portfolio_returns = np.random.normal(0.0008, 0.015, 252)
        benchmark_returns = np.random.normal(0.0005, 0.012, 252)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        
        return {
            'dates': dates,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'risk_free_rate': risk_free_rate
        }
    
    def test_sharpe_ratio(self, sample_data):
        """Test Sharpe ratio calculation"""
        risk_metrics = RiskMetrics()
        
        sharpe = risk_metrics.sharpe_ratio(
            sample_data['portfolio_returns'],
            sample_data['risk_free_rate']
        )
        
        assert isinstance(sharpe, float)
        # Sharpe ratio should be reasonable for normal market data
        assert -5 < sharpe < 5
    
    def test_sortino_ratio(self, sample_data):
        """Test Sortino ratio calculation"""
        risk_metrics = RiskMetrics()
        
        sortino = risk_metrics.sortino_ratio(
            sample_data['portfolio_returns'],
            sample_data['risk_free_rate']
        )
        
        assert isinstance(sortino, float)
        # Sortino should generally be higher than Sharpe for same data
        sharpe = risk_metrics.sharpe_ratio(
            sample_data['portfolio_returns'],
            sample_data['risk_free_rate']
        )
        assert sortino >= sharpe
    
    def test_beta_calculation(self, sample_data):
        """Test beta calculation"""
        risk_metrics = RiskMetrics()
        
        beta = risk_metrics.beta(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns']
        )
        
        assert isinstance(beta, float)
        # Beta should be reasonable
        assert -2 < beta < 3
    
    def test_information_ratio(self, sample_data):
        """Test information ratio calculation"""
        risk_metrics = RiskMetrics()
        
        info_ratio = risk_metrics.information_ratio(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns']
        )
        
        assert isinstance(info_ratio, float)
    
    def test_max_drawdown(self, sample_data):
        """Test maximum drawdown calculation"""
        risk_metrics = RiskMetrics()
        
        # Create cumulative returns
        cumulative_returns = np.cumprod(1 + sample_data['portfolio_returns'])
        
        max_dd = risk_metrics.max_drawdown(cumulative_returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert max_dd >= -1  # Drawdown shouldn't exceed -100%


class TestDrawdownAnalyzer:
    """Test drawdown analysis"""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price series with known drawdowns"""
        # Create a price series with deliberate drawdowns
        prices = [100]
        for i in range(1, 1000):
            if 200 <= i <= 250:  # Drawdown period
                change = np.random.normal(-0.001, 0.02)
            elif 500 <= i <= 600:  # Another drawdown period
                change = np.random.normal(-0.0005, 0.015)
            else:
                change = np.random.normal(0.0005, 0.01)
            
            prices.append(prices[-1] * (1 + change))
        
        return np.array(prices)
    
    def test_drawdown_series(self, sample_prices):
        """Test drawdown series calculation"""
        analyzer = DrawdownAnalyzer()
        
        drawdowns = analyzer.drawdown_series(sample_prices)
        
        assert len(drawdowns) == len(sample_prices)
        assert all(dd <= 0 for dd in drawdowns)  # All drawdowns should be <= 0
        assert isinstance(drawdowns, np.ndarray)
    
    def test_drawdown_periods(self, sample_prices):
        """Test drawdown period identification"""
        analyzer = DrawdownAnalyzer()
        
        periods = analyzer.drawdown_periods(sample_prices, min_duration=5)
        
        assert isinstance(periods, list)
        
        # Check structure of drawdown periods
        for period in periods:
            assert 'start' in period
            assert 'end' in period
            assert 'duration' in period
            assert 'max_drawdown' in period
            assert period['duration'] >= 5  # Minimum duration filter
            assert period['max_drawdown'] <= 0
    
    def test_drawdown_statistics(self, sample_prices):
        """Test comprehensive drawdown statistics"""
        analyzer = DrawdownAnalyzer()
        
        stats = analyzer.drawdown_statistics(sample_prices)
        
        required_keys = [
            'max_drawdown', 'avg_drawdown', 'drawdown_duration_avg',
            'drawdown_duration_max', 'recovery_time_avg', 'recovery_time_max',
            'num_drawdowns', 'time_underwater_pct'
        ]
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
        
        # Validate ranges
        assert stats['max_drawdown'] <= 0
        assert stats['avg_drawdown'] <= 0
        assert stats['num_drawdowns'] >= 0
        assert 0 <= stats['time_underwater_pct'] <= 100
    
    def test_stress_testing(self, sample_prices):
        """Test stress testing scenarios"""
        analyzer = DrawdownAnalyzer()
        
        # Test market crash scenario
        crash_result = analyzer.stress_test(sample_prices, scenario='market_crash')
        
        assert 'stressed_prices' in crash_result
        assert 'max_drawdown' in crash_result
        assert 'recovery_time' in crash_result
        
        # Stressed max drawdown should be worse than original
        original_dd = analyzer.drawdown_statistics(sample_prices)['max_drawdown']
        assert crash_result['max_drawdown'] <= original_dd
        
        # Test custom scenario
        custom_result = analyzer.stress_test(
            sample_prices, 
            scenario='custom',
            shock_magnitude=-0.3,
            shock_duration=20
        )
        
        assert 'stressed_prices' in custom_result
        assert len(custom_result['stressed_prices']) == len(sample_prices)


class TestIntegration:
    """Integration tests for risk analytics"""
    
    def test_complete_risk_analysis(self):
        """Test complete risk analysis workflow"""
        # Generate sample data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        prices = np.cumprod(1 + returns) * 100
        
        # Initialize analyzers
        var_calc = VaRCalculator()
        risk_metrics = RiskMetrics()
        dd_analyzer = DrawdownAnalyzer()
        
        # Perform comprehensive analysis
        analysis = {
            'var_5': var_calc.historical_var(returns, 0.05),
            'var_1': var_calc.historical_var(returns, 0.01),
            'expected_shortfall': var_calc.expected_shortfall(returns, 0.05),
            'sharpe_ratio': risk_metrics.sharpe_ratio(returns, 0.02/252),
            'sortino_ratio': risk_metrics.sortino_ratio(returns, 0.02/252),
            'beta': risk_metrics.beta(returns, benchmark_returns),
            'max_drawdown': risk_metrics.max_drawdown(prices),
            'drawdown_stats': dd_analyzer.drawdown_statistics(prices)
        }
        
        # Validate all metrics are calculated
        for key, value in analysis.items():
            if key != 'drawdown_stats':
                assert isinstance(value, float)
                assert not np.isnan(value)
            else:
                assert isinstance(value, dict)
        
        # Validate logical relationships
        assert analysis['var_1'] <= analysis['var_5']  # 1% VaR more extreme
        assert analysis['expected_shortfall'] <= analysis['var_5']  # ES more extreme
        assert analysis['max_drawdown'] <= 0  # Drawdown negative