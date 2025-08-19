import pytest
import numpy as np
import pandas as pd
from quantlib.analytics.performance import PerformanceAnalyzer, PerformanceAttribution


class TestPerformanceAnalyzer:
    """Test performance analysis methods"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample performance data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Generate correlated returns
        portfolio_returns = np.random.normal(0.0008, 0.015, 252)
        benchmark_returns = 0.7 * portfolio_returns + 0.3 * np.random.normal(0.0005, 0.012, 252)
        
        portfolio_values = np.cumprod(1 + portfolio_returns) * 100000
        benchmark_values = np.cumprod(1 + benchmark_returns) * 100000
        
        return {
            'dates': dates,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'risk_free_rate': 0.02 / 252
        }
    
    def test_total_return(self, sample_data):
        """Test total return calculation"""
        analyzer = PerformanceAnalyzer()
        
        total_ret = analyzer.total_return(sample_data['portfolio_values'])
        
        assert isinstance(total_ret, float)
        # Should match manual calculation
        expected = (sample_data['portfolio_values'][-1] / sample_data['portfolio_values'][0]) - 1
        assert abs(total_ret - expected) < 1e-10
    
    def test_annualized_return(self, sample_data):
        """Test annualized return calculation"""
        analyzer = PerformanceAnalyzer()
        
        ann_ret = analyzer.annualized_return(sample_data['portfolio_returns'])
        
        assert isinstance(ann_ret, float)
        # Should be reasonable for daily data
        assert -1 < ann_ret < 2  # Between -100% and 200%
    
    def test_volatility(self, sample_data):
        """Test volatility calculation"""
        analyzer = PerformanceAnalyzer()
        
        vol = analyzer.volatility(sample_data['portfolio_returns'])
        
        assert isinstance(vol, float)
        assert vol > 0  # Volatility should be positive
        
        # Test annualized volatility
        ann_vol = analyzer.volatility(sample_data['portfolio_returns'], annualized=True)
        assert ann_vol > vol  # Annualized should be higher for daily data
    
    def test_sharpe_ratio(self, sample_data):
        """Test Sharpe ratio calculation"""
        analyzer = PerformanceAnalyzer()
        
        sharpe = analyzer.sharpe_ratio(
            sample_data['portfolio_returns'],
            sample_data['risk_free_rate']
        )
        
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
    
    def test_max_drawdown(self, sample_data):
        """Test maximum drawdown calculation"""
        analyzer = PerformanceAnalyzer()
        
        max_dd = analyzer.max_drawdown(sample_data['portfolio_values'])
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Should be negative or zero
        assert max_dd >= -1  # Shouldn't exceed -100%
    
    def test_calmar_ratio(self, sample_data):
        """Test Calmar ratio calculation"""
        analyzer = PerformanceAnalyzer()
        
        calmar = analyzer.calmar_ratio(
            sample_data['portfolio_returns'],
            sample_data['portfolio_values']
        )
        
        assert isinstance(calmar, float)
        # Should be finite (not inf or -inf)
        assert np.isfinite(calmar)
    
    def test_var_calculation(self, sample_data):
        """Test VaR calculation"""
        analyzer = PerformanceAnalyzer()
        
        var_5 = analyzer.value_at_risk(sample_data['portfolio_returns'], 0.05)
        var_1 = analyzer.value_at_risk(sample_data['portfolio_returns'], 0.01)
        
        assert isinstance(var_5, float)
        assert isinstance(var_1, float)
        assert var_5 > var_1  # 5% VaR should be less extreme than 1% VaR
        assert var_5 < 0  # VaR should be negative
    
    def test_win_loss_ratio(self, sample_data):
        """Test win/loss ratio calculation"""
        analyzer = PerformanceAnalyzer()
        
        win_loss = analyzer.win_loss_ratio(sample_data['portfolio_returns'])
        
        assert isinstance(win_loss, float)
        assert win_loss >= 0  # Should be non-negative
    
    def test_alpha_beta(self, sample_data):
        """Test alpha and beta calculation"""
        analyzer = PerformanceAnalyzer()
        
        alpha = analyzer.alpha(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns'],
            sample_data['risk_free_rate']
        )
        
        beta = analyzer.beta(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns']
        )
        
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert -2 < beta < 3  # Reasonable beta range
    
    def test_tracking_error(self, sample_data):
        """Test tracking error calculation"""
        analyzer = PerformanceAnalyzer()
        
        te = analyzer.tracking_error(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns']
        )
        
        assert isinstance(te, float)
        assert te >= 0  # Should be non-negative
    
    def test_information_ratio(self, sample_data):
        """Test information ratio calculation"""
        analyzer = PerformanceAnalyzer()
        
        ir = analyzer.information_ratio(
            sample_data['portfolio_returns'],
            sample_data['benchmark_returns']
        )
        
        assert isinstance(ir, float)
        assert np.isfinite(ir)
    
    def test_comprehensive_analysis(self, sample_data):
        """Test comprehensive performance analysis"""
        analyzer = PerformanceAnalyzer()
        
        analysis = analyzer.comprehensive_analysis(
            sample_data['portfolio_returns'],
            sample_data['portfolio_values'],
            sample_data['benchmark_returns'],
            sample_data['risk_free_rate']
        )
        
        # Check all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'var_5', 'var_1',
            'win_rate', 'win_loss_ratio', 'alpha', 'beta', 'tracking_error',
            'information_ratio', 'up_capture', 'down_capture'
        ]
        
        for metric in expected_metrics:
            assert metric in analysis
            assert isinstance(analysis[metric], (int, float))
            assert np.isfinite(analysis[metric])


class TestPerformanceAttribution:
    """Test performance attribution methods"""
    
    @pytest.fixture
    def sample_attribution_data(self):
        """Generate sample data for attribution analysis"""
        np.random.seed(42)
        
        # Create factor returns
        factors = pd.DataFrame({
            'Market': np.random.normal(0.0005, 0.012, 252),
            'Value': np.random.normal(0.0002, 0.008, 252),
            'Growth': np.random.normal(0.0003, 0.010, 252),
            'Size': np.random.normal(0.0001, 0.006, 252)
        })
        
        # Create portfolio returns as combination of factors
        portfolio_returns = (
            0.6 * factors['Market'] +
            0.2 * factors['Value'] +
            0.1 * factors['Growth'] +
            0.1 * factors['Size'] +
            np.random.normal(0, 0.005, 252)  # Idiosyncratic risk
        )
        
        # Create benchmark returns
        benchmark_returns = (
            0.8 * factors['Market'] +
            0.1 * factors['Value'] +
            0.1 * factors['Growth'] +
            np.random.normal(0, 0.003, 252)
        )
        
        # Sector data
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']
        sector_weights_portfolio = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        sector_weights_benchmark = np.array([0.25, 0.2, 0.25, 0.15, 0.15])
        
        sector_returns = pd.DataFrame({
            sector: np.random.normal(0.0005, 0.015, 252) for sector in sectors
        })
        
        return {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'factor_returns': factors,
            'sector_returns': sector_returns,
            'sector_weights_portfolio': sector_weights_portfolio,
            'sector_weights_benchmark': sector_weights_benchmark,
            'sectors': sectors
        }
    
    def test_factor_attribution(self, sample_attribution_data):
        """Test factor-based attribution"""
        attribution = PerformanceAttribution()
        
        result = attribution.factor_attribution(
            sample_attribution_data['portfolio_returns'],
            sample_attribution_data['factor_returns']
        )
        
        assert 'factor_exposures' in result
        assert 'factor_contributions' in result
        assert 'r_squared' in result
        assert 'residual_return' in result
        
        # Check factor exposures
        exposures = result['factor_exposures']
        assert len(exposures) == len(sample_attribution_data['factor_returns'].columns)
        
        # Check contributions sum approximately to total return
        contributions = result['factor_contributions']
        total_contribution = sum(contributions.values())
        total_return = np.mean(sample_attribution_data['portfolio_returns'])
        
        # Should be close (within residual)
        assert abs(total_contribution - total_return) < 0.01
    
    def test_sector_attribution(self, sample_attribution_data):
        """Test sector-based attribution"""
        attribution = PerformanceAttribution()
        
        result = attribution.sector_attribution(
            sample_attribution_data['sector_weights_portfolio'],
            sample_attribution_data['sector_weights_benchmark'],
            sample_attribution_data['sector_returns'].mean().values,
            sample_attribution_data['sectors']
        )
        
        assert 'allocation_effect' in result
        assert 'selection_effect' in result
        assert 'interaction_effect' in result
        assert 'total_effect' in result
        assert 'sector_breakdown' in result
        
        # Check sector breakdown
        breakdown = result['sector_breakdown']
        assert len(breakdown) == len(sample_attribution_data['sectors'])
        
        for sector_data in breakdown:
            assert 'sector' in sector_data
            assert 'allocation' in sector_data
            assert 'selection' in sector_data
            assert 'total' in sector_data
    
    def test_style_attribution(self, sample_attribution_data):
        """Test style-based attribution"""
        attribution = PerformanceAttribution()
        
        # Create style factor data
        style_factors = sample_attribution_data['factor_returns'][['Value', 'Growth', 'Size']]
        
        result = attribution.style_attribution(
            sample_attribution_data['portfolio_returns'],
            sample_attribution_data['benchmark_returns'],
            style_factors
        )
        
        assert 'style_exposures' in result
        assert 'style_contributions' in result
        assert 'active_exposures' in result
        assert 'attribution_summary' in result
        
        # Check active exposures
        active_exp = result['active_exposures']
        assert len(active_exp) == len(style_factors.columns)
        
        # Check attribution summary
        summary = result['attribution_summary']
        assert 'total_active_return' in summary
        assert 'explained_active_return' in summary
        assert 'unexplained_active_return' in summary
    
    def test_brinson_attribution(self, sample_attribution_data):
        """Test Brinson attribution model"""
        attribution = PerformanceAttribution()
        
        result = attribution.brinson_attribution(
            sample_attribution_data['sector_weights_portfolio'],
            sample_attribution_data['sector_weights_benchmark'],
            sample_attribution_data['sector_returns'].mean().values,
            sample_attribution_data['sector_returns'].mean().values * 0.9,  # Benchmark sector returns
            sample_attribution_data['sectors']
        )
        
        assert 'asset_allocation' in result
        assert 'security_selection' in result
        assert 'interaction' in result
        assert 'total_attribution' in result
        assert 'sector_details' in result
        
        # Validate that components sum to total
        total_calc = (result['asset_allocation'] + 
                     result['security_selection'] + 
                     result['interaction'])
        
        assert abs(total_calc - result['total_attribution']) < 1e-10


class TestIntegration:
    """Integration tests for performance analytics"""
    
    def test_complete_performance_analysis(self):
        """Test complete performance analysis workflow"""
        # Generate comprehensive test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Portfolio and benchmark returns
        portfolio_returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        portfolio_values = np.cumprod(1 + portfolio_returns) * 100000
        
        # Factor data
        factor_returns = pd.DataFrame({
            'Market': benchmark_returns,
            'Value': np.random.normal(0.0002, 0.008, 252),
            'Growth': np.random.normal(0.0003, 0.010, 252)
        })
        
        # Initialize analyzers
        perf_analyzer = PerformanceAnalyzer()
        attribution = PerformanceAttribution()
        
        # Comprehensive analysis
        analysis = {
            'performance_metrics': perf_analyzer.comprehensive_analysis(
                portfolio_returns, portfolio_values, benchmark_returns, 0.02/252
            ),
            'factor_attribution': attribution.factor_attribution(
                portfolio_returns, factor_returns
            )
        }
        
        # Validate structure
        assert 'performance_metrics' in analysis
        assert 'factor_attribution' in analysis
        
        # Validate performance metrics
        perf_metrics = analysis['performance_metrics']
        required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'alpha', 'beta']
        for metric in required_metrics:
            assert metric in perf_metrics
            assert isinstance(perf_metrics[metric], (int, float))
            assert np.isfinite(perf_metrics[metric])
        
        # Validate factor attribution
        factor_attr = analysis['factor_attribution']
        assert 'factor_exposures' in factor_attr
        assert 'r_squared' in factor_attr
        assert 0 <= factor_attr['r_squared'] <= 1
    
    def test_performance_comparison(self):
        """Test performance comparison between multiple strategies"""
        np.random.seed(42)
        
        # Generate multiple strategy returns
        strategies = {
            'Strategy_A': np.random.normal(0.001, 0.015, 252),
            'Strategy_B': np.random.normal(0.0008, 0.018, 252),
            'Strategy_C': np.random.normal(0.0012, 0.020, 252)
        }
        
        benchmark_returns = np.random.normal(0.0005, 0.012, 252)
        
        analyzer = PerformanceAnalyzer()
        
        # Analyze each strategy
        comparison = {}
        for name, returns in strategies.items():
            values = np.cumprod(1 + returns) * 100000
            comparison[name] = analyzer.comprehensive_analysis(
                returns, values, benchmark_returns, 0.02/252
            )
        
        # Validate comparison structure
        assert len(comparison) == 3
        
        # Check that all strategies have same metrics
        metric_keys = set(comparison['Strategy_A'].keys())
        for strategy in comparison.values():
            assert set(strategy.keys()) == metric_keys
        
        # Validate ranking capabilities
        sharpe_ratios = {name: metrics['sharpe_ratio'] 
                        for name, metrics in comparison.items()}
        
        # Should be able to rank strategies
        ranked_strategies = sorted(sharpe_ratios.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        assert len(ranked_strategies) == 3
        assert all(isinstance(ratio, float) for _, ratio in ranked_strategies)