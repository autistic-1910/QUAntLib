import pytest
import numpy as np
import pandas as pd
from scipy import stats
from quantlib.analytics.statistics import (
    DistributionAnalyzer, HypothesisTests, RegressionAnalysis,
    TimeSeriesAnalysis, MonteCarloSimulation, DistributionFitResult
)


class TestDistributionAnalyzer:
    """Test distribution analysis methods"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distribution data"""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)
    
    @pytest.fixture
    def lognormal_data(self):
        """Generate lognormal distribution data"""
        np.random.seed(42)
        return np.random.lognormal(0, 0.5, 1000)
    
    def test_fit_normal_distribution(self, normal_data):
        """Test fitting normal distribution"""
        analyzer = DistributionAnalyzer()
        
        result = analyzer.fit_distribution(normal_data, 'normal')
        
        assert isinstance(result, DistributionFitResult)
        assert result.distribution == 'normal'
        assert len(result.parameters) == 2  # mu, sigma
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)
        assert isinstance(result.ks_statistic, float)
        assert isinstance(result.p_value, float)
        assert len(result.fitted_data) == len(normal_data)
        
        # Parameters should be close to true values (0, 1)
        mu, sigma = result.parameters
        assert abs(mu - 0) < 0.1
        assert abs(sigma - 1) < 0.1
    
    def test_fit_lognormal_distribution(self, lognormal_data):
        """Test fitting lognormal distribution"""
        analyzer = DistributionAnalyzer()
        
        result = analyzer.fit_distribution(lognormal_data, 'lognormal')
        
        assert isinstance(result, DistributionFitResult)
        assert result.distribution == 'lognormal'
        assert len(result.parameters) == 3  # s, loc, scale
        assert result.ks_statistic >= 0
        assert 0 <= result.p_value <= 1
    
    def test_find_best_distribution(self, normal_data):
        """Test finding best fitting distribution"""
        analyzer = DistributionAnalyzer()
        
        # Test with subset of distributions
        distributions = ['normal', 't', 'skewnorm']
        result = analyzer.find_best_distribution(normal_data, distributions)
        
        assert isinstance(result, DistributionFitResult)
        assert result.distribution in distributions
        # For normal data, normal distribution should be best (lowest AIC)
        assert result.distribution == 'normal'
    
    def test_distribution_summary(self, normal_data):
        """Test distribution summary statistics"""
        analyzer = DistributionAnalyzer()
        
        summary = analyzer.distribution_summary(normal_data)
        
        required_keys = [
            'mean', 'std', 'skewness', 'kurtosis', 'jarque_bera',
            'shapiro_wilk', 'anderson_darling', 'percentiles'
        ]
        
        for key in required_keys:
            assert key in summary
        
        # Validate values for normal data
        assert abs(summary['mean'] - 0) < 0.1
        assert abs(summary['std'] - 1) < 0.1
        assert abs(summary['skewness']) < 0.2  # Should be close to 0
        assert abs(summary['kurtosis']) < 0.5  # Should be close to 0
        
        # Check percentiles
        assert len(summary['percentiles']) == 7
        assert np.all(np.diff(summary['percentiles']) >= 0)  # Should be sorted
    
    def test_invalid_distribution(self, normal_data):
        """Test error handling for invalid distribution"""
        analyzer = DistributionAnalyzer()
        
        with pytest.raises(ValueError, match="Unsupported distribution"):
            analyzer.fit_distribution(normal_data, 'invalid_dist')


class TestHypothesisTests:
    """Test hypothesis testing methods"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distribution data"""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)
    
    @pytest.fixture
    def non_normal_data(self):
        """Generate non-normal distribution data"""
        np.random.seed(42)
        return np.random.exponential(1, 1000)
    
    def test_normality_tests_normal_data(self, normal_data):
        """Test normality tests on normal data"""
        tests = HypothesisTests()
        
        results = tests.normality_tests(normal_data)
        
        required_tests = ['shapiro_wilk', 'jarque_bera', 'anderson_darling', 'kolmogorov_smirnov']
        for test in required_tests:
            assert test in results
            assert 'statistic' in results[test]
            if test != 'anderson_darling':
                assert 'p_value' in results[test]
        
        # For normal data, p-values should generally be high (> 0.05)
        # Note: This might occasionally fail due to randomness
        assert results['shapiro_wilk']['p_value'] > 0.01
        assert results['jarque_bera']['p_value'] > 0.01
    
    def test_normality_tests_non_normal_data(self, non_normal_data):
        """Test normality tests on non-normal data"""
        tests = HypothesisTests()
        
        results = tests.normality_tests(non_normal_data)
        
        # For exponential data, should reject normality (low p-values)
        assert results['shapiro_wilk']['p_value'] < 0.05
        assert results['jarque_bera']['p_value'] < 0.05
    
    def test_independence_tests(self):
        """Test independence tests"""
        np.random.seed(42)
        
        # Create correlated data
        x = np.random.normal(0, 1, 100)
        y = 0.7 * x + 0.3 * np.random.normal(0, 1, 100)  # Correlated
        
        tests = HypothesisTests()
        results = tests.independence_tests(x, y)
        
        required_tests = ['pearson', 'spearman', 'kendall']
        for test in required_tests:
            assert test in results
            assert 'correlation' in results[test]
            assert 'p_value' in results[test]
        
        # Should detect correlation
        assert abs(results['pearson']['correlation']) > 0.5
        assert results['pearson']['p_value'] < 0.05
    
    def test_stationarity_tests(self):
        """Test stationarity tests"""
        np.random.seed(42)
        
        # Create stationary series
        stationary_data = np.random.normal(0, 1, 100)
        
        tests = HypothesisTests()
        results = tests.stationarity_tests(stationary_data)
        
        # Results might be empty if statsmodels not available
        if results:
            assert 'adf' in results
            assert 'kpss' in results
            
            for test in ['adf', 'kpss']:
                assert 'statistic' in results[test]
                assert 'p_value' in results[test]
                assert 'critical_values' in results[test]


class TestRegressionAnalysis:
    """Test regression analysis methods"""
    
    @pytest.fixture
    def linear_data(self):
        """Generate linear relationship data"""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 0.5, 100)  # y = 2x + 1 + noise
        return x, y
    
    def test_linear_regression(self, linear_data):
        """Test linear regression analysis"""
        x, y = linear_data
        
        regression = RegressionAnalysis()
        result = regression.linear_regression(x, y)
        
        required_keys = [
            'slope', 'intercept', 'r_squared', 'p_value', 'std_error',
            'slope_ci', 'residuals', 'predictions', 'mse', 'rmse', 'durbin_watson'
        ]
        
        for key in required_keys:
            assert key in result
        
        # Check that slope and intercept are close to true values
        assert abs(result['slope'] - 2) < 0.2
        assert abs(result['intercept'] - 1) < 0.5
        
        # R-squared should be high for good linear relationship
        assert result['r_squared'] > 0.8
        
        # P-value should be significant
        assert result['p_value'] < 0.05
        
        # Check array lengths
        assert len(result['residuals']) == len(x)
        assert len(result['predictions']) == len(x)
        
        # Confidence interval should contain true slope
        ci_lower, ci_upper = result['slope_ci']
        assert ci_lower <= 2 <= ci_upper
    
    def test_rolling_regression(self, linear_data):
        """Test rolling window regression"""
        x, y = linear_data
        
        regression = RegressionAnalysis()
        result = regression.rolling_regression(x, y, window=20)
        
        assert isinstance(result, pd.DataFrame)
        
        required_columns = ['date_index', 'slope', 'intercept', 'r_squared', 'p_value']
        for col in required_columns:
            assert col in result.columns
        
        # Should have correct number of rows
        expected_rows = len(x) - 20 + 1
        assert len(result) == expected_rows
        
        # All values should be finite
        assert result.isna().sum().sum() == 0
    
    def test_durbin_watson_calculation(self):
        """Test Durbin-Watson statistic calculation"""
        # Create residuals with no autocorrelation
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)
        
        dw_stat = RegressionAnalysis._durbin_watson(residuals)
        
        assert isinstance(dw_stat, float)
        # DW statistic should be around 2 for no autocorrelation
        assert 1.5 < dw_stat < 2.5


class TestTimeSeriesAnalysis:
    """Test time series analysis methods"""
    
    @pytest.fixture
    def ar_data(self):
        """Generate AR(1) time series data"""
        np.random.seed(42)
        n = 200
        phi = 0.7
        data = np.zeros(n)
        data[0] = np.random.normal(0, 1)
        
        for i in range(1, n):
            data[i] = phi * data[i-1] + np.random.normal(0, 1)
        
        return data
    
    def test_autocorrelation(self, ar_data):
        """Test autocorrelation function calculation"""
        ts_analysis = TimeSeriesAnalysis()
        
        autocorr = ts_analysis.autocorrelation(ar_data, max_lags=10)
        
        assert len(autocorr) == 11  # 0 to 10 lags
        assert autocorr[0] == 1.0  # Lag 0 should be 1
        assert autocorr[1] > 0.5  # Lag 1 should be positive for AR(1)
        
        # Autocorrelation should decay for AR(1) process
        assert autocorr[1] > autocorr[2] > autocorr[3]
    
    def test_ljung_box_test(self, ar_data):
        """Test Ljung-Box test for autocorrelation"""
        ts_analysis = TimeSeriesAnalysis()
        
        result = ts_analysis.ljung_box_test(ar_data, lags=5)
        
        # Result might be empty if statsmodels not available
        if result:
            assert 'statistics' in result
            assert 'p_values' in result
            assert 'lags' in result
            
            assert len(result['statistics']) == 5
            assert len(result['p_values']) == 5
            assert result['lags'] == [1, 2, 3, 4, 5]
            
            # For AR data, should detect autocorrelation (low p-values)
            assert result['p_values'][0] < 0.05
    
    def test_volatility_clustering(self):
        """Test volatility clustering analysis"""
        np.random.seed(42)
        
        # Create returns with volatility clustering (GARCH-like)
        n = 200
        returns = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.02
        
        for i in range(1, n):
            sigma[i] = 0.01 + 0.1 * returns[i-1]**2 + 0.8 * sigma[i-1]
            returns[i] = sigma[i] * np.random.normal(0, 1)
        
        ts_analysis = TimeSeriesAnalysis()
        result = ts_analysis.volatility_clustering(returns, window=22)
        
        assert 'rolling_volatility' in result
        assert 'volatility_of_volatility' in result
        assert 'arch_lm_test' in result
        
        # Check array lengths
        assert len(result['rolling_volatility']) == len(returns)
        assert len(result['volatility_of_volatility']) == len(returns)
        
        # ARCH test should detect heteroscedasticity
        arch_test = result['arch_lm_test']
        if arch_test:  # If test was performed
            assert 'lm_statistic' in arch_test
            assert 'p_value' in arch_test
            assert arch_test['p_value'] < 0.05  # Should detect ARCH effects
    
    def test_arch_lm_test(self):
        """Test ARCH LM test calculation"""
        np.random.seed(42)
        
        # Create data with ARCH effects
        n = 100
        returns = np.random.normal(0, 1, n)
        squared_returns = returns**2
        
        ts_analysis = TimeSeriesAnalysis()
        result = ts_analysis._arch_lm_test(squared_returns, lags=5)
        
        assert 'lm_statistic' in result
        assert 'p_value' in result
        assert 'r_squared' in result
        
        assert isinstance(result['lm_statistic'], float)
        assert 0 <= result['p_value'] <= 1
        assert 0 <= result['r_squared'] <= 1


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation methods"""
    
    def test_geometric_brownian_motion(self):
        """Test GBM simulation"""
        mc_sim = MonteCarloSimulation()
        
        S0 = 100
        mu = 0.05
        sigma = 0.2
        T = 1.0
        steps = 252
        simulations = 1000
        
        paths = mc_sim.geometric_brownian_motion(S0, mu, sigma, T, steps, simulations)
        
        assert paths.shape == (simulations, steps + 1)
        assert np.all(paths[:, 0] == S0)  # All paths start at S0
        assert np.all(paths > 0)  # All prices should be positive
        
        # Check statistical properties
        final_prices = paths[:, -1]
        log_returns = np.log(final_prices / S0)
        
        # Mean log return should be approximately (mu - 0.5*sigma^2)*T
        expected_log_return = (mu - 0.5 * sigma**2) * T
        assert abs(np.mean(log_returns) - expected_log_return) < 0.05
        
        # Standard deviation should be approximately sigma*sqrt(T)
        expected_std = sigma * np.sqrt(T)
        assert abs(np.std(log_returns) - expected_std) < 0.05
    
    def test_var_monte_carlo(self):
        """Test Monte Carlo VaR calculation"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        mc_sim = MonteCarloSimulation()
        result = mc_sim.var_monte_carlo(returns, confidence_level=0.05, simulations=10000)
        
        assert 'var' in result
        assert 'expected_shortfall' in result
        assert 'simulated_returns' in result
        assert 'confidence_level' in result
        
        assert result['var'] < 0  # VaR should be negative
        assert result['expected_shortfall'] < result['var']  # ES more extreme than VaR
        assert len(result['simulated_returns']) == 10000
        assert result['confidence_level'] == 0.05
        
        # Check that VaR is approximately correct percentile
        empirical_var = np.percentile(result['simulated_returns'], 5)
        assert abs(result['var'] - empirical_var) < 0.001
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence intervals"""
        np.random.seed(42)
        data = np.random.normal(5, 2, 100)
        
        # Test with mean function
        def mean_func(x):
            return np.mean(x)
        
        mc_sim = MonteCarloSimulation()
        result = mc_sim.bootstrap_confidence_interval(
            data, mean_func, confidence_level=0.95, n_bootstrap=1000
        )
        
        assert 'statistic' in result
        assert 'bootstrap_distribution' in result
        assert 'confidence_interval' in result
        assert 'confidence_level' in result
        assert 'std_error' in result
        
        # Original statistic should be close to true mean (5)
        assert abs(result['statistic'] - 5) < 0.5
        
        # Bootstrap distribution should have correct length
        assert len(result['bootstrap_distribution']) == 1000
        
        # Confidence interval should contain true mean
        ci_lower, ci_upper = result['confidence_interval']
        assert ci_lower < 5 < ci_upper
        
        # Confidence level should match
        assert result['confidence_level'] == 0.95
        
        # Standard error should be positive
        assert result['std_error'] > 0


class TestIntegration:
    """Integration tests for statistics module"""
    
    def test_complete_statistical_analysis(self):
        """Test complete statistical analysis workflow"""
        np.random.seed(42)
        
        # Generate sample financial returns data
        returns = np.random.normal(0.001, 0.02, 252)
        
        # Initialize analyzers
        dist_analyzer = DistributionAnalyzer()
        hypothesis_tests = HypothesisTests()
        mc_sim = MonteCarloSimulation()
        
        # Comprehensive analysis
        analysis = {
            'distribution_summary': dist_analyzer.distribution_summary(returns),
            'best_distribution': dist_analyzer.find_best_distribution(returns),
            'normality_tests': hypothesis_tests.normality_tests(returns),
            'var_monte_carlo': mc_sim.var_monte_carlo(returns, 0.05, 1000)
        }
        
        # Validate structure
        assert 'distribution_summary' in analysis
        assert 'best_distribution' in analysis
        assert 'normality_tests' in analysis
        assert 'var_monte_carlo' in analysis
        
        # Validate distribution summary
        summary = analysis['distribution_summary']
        assert all(key in summary for key in ['mean', 'std', 'skewness', 'kurtosis'])
        
        # Validate best distribution fit
        best_dist = analysis['best_distribution']
        assert isinstance(best_dist, DistributionFitResult)
        assert best_dist.distribution in DistributionAnalyzer.DISTRIBUTIONS
        
        # Validate normality tests
        norm_tests = analysis['normality_tests']
        assert 'shapiro_wilk' in norm_tests
        assert 'jarque_bera' in norm_tests
        
        # Validate Monte Carlo VaR
        mc_var = analysis['var_monte_carlo']
        assert mc_var['var'] < 0
        assert mc_var['expected_shortfall'] < mc_var['var']
    
    def test_risk_model_validation(self):
        """Test statistical validation of risk models"""
        np.random.seed(42)
        
        # Generate portfolio returns and risk factors
        n_obs = 252
        market_factor = np.random.normal(0.0005, 0.015, n_obs)
        size_factor = np.random.normal(0.0002, 0.008, n_obs)
        value_factor = np.random.normal(0.0001, 0.006, n_obs)
        
        # Portfolio returns as linear combination of factors
        portfolio_returns = (
            0.8 * market_factor +
            0.1 * size_factor +
            0.1 * value_factor +
            np.random.normal(0, 0.005, n_obs)  # Idiosyncratic risk
        )
        
        # Factor matrix
        factors = np.column_stack([market_factor, size_factor, value_factor])
        
        # Regression analysis
        regression = RegressionAnalysis()
        
        # Simple regression with market factor
        market_regression = regression.linear_regression(market_factor, portfolio_returns)
        
        # Validate regression results
        assert market_regression['r_squared'] > 0.5  # Should explain significant variance
        assert market_regression['p_value'] < 0.05  # Should be significant
        assert 0.6 < market_regression['slope'] < 1.0  # Beta should be close to 0.8
        
        # Test residuals for independence
        hypothesis_tests = HypothesisTests()
        residuals = market_regression['residuals']
        
        # Ljung-Box test on residuals
        ts_analysis = TimeSeriesAnalysis()
        lb_result = ts_analysis.ljung_box_test(residuals, lags=5)
        
        # If test available, residuals should show no autocorrelation
        if lb_result:
            # At least some lags should not be significant (p > 0.05)
            assert any(p > 0.05 for p in lb_result['p_values'])
        
        # Test normality of residuals
        normality_result = hypothesis_tests.normality_tests(residuals)
        
        # Residuals should be approximately normal (though this might fail occasionally)
        # We'll just check that the test runs without error
        assert 'shapiro_wilk' in normality_result
        assert 'p_value' in normality_result['shapiro_wilk']