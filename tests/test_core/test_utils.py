"""
Tests for utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import yaml
import json

from quantlib.core.utils import (
    Logger, Config, DateUtils, ValidationUtils, PerformanceUtils
)


class TestLogger:
    """
    Test cases for Logger utility.
    """
    
    def test_logger_creation(self):
        """
        Test logger creation and configuration.
        """
        logger = Logger.get_logger("test")
        assert logger is not None
        assert logger.name == "quantlib.test"
        
        # Test that same logger is returned for same name
        logger2 = Logger.get_logger("test")
        assert logger is logger2


class TestConfig:
    """
    Test cases for Config utility.
    """
    
    def test_default_config(self):
        """
        Test default configuration.
        """
        config = Config.get_default_config()
        assert isinstance(config, dict)
        assert 'data_dir' in config
        assert 'risk' in config
        assert 'performance' in config
        assert 'backtesting' in config
    
    def test_config_loading(self):
        """
        Test configuration loading from file.
        """
        test_config = {
            'data_dir': './test_data',
            'log_level': 'DEBUG',
            'database': {
                'type': 'postgresql',
                'host': 'localhost'
            }
        }
        
        # Test YAML loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            yaml_path = f.name
        
        try:
            loaded_config = Config.load_config(yaml_path)
            assert loaded_config == test_config
        finally:
            import os
            os.unlink(yaml_path)
        
        # Test JSON loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            json_path = f.name
        
        try:
            loaded_config = Config.load_config(json_path)
            assert loaded_config == test_config
        finally:
            import os
            os.unlink(json_path)
    
    def test_config_get(self):
        """
        Test configuration value retrieval.
        """
        test_config = {
            'database': {
                'type': 'sqlite',
                'path': './test.db'
            },
            'log_level': 'INFO'
        }
        
        Config.set_config(test_config)
        
        # Test simple key
        assert Config.get('log_level') == 'INFO'
        
        # Test nested key
        assert Config.get('database.type') == 'sqlite'
        assert Config.get('database.path') == './test.db'
        
        # Test default value
        assert Config.get('nonexistent', 'default') == 'default'
        assert Config.get('database.nonexistent', 'default') == 'default'


class TestDateUtils:
    """
    Test cases for DateUtils utility.
    """
    
    def test_trading_days(self):
        """
        Test trading days calculation.
        """
        start_date = datetime(2023, 1, 1)  # Sunday
        end_date = datetime(2023, 1, 7)    # Saturday
        
        trading_days = DateUtils.get_trading_days(start_date, end_date)
        
        # Should exclude weekends
        assert len(trading_days) == 5  # Mon-Fri
        assert all(day.weekday() < 5 for day in trading_days)
    
    def test_business_days(self):
        """
        Test business days addition.
        """
        start_date = datetime(2023, 1, 2)  # Monday
        result = DateUtils.add_business_days(start_date, 5)
        
        # Should be the following Monday (skipping weekend)
        expected = datetime(2023, 1, 9)
        assert result.date() == expected.date()
    
    def test_period_start(self):
        """
        Test period start calculation.
        """
        test_date = datetime(2023, 6, 15, 14, 30, 45)  # Mid-month, mid-day
        
        # Test day start
        day_start = DateUtils.get_period_start(test_date, 'day')
        assert day_start == datetime(2023, 6, 15, 0, 0, 0)
        
        # Test month start
        month_start = DateUtils.get_period_start(test_date, 'month')
        assert month_start == datetime(2023, 6, 1, 0, 0, 0)
        
        # Test year start
        year_start = DateUtils.get_period_start(test_date, 'year')
        assert year_start == datetime(2023, 1, 1, 0, 0, 0)
    
    def test_market_hours(self):
        """
        Test market hours calculation.
        """
        test_date = datetime(2023, 6, 15)
        
        open_time, close_time = DateUtils.get_market_hours(test_date, 'NYSE')
        
        assert open_time.hour == 9
        assert open_time.minute == 30
        assert close_time.hour == 16
        assert close_time.minute == 0


class TestValidationUtils:
    """
    Test cases for ValidationUtils.
    """
    
    def test_price_data_validation(self, sample_price_data):
        """
        Test price data validation.
        """
        report = ValidationUtils.validate_price_data(sample_price_data)
        
        assert isinstance(report, dict)
        assert 'valid' in report
        assert 'issues' in report
        assert 'warnings' in report
        assert 'stats' in report
        
        # Valid data should pass validation
        assert report['valid'] is True
    
    def test_returns_validation(self, sample_returns):
        """
        Test returns validation.
        """
        report = ValidationUtils.validate_returns(sample_returns)
        
        assert isinstance(report, dict)
        assert 'valid' in report
        assert 'stats' in report
        
        # Check statistics
        stats = report['stats']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats


class TestPerformanceUtils:
    """
    Test cases for PerformanceUtils.
    """
    
    def test_sharpe_ratio(self, sample_returns):
        """
        Test Sharpe ratio calculation.
        """
        sharpe = PerformanceUtils.calculate_sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_max_drawdown(self, sample_returns):
        """
        Test maximum drawdown calculation.
        """
        dd_metrics = PerformanceUtils.calculate_max_drawdown(sample_returns)
        
        assert isinstance(dd_metrics, dict)
        assert 'max_drawdown' in dd_metrics
        assert 'max_drawdown_date' in dd_metrics
        assert 'recovery_date' in dd_metrics
        assert 'recovery_days' in dd_metrics
        
        # Max drawdown should be negative or zero
        assert dd_metrics['max_drawdown'] <= 0