"""
Pytest configuration and fixtures for QuantLib tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quantlib.core.data import MarketData
from quantlib.core.utils import Config, Logger


@pytest.fixture(scope="session")
def setup_test_environment():
    """
    Setup test environment.
    """
    # Configure logging for tests
    Logger.configure(level="DEBUG")
    
    # Set test configuration
    test_config = {
        'data_dir': './test_data',
        'cache_dir': './test_cache',
        'log_level': 'DEBUG',
        'database': {
            'type': 'sqlite',
            'path': ':memory:'  # In-memory database for tests
        }
    }
    Config.set_config(test_config)
    
    yield
    
    # Cleanup after tests
    pass


@pytest.fixture
def sample_price_data():
    """
    Generate sample price data for testing.
    """
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic price data
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = initial_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationships
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data


@pytest.fixture
def sample_market_data(sample_price_data):
    """
    Create MarketData object for testing.
    """
    return MarketData(sample_price_data, "TEST")


@pytest.fixture
def sample_returns(sample_price_data):
    """
    Generate sample returns for testing.
    """
    return sample_price_data['close'].pct_change().dropna()