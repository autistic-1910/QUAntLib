"""
Tests for data management functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from quantlib.core.data import MarketData, DataManager


class TestMarketData:
    """
    Test cases for MarketData class.
    """
    
    def test_market_data_creation(self, sample_price_data):
        """
        Test MarketData object creation.
        """
        market_data = MarketData(sample_price_data, "TEST")
        
        assert market_data.symbol == "TEST"
        assert isinstance(market_data.data, pd.DataFrame)
        assert len(market_data.data) > 0
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in market_data.data.columns for col in required_cols)
    
    def test_returns_calculation(self, sample_market_data):
        """
        Test returns calculation.
        """
        # Test simple returns
        simple_returns = sample_market_data.get_returns('simple')
        assert isinstance(simple_returns, pd.Series)
        assert len(simple_returns) == len(sample_market_data.data) - 1
        
        # Test log returns
        log_returns = sample_market_data.get_returns('log')
        assert isinstance(log_returns, pd.Series)
        assert len(log_returns) == len(sample_market_data.data) - 1
        
        # Test invalid method
        with pytest.raises(ValueError):
            sample_market_data.get_returns('invalid')
    
    def test_resampling(self, sample_market_data):
        """
        Test data resampling.
        """
        # Resample to weekly data
        weekly_data = sample_market_data.resample('W')
        
        assert isinstance(weekly_data, MarketData)
        assert len(weekly_data.data) < len(sample_market_data.data)
        assert weekly_data.symbol == sample_market_data.symbol
        
        # Check OHLCV structure is maintained
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in weekly_data.data.columns for col in required_cols)
    
    def test_data_validation(self):
        """
        Test data validation and cleaning.
        """
        # Create invalid data
        dates = pd.date_range('2023-01-01', periods=10)
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)
        
        # Introduce invalid OHLC relationship
        invalid_data.loc[dates[5], 'high'] = 90  # High < Low
        
        # This should clean the invalid data
        market_data = MarketData(invalid_data, "TEST")
        
        # Check that invalid row was removed
        assert len(market_data.data) == 9  # One row should be removed


class TestDataManager:
    """
    Test cases for DataManager class.
    """
    
    def test_data_manager_creation(self, setup_test_environment):
        """
        Test DataManager creation.
        """
        data_manager = DataManager()
        assert data_manager is not None
        assert hasattr(data_manager, 'cache')
        assert hasattr(data_manager, 'db_path')
    
    def test_csv_loading(self, sample_price_data, setup_test_environment):
        """
        Test CSV data loading.
        """
        data_manager = DataManager()
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_price_data.to_csv(f.name)
            csv_path = f.name
        
        try:
            # Load data from CSV
            market_data = data_manager.load_csv(csv_path, "TEST")
            
            assert isinstance(market_data, MarketData)
            assert market_data.symbol == "TEST"
            assert len(market_data.data) > 0
            
        finally:
            # Cleanup
            os.unlink(csv_path)
    
    def test_yahoo_data_fetch(self, setup_test_environment):
        """
        Test Yahoo Finance data fetching (mock implementation).
        """
        data_manager = DataManager()
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # This uses the mock implementation that generates sample data
        market_data = data_manager.fetch_yahoo_data("AAPL", start_date, end_date)
        
        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "AAPL"
        assert len(market_data.data) > 0
        
        # Check date range
        assert market_data.data.index.min().date() >= start_date.date()
        assert market_data.data.index.max().date() <= end_date.date()