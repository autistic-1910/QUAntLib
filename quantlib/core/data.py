"""
Data management and market data handling.

Provides interfaces for data ingestion, storage, and retrieval.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import requests
import sqlite3
from pathlib import Path

from quantlib.core.utils import Logger, Config


class MarketData:
    """
    Container for market data with OHLCV structure.
    """
    
    def __init__(self, data: pd.DataFrame, symbol: str = ""):
        self.symbol = symbol
        self.logger = Logger.get_logger(f"marketdata.{symbol}")
        self.data = self._validate_and_clean(data)
        
    def _validate_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean market data.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Cleaned and validated DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
                data.index = pd.to_datetime(data.index)
            else:
                raise ValueError("Data must have datetime index or 'date' column")
        
        # Remove duplicates and sort
        data = data.drop_duplicates().sort_index()
        
        # Handle missing values
        data = data.fillna(method='ffill').dropna()
        
        # Validate price relationships
        invalid_rows = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_rows.any():
            self.logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC relationships")
            data = data[~invalid_rows]
        
        return data
    
    def get_returns(self, method: str = 'simple') -> pd.Series:
        """
        Calculate returns.
        
        Args:
            method: 'simple' or 'log' returns
            
        Returns:
            Returns series
        """
        if method == 'simple':
            return self.data['close'].pct_change().dropna()
        elif method == 'log':
            return np.log(self.data['close'] / self.data['close'].shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    def resample(self, frequency: str) -> 'MarketData':
        """
        Resample data to different frequency.
        
        Args:
            frequency: Pandas frequency string (e.g., '1D', '1H', '5T')
            
        Returns:
            New MarketData object with resampled data
        """
        resampled = self.data.resample(frequency).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return MarketData(resampled, self.symbol)


class DataManager:
    """
    Central data management system for market data.
    
    Handles data ingestion from multiple sources, caching, and retrieval.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or Config.get_default_config()
        self.logger = Logger.get_logger("datamanager")
        self.cache = {}
        self._setup_database()
        
    def _setup_database(self) -> None:
        """
        Setup local SQLite database for data caching.
        """
        db_path = Path(self.config.get('data_dir', './data')) / 'market_data.db'
        db_path.parent.mkdir(exist_ok=True)
        
        self.db_path = str(db_path)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """)
    
    def load_csv(self, file_path: str, symbol: str, 
                 date_column: str = 'date') -> MarketData:
        """
        Load market data from CSV file.
        
        Args:
            file_path: Path to CSV file
            symbol: Asset symbol
            date_column: Name of date column
            
        Returns:
            MarketData object
        """
        try:
            data = pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
            self.logger.info(f"Loaded {len(data)} rows for {symbol} from {file_path}")
            
            # Cache in database
            self._cache_to_database(symbol, data)
            
            return MarketData(data, symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV {file_path}: {e}")
            # Fallback: try to infer index if date column missing
            try:
                data = pd.read_csv(file_path)
                # Try to find a date-like column
                candidate_cols = [c for c in data.columns if c.lower() in ('date', 'datetime', 'timestamp')]
                if candidate_cols:
                    dc = candidate_cols[0]
                    data[dc] = pd.to_datetime(data[dc])
                    data = data.set_index(dc)
                else:
                    # Create a synthetic daily index starting today
                    data.index = pd.date_range(start=datetime.today(), periods=len(data), freq='D')
                self.logger.warning("Parsed CSV without explicit date column; inferred index applied")
                self._cache_to_database(symbol, data)
                return MarketData(data, symbol)
            except Exception as e2:
                self.logger.error(f"Fallback CSV parsing failed for {file_path}: {e2}")
                raise
    
    def fetch_yahoo_data(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> MarketData:
        """
        Fetch data from Yahoo Finance (simplified implementation).
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            MarketData object
        """
        # This is a simplified implementation
        # In practice, you'd use yfinance or similar library
        self.logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        # Check cache first
        cached_data = self._get_from_database(symbol, start_date, end_date)
        if cached_data is not None and len(cached_data) > 0:
            self.logger.info(f"Using cached data for {symbol}")
            return MarketData(cached_data, symbol)
        
        # For demo purposes, generate sample data
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)  # For reproducible results
        
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
        
        # Cache in database
        self._cache_to_database(symbol, data)
        
        return MarketData(data, symbol)
    
    def _cache_to_database(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Cache data to SQLite database.
        
        Args:
            symbol: Asset symbol
            data: Market data DataFrame
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data_to_insert = data.copy()
                data_to_insert['symbol'] = symbol
                data_to_insert['date'] = data_to_insert.index.strftime('%Y-%m-%d')
                
                data_to_insert.to_sql('market_data', conn, if_exists='replace', index=False)
                
        except Exception as e:
            self.logger.error(f"Failed to cache data for {symbol}: {e}")
    
    def _get_from_database(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Retrieve data from SQLite database.
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """
                
                data = pd.read_sql_query(
                    query, conn, 
                    params=[symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')],
                    parse_dates=['date'],
                    index_col='date'
                )
                
                return data if len(data) > 0 else None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached data for {symbol}: {e}")
            return None