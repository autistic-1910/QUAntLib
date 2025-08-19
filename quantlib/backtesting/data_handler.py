"""Data handler for backtesting engine.

Provides historical data management and market event generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Iterator, Union
from abc import ABC, abstractmethod

from .events import MarketEvent
from quantlib.core.utils import Logger


class DataHandler(ABC):
    """Abstract base class for data handlers."""
    
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """Get the latest bar for a symbol."""
        pass
        
    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Dict]:
        """Get the latest N bars for a symbol."""
        pass
        
    @abstractmethod
    def update_bars(self) -> MarketEvent:
        """Update bars and return market event."""
        pass
        
    @abstractmethod
    def continue_backtest(self) -> bool:
        """Check if backtest should continue."""
        pass


class HistoricalDataHandler(DataHandler):
    """Historical data handler for backtesting."""
    
    def __init__(self, data: Dict[str, pd.DataFrame], start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None):
        """
        Initialize historical data handler.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            start_date: Start date for backtesting
            end_date: End date for backtesting
        """
        self.data = data
        self.symbols = list(data.keys())
        self.start_date = start_date
        self.end_date = end_date
        self.logger = Logger.get_logger("data_handler")
        
        # Prepare data
        self._prepare_data()
        
        # Initialize iterators
        self.current_datetime = None
        self.bar_index = 0
        self.latest_symbol_data = {symbol: [] for symbol in self.symbols}
        
    def _prepare_data(self):
        """Prepare and validate data for backtesting."""
        # Ensure all DataFrames have datetime index
        for symbol in self.symbols:
            df = self.data[symbol]
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                elif 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    raise ValueError(f"No datetime index or column found for {symbol}")
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            
            # Filter by date range if specified
            if self.start_date:
                df = df[df.index >= self.start_date]
            if self.end_date:
                df = df[df.index <= self.end_date]
                
            self.data[symbol] = df.sort_index()
        
        # Get common date range across all symbols
        self._align_data()
        
    def _align_data(self):
        """Align data across all symbols to common date range."""
        if not self.symbols:
            return
            
        # Find common date range
        start_dates = [self.data[symbol].index.min() for symbol in self.symbols]
        end_dates = [self.data[symbol].index.max() for symbol in self.symbols]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Filter all data to common range
        for symbol in self.symbols:
            self.data[symbol] = self.data[symbol][
                (self.data[symbol].index >= common_start) & 
                (self.data[symbol].index <= common_end)
            ]
        
        # Get unique timestamps across all symbols
        all_timestamps = set()
        for symbol in self.symbols:
            all_timestamps.update(self.data[symbol].index)
        
        self.timestamps = sorted(list(all_timestamps))
        self.logger.info(f"Data aligned: {len(self.timestamps)} timestamps from {common_start} to {common_end}")
        
    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """Get the latest bar for a symbol."""
        if symbol not in self.latest_symbol_data or not self.latest_symbol_data[symbol]:
            return None
        return self.latest_symbol_data[symbol][-1]
        
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Dict]:
        """Get the latest N bars for a symbol."""
        if symbol not in self.latest_symbol_data:
            return []
        return self.latest_symbol_data[symbol][-n:]
        
    def get_latest_bar_datetime(self, symbol: str) -> Optional[datetime]:
        """Get the datetime of the latest bar for a symbol."""
        latest_bar = self.get_latest_bar(symbol)
        return latest_bar['datetime'] if latest_bar else None
        
    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        """Get a specific value from the latest bar."""
        latest_bar = self.get_latest_bar(symbol)
        return latest_bar.get(val_type) if latest_bar else None
        
    def update_bars(self) -> Optional[MarketEvent]:
        """Update bars and return market event."""
        if self.bar_index >= len(self.timestamps):
            return None
            
        self.current_datetime = self.timestamps[self.bar_index]
        
        # Update data for each symbol
        updated_symbols = []
        for symbol in self.symbols:
            df = self.data[symbol]
            
            # Check if we have data for this timestamp
            if self.current_datetime in df.index:
                bar_data = df.loc[self.current_datetime]
                
                bar_dict = {
                    'datetime': self.current_datetime,
                    'symbol': symbol,
                    'open': bar_data.get('open', np.nan),
                    'high': bar_data.get('high', np.nan),
                    'low': bar_data.get('low', np.nan),
                    'close': bar_data.get('close', np.nan),
                    'volume': bar_data.get('volume', 0),
                }
                
                # Add any additional columns
                for col in df.columns:
                    if col not in bar_dict:
                        bar_dict[col] = bar_data.get(col, np.nan)
                
                self.latest_symbol_data[symbol].append(bar_dict)
                updated_symbols.append(symbol)
        
        self.bar_index += 1
        
        if updated_symbols:
            return MarketEvent(self.current_datetime, updated_symbols)
        else:
            # No data for this timestamp, try next one
            return self.update_bars()
            
    def continue_backtest(self) -> bool:
        """Check if backtest should continue."""
        return self.bar_index < len(self.timestamps)
        
    def reset(self):
        """Reset the data handler to the beginning."""
        self.bar_index = 0
        self.current_datetime = None
        self.latest_symbol_data = {symbol: [] for symbol in self.symbols}
        
    def get_data_summary(self) -> Dict:
        """Get summary of available data."""
        summary = {
            'symbols': self.symbols,
            'total_timestamps': len(self.timestamps),
            'start_date': self.timestamps[0] if self.timestamps else None,
            'end_date': self.timestamps[-1] if self.timestamps else None,
            'current_index': self.bar_index,
        }
        
        # Add per-symbol statistics
        for symbol in self.symbols:
            df = self.data[symbol]
            summary[f'{symbol}_bars'] = len(df)
            summary[f'{symbol}_start'] = df.index.min()
            summary[f'{symbol}_end'] = df.index.max()
            
        return summary