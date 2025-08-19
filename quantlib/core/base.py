"""
Abstract base classes for the QuantLib framework.

Provides foundational interfaces for strategies, indicators, and portfolios.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from quantlib.core.utils import Logger


class BaseComponent(ABC):
    """
    Abstract base class for all QuantLib components.
    
    Provides common functionality for logging, configuration, and lifecycle management.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = Logger.get_logger(f"component.{name}")
        self._is_initialized = False
        self._is_running = False
        
    def initialize(self) -> None:
        """
        Initialize the component.
        """
        if self._is_initialized:
            self.logger.warning(f"Component {self.name} already initialized")
            return
            
        self._initialize()
        self._is_initialized = True
        self.logger.info(f"Component {self.name} initialized")
        
    def start(self) -> None:
        """
        Start the component.
        """
        if not self._is_initialized:
            self.initialize()
            
        if self._is_running:
            self.logger.warning(f"Component {self.name} already running")
            return
            
        self._start()
        self._is_running = True
        self.logger.info(f"Component {self.name} started")
        
    def stop(self) -> None:
        """
        Stop the component.
        """
        if not self._is_running:
            self.logger.warning(f"Component {self.name} not running")
            return
            
        self._stop()
        self._is_running = False
        self.logger.info(f"Component {self.name} stopped")
        
    def cleanup(self) -> None:
        """
        Cleanup component resources.
        """
        if self._is_running:
            self.stop()
            
        self._cleanup()
        self._is_initialized = False
        self.logger.info(f"Component {self.name} cleaned up")
        
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._is_initialized
        
    @property
    def is_running(self) -> bool:
        """Check if component is running."""
        return self._is_running
        
    def _initialize(self) -> None:
        """Override this method for component-specific initialization."""
        pass
        
    def _start(self) -> None:
        """Override this method for component-specific start logic."""
        pass
        
    def _stop(self) -> None:
        """Override this method for component-specific stop logic."""
        pass
        
    def _cleanup(self) -> None:
        """Override this method for component-specific cleanup."""
        pass


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All trading strategies should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = Logger.get_logger(f"strategy.{name}")
        self.positions = {}
        self.signals = []
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Series with trading signals (1: buy, -1: sell, 0: hold)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: float, current_price: float, 
                              portfolio_value: float) -> float:
        """
        Calculate position size based on signal strength and risk management.
        
        Args:
            signal: Trading signal strength
            current_price: Current asset price
            portfolio_value: Total portfolio value
            
        Returns:
            Position size (number of shares/contracts)
        """
        pass
    
    def update_performance(self, returns: pd.Series) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            returns: Strategy returns series
        """
        self.performance_metrics.update({
            'total_return': returns.sum(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
            'win_rate': (returns > 0).mean(),
        })


class BaseIndicator(ABC):
    """
    Abstract base class for technical indicators.
    
    All technical indicators should inherit from this class.
    """
    
    def __init__(self, name: str, period: int = 20, **kwargs):
        self.name = name
        self.period = period
        self.params = kwargs
        self.logger = Logger.get_logger(f"indicator.{name}")
        
    @abstractmethod
    def calculate(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate the indicator values.
        
        Args:
            data: Price data (Series) or OHLCV data (DataFrame)
            
        Returns:
            Calculated indicator values
        """
        pass
    
    def validate_data(self, data: Union[pd.Series, pd.DataFrame]) -> bool:
        """
        Validate input data for indicator calculation.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if len(data) < self.period:
            self.logger.warning(f"Insufficient data for {self.name}: {len(data)} < {self.period}")
            return False
        return True


class BasePortfolio(ABC):
    """
    Abstract base class for portfolio management.
    
    Handles position tracking, risk management, and performance calculation.
    """
    
    def __init__(self, initial_capital: float, name: str = "Portfolio"):
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.performance_history = []
        self.logger = Logger.get_logger(f"portfolio.{name}")
        
    @abstractmethod
    def add_position(self, symbol: str, quantity: float, price: float, 
                   timestamp: datetime) -> None:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Asset symbol
            quantity: Number of shares/contracts
            price: Entry price
            timestamp: Trade timestamp
        """
        pass
    
    @abstractmethod
    def close_position(self, symbol: str, quantity: float, price: float,
                      timestamp: datetime) -> None:
        """
        Close an existing position.
        
        Args:
            symbol: Asset symbol
            quantity: Number of shares/contracts to close
            price: Exit price
            timestamp: Trade timestamp
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_prices: Dictionary of current asset prices
            
        Returns:
            Total portfolio value
        """
        pass
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get portfolio performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_history:
            return {}
            
        returns = pd.Series([p['return'] for p in self.performance_history])
        
        return {
            'total_return': (self.current_capital / self.initial_capital - 1) * 100,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'total_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t.get('pnl', 0) > 0]) / len(self.trades) * 100 if self.trades else 0,
        }
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown percentage
        """
        if not self.performance_history:
            return 0.0
            
        values = [p['portfolio_value'] for p in self.performance_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd * 100