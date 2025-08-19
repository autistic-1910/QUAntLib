"""
Utility functions and classes for the QuantLib framework.

Provides logging, configuration management, date utilities, and other helper functions.
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import pytz
from dateutil.relativedelta import relativedelta
import pandas as pd


class Logger:
    """
    Centralized logging system for QuantLib.
    """
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def configure(cls, level: str = "INFO", log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
        """
        Configure the logging system.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            format_string: Custom format string
        """
        if cls._configured:
            return
            
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=format_string,
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(log_file)] if log_file else [])
            ]
        )
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if not cls._configured:
            cls.configure()
            
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(f"quantlib.{name}")
            
        return cls._loggers[name]


class Config:
    """
    Configuration management for QuantLib.
    """
    
    _config = None
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        cls._config = config
        return config
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        if cls._config is None:
            cls._config = cls.get_default_config()
        return cls._config
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'data_dir': './data',
            'cache_dir': './cache',
            'log_level': 'INFO',
            'log_file': None,
            'database': {
                'type': 'sqlite',
                'path': './data/quantlib.db'
            },
            'risk': {
                'confidence_levels': [0.95, 0.99],
                'var_methods': ['historical', 'parametric', 'monte_carlo'],
                'monte_carlo_simulations': 10000
            },
            'performance': {
                'benchmark': 'SPY',
                'risk_free_rate': 0.02,
                'trading_days_per_year': 252
            },
            'backtesting': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005
            }
        }
    
    @classmethod
    def set_config(cls, config: Dict[str, Any]) -> None:
        """
        Set configuration.
        
        Args:
            config: Configuration dictionary
        """
        cls._config = config
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        config = cls.get_config()
        
        # Support dot notation (e.g., 'database.type')
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value


class DateUtils:
    """
    Date and time utilities for financial calculations.
    """
    
    @staticmethod
    def get_trading_days(start_date: datetime, end_date: datetime,
                        calendar: str = 'NYSE') -> pd.DatetimeIndex:
        """
        Get trading days between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            calendar: Trading calendar ('NYSE', 'NASDAQ', etc.)
            
        Returns:
            DatetimeIndex of trading days
        """
        # Simplified implementation - excludes weekends
        # In practice, you'd use a proper trading calendar library
        all_days = pd.date_range(start_date, end_date, freq='D')
        trading_days = all_days[all_days.weekday < 5]  # Monday=0, Sunday=6
        
        # TODO: Add holiday exclusions based on calendar
        return trading_days
    
    @staticmethod
    def add_business_days(date: datetime, days: int) -> datetime:
        """
        Add business days to a date.
        
        Args:
            date: Starting date
            days: Number of business days to add
            
        Returns:
            New date
        """
        return pd.bdate_range(start=date, periods=days + 1)[-1].to_pydatetime()
    
    @staticmethod
    def get_period_start(date: datetime, period: str) -> datetime:
        """
        Get the start of a period containing the given date.
        
        Args:
            date: Reference date
            period: Period type ('day', 'week', 'month', 'quarter', 'year')
            
        Returns:
            Start of period
        """
        if period == 'day':
            return date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            return date - timedelta(days=date.weekday())
        elif period == 'month':
            return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == 'quarter':
            quarter_start_month = ((date.month - 1) // 3) * 3 + 1
            return date.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == 'year':
            return date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported period: {period}")
    
    @staticmethod
    def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """
        Convert datetime between timezones.
        
        Args:
            dt: Datetime to convert
            from_tz: Source timezone
            to_tz: Target timezone
            
        Returns:
            Converted datetime
        """
        from_timezone = pytz.timezone(from_tz)
        to_timezone = pytz.timezone(to_tz)
        
        # Localize if naive
        if dt.tzinfo is None:
            dt = from_timezone.localize(dt)
        
        return dt.astimezone(to_timezone)
    
    @staticmethod
    def get_market_hours(date: datetime, market: str = 'NYSE') -> tuple:
        """
        Get market opening and closing hours for a given date.
        
        Args:
            date: Trading date
            market: Market identifier
            
        Returns:
            Tuple of (open_time, close_time)
        """
        # Simplified implementation
        market_hours = {
            'NYSE': ('09:30', '16:00'),
            'NASDAQ': ('09:30', '16:00'),
            'LSE': ('08:00', '16:30'),
            'TSE': ('09:00', '15:00'),
        }
        
        if market not in market_hours:
            raise ValueError(f"Unsupported market: {market}")
        
        open_str, close_str = market_hours[market]
        
        open_time = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {open_str}", '%Y-%m-%d %H:%M')
        close_time = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {close_str}", '%Y-%m-%d %H:%M')
        
        return open_time, close_time


class ValidationUtils:
    """
    Data validation utilities.
    """
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate price data for common issues.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            report['valid'] = False
            report['issues'].append(f"Missing columns: {missing_columns}")
            return report
        
        # Check for negative prices
        negative_prices = (data[['open', 'high', 'low', 'close']] < 0).any(axis=1)
        if negative_prices.any():
            report['issues'].append(f"Found {negative_prices.sum()} rows with negative prices")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            report['issues'].append(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            report['warnings'].append(f"Missing values: {missing_values.to_dict()}")
        
        # Check for duplicated timestamps
        duplicated_index = data.index.duplicated()
        if duplicated_index.any():
            report['warnings'].append(f"Found {duplicated_index.sum()} duplicated timestamps")
        
        # Calculate basic statistics
        report['stats'] = {
            'total_rows': len(data),
            'date_range': (data.index.min(), data.index.max()),
            'avg_volume': data['volume'].mean(),
            'price_range': (data['close'].min(), data['close'].max())
        }
        
        if report['issues']:
            report['valid'] = False
        
        return report
    
    @staticmethod
    def validate_returns(returns: pd.Series, max_return: float = 0.5) -> Dict[str, Any]:
        """
        Validate return series for outliers and anomalies.
        
        Args:
            returns: Return series
            max_return: Maximum reasonable daily return
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for extreme returns
        extreme_returns = (returns.abs() > max_return)
        if extreme_returns.any():
            report['warnings'].append(f"Found {extreme_returns.sum()} extreme returns (>{max_return*100}%)")
        
        # Check for infinite or NaN values
        import numpy as np  # Local import to avoid global dependency if not needed elsewhere
        invalid_values = ~returns.replace([np.inf, -np.inf], np.nan).notna()
        if invalid_values.any():
            report['issues'].append(f"Found {invalid_values.sum()} invalid return values")
            report['valid'] = False
        
        # Calculate statistics
        report['stats'] = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max()
        }
        
        return report


class PerformanceUtils:
    """
    Performance calculation utilities.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        import numpy as np  # Local import to ensure np is defined
        excess_returns = returns - risk_free_rate / periods_per_year
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with drawdown metrics
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find recovery date
        recovery_date = None
        if max_dd_date < drawdown.index[-1]:
            post_max_dd = drawdown[max_dd_date:]
            recovery_mask = post_max_dd >= 0
            if recovery_mask.any():
                recovery_date = post_max_dd[recovery_mask].index[0]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'recovery_date': recovery_date,
            'recovery_days': (recovery_date - max_dd_date).days if recovery_date else None
        }