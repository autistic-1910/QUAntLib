#!/usr/bin/env python3
"""
Base Strategy Framework

Provides the foundational classes and interfaces for strategy development:
- BaseStrategy: Abstract base class for all strategies
- StrategySignal: Signal data structure
- StrategyResult: Strategy execution results
- StrategyManager: Manages multiple strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from quantlib.core.utils import Logger


class SignalType(Enum):
    """Signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


class PositionType(Enum):
    """Position types"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class StrategySignal:
    """Strategy signal data structure"""
    timestamp: datetime
    asset: str
    signal_type: SignalType
    strength: float  # Signal strength [0, 1]
    confidence: float  # Signal confidence [0, 1]
    target_position: float  # Target position size
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate signal data"""
        if not 0 <= self.strength <= 1:
            raise ValueError("Signal strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Signal confidence must be between 0 and 1")


@dataclass
class StrategyResult:
    """Strategy execution result"""
    timestamp: datetime
    asset: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    price: float
    position_before: float
    position_after: float
    pnl: float
    cumulative_pnl: float
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, assets: List[str], 
                 lookback_period: int = 252,
                 rebalance_frequency: str = 'daily'):
        self.name = name
        self.assets = assets
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.logger = Logger.get_logger(f"strategy_{name.lower()}")
        
        # Strategy state
        self.positions = {asset: 0.0 for asset in assets}
        self.signals_history = []
        self.results_history = []
        self.last_rebalance = None
        self.is_initialized = False
        
        # Performance tracking
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, 
                        timestamp: datetime) -> List[StrategySignal]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: StrategySignal, 
                              current_position: float,
                              portfolio_value: float) -> float:
        """Calculate position size for a given signal"""
        pass
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data"""
        self.logger.info(f"Initializing strategy {self.name}")
        
        # Validate data
        if len(data) < self.lookback_period:
            self.logger.warning(f"Insufficient data for lookback period: {len(data)} < {self.lookback_period}")
        
        # Strategy-specific initialization
        self._initialize_strategy(data)
        self.is_initialized = True
        
        self.logger.info(f"Strategy {self.name} initialized successfully")
    
    def _initialize_strategy(self, data: pd.DataFrame) -> None:
        """Strategy-specific initialization (override in subclasses)"""
        pass
    
    def update(self, data: pd.DataFrame, timestamp: datetime,
              portfolio_value: float) -> List[StrategyResult]:
        """Update strategy with new market data"""
        if not self.is_initialized:
            raise RuntimeError("Strategy must be initialized before updating")
        
        # Check if rebalancing is needed
        if not self._should_rebalance(timestamp):
            return []
        
        # Generate signals
        signals = self.generate_signals(data, timestamp)
        
        # Execute signals
        results = []
        for signal in signals:
            result = self._execute_signal(signal, portfolio_value, timestamp)
            if result:
                results.append(result)
                self.results_history.append(result)
        
        # Update last rebalance time
        self.last_rebalance = timestamp
        
        return results
    
    def _should_rebalance(self, timestamp: datetime) -> bool:
        """Determine if strategy should rebalance"""
        if self.last_rebalance is None:
            return True
        
        days_since_rebalance = (timestamp - self.last_rebalance).days
        
        if self.rebalance_frequency == 'daily':
            return days_since_rebalance >= 1
        elif self.rebalance_frequency == 'weekly':
            return days_since_rebalance >= 7
        elif self.rebalance_frequency == 'monthly':
            return days_since_rebalance >= 30
        elif self.rebalance_frequency == 'quarterly':
            return days_since_rebalance >= 90
        else:
            return True
    
    def _execute_signal(self, signal: StrategySignal, 
                       portfolio_value: float,
                       timestamp: datetime) -> Optional[StrategyResult]:
        """Execute a trading signal"""
        current_position = self.positions.get(signal.asset, 0.0)
        
        # Calculate target position size
        target_position = self.calculate_position_size(
            signal, current_position, portfolio_value
        )
        
        # Calculate trade quantity
        trade_quantity = target_position - current_position
        
        # Skip if no trade needed
        if abs(trade_quantity) < 1e-6:
            return None
        
        # Determine action
        action = 'buy' if trade_quantity > 0 else 'sell'
        
        # For simulation, use last available price
        # In live trading, this would be the execution price
        price = signal.metadata.get('price', 100.0) if signal.metadata else 100.0
        
        # Calculate P&L (simplified)
        pnl = 0.0  # Would be calculated based on previous position and price change
        self.total_pnl += pnl
        
        # Update position
        self.positions[signal.asset] = target_position
        
        # Update trade statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        
        # Create result
        result = StrategyResult(
            timestamp=timestamp,
            asset=signal.asset,
            action=action,
            quantity=abs(trade_quantity),
            price=price,
            position_before=current_position,
            position_after=target_position,
            pnl=pnl,
            cumulative_pnl=self.total_pnl,
            metadata={
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'signal_type': signal.signal_type.name
            }
        )
        
        self.logger.info(f"Executed {action} {abs(trade_quantity):.4f} {signal.asset} at {price:.4f}")
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        if self.total_trades == 0:
            return {
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        win_rate = self.winning_trades / self.total_trades
        
        # Calculate average win/loss (simplified)
        avg_win = 0.0
        avg_loss = 0.0
        if self.winning_trades > 0 and self.losing_trades > 0:
            # This would be calculated from actual trade results
            avg_win = self.total_pnl / self.winning_trades if self.winning_trades > 0 else 0
            avg_loss = abs(self.total_pnl) / self.losing_trades if self.losing_trades > 0 else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        return {
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.positions = {asset: 0.0 for asset in self.assets}
        self.signals_history = []
        self.results_history = []
        self.last_rebalance = None
        self.is_initialized = False
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self.logger.info(f"Strategy {self.name} reset")


class StrategyManager:
    """Manages multiple strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.logger = Logger.get_logger("strategy_manager")
        
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the manager"""
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Added strategy: {strategy.name}")
        
    def remove_strategy(self, name: str) -> None:
        """Remove a strategy from the manager"""
        if name in self.strategies:
            del self.strategies[name]
            self.logger.info(f"Removed strategy: {name}")
        else:
            self.logger.warning(f"Strategy not found: {name}")
    
    def initialize_all(self, data: pd.DataFrame) -> None:
        """Initialize all strategies"""
        for strategy in self.strategies.values():
            strategy.initialize(data)
    
    def update_all(self, data: pd.DataFrame, timestamp: datetime,
                  portfolio_value: float) -> Dict[str, List[StrategyResult]]:
        """Update all strategies"""
        results = {}
        
        for name, strategy in self.strategies.items():
            try:
                strategy_results = strategy.update(data, timestamp, portfolio_value)
                results[name] = strategy_results
            except Exception as e:
                self.logger.error(f"Error updating strategy {name}: {e}")
                results[name] = []
        
        return results
    
    def get_combined_signals(self, data: pd.DataFrame, 
                           timestamp: datetime) -> List[StrategySignal]:
        """Get combined signals from all strategies"""
        all_signals = []
        
        for strategy in self.strategies.values():
            if strategy.is_initialized:
                try:
                    signals = strategy.generate_signals(data, timestamp)
                    all_signals.extend(signals)
                except Exception as e:
                    self.logger.error(f"Error generating signals for {strategy.name}: {e}")
        
        return all_signals
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary for all strategies"""
        summary_data = []
        
        for name, strategy in self.strategies.items():
            metrics = strategy.get_performance_metrics()
            metrics['strategy_name'] = name
            summary_data.append(metrics)
        
        return pd.DataFrame(summary_data)
    
    def reset_all(self) -> None:
        """Reset all strategies"""
        for strategy in self.strategies.values():
            strategy.reset()
        
        self.logger.info("All strategies reset")