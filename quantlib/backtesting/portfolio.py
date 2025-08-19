"""Portfolio management for backtesting.

Tracks positions, cash, and performance metrics during backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .events import FillEvent, MarketEvent
from .data_handler import DataHandler
from quantlib.core.utils import Logger


class Position:
    """Represents a position in a single asset."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0
        self.avg_price = 0.0
        self.total_cost = 0.0
        self.realized_pnl = 0.0
        
    def update_position(self, quantity: int, price: float, commission: float = 0.0):
        """Update position with new fill."""
        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.avg_price = price
            self.total_cost = abs(quantity) * price + commission
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Adding to existing position
            total_quantity = self.quantity + quantity
            total_value = self.total_cost + abs(quantity) * price + commission
            self.avg_price = total_value / abs(total_quantity) if total_quantity != 0 else 0
            self.quantity = total_quantity
            self.total_cost = total_value
        else:
            # Reducing or closing position
            if abs(quantity) >= abs(self.quantity):
                # Closing or reversing position
                closing_quantity = self.quantity
                self.realized_pnl += closing_quantity * (price - self.avg_price) - commission
                
                remaining_quantity = quantity + self.quantity
                if remaining_quantity != 0:
                    # Reversing position
                    self.quantity = remaining_quantity
                    self.avg_price = price
                    self.total_cost = abs(remaining_quantity) * price + commission
                else:
                    # Closing position
                    self.quantity = 0
                    self.avg_price = 0.0
                    self.total_cost = 0.0
            else:
                # Partially reducing position
                self.realized_pnl += quantity * (price - self.avg_price) - commission
                self.quantity += quantity
                self.total_cost -= abs(quantity) * self.avg_price
                
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.quantity == 0:
            return 0.0
        return self.quantity * (current_price - self.avg_price)
        
    def get_market_value(self, current_price: float) -> float:
        """Get current market value of position."""
        return abs(self.quantity) * current_price
        
    def __str__(self) -> str:
        return f"Position({self.symbol}: {self.quantity} @ {self.avg_price:.2f}, realized_pnl={self.realized_pnl:.2f})"


class BacktestPortfolio:
    """Portfolio for backtesting with position tracking and performance calculation."""
    
    def __init__(self, data_handler: DataHandler, initial_capital: float = 100000.0,
                 commission_rate: float = 0.001):
        """
        Initialize backtest portfolio.
        
        Args:
            data_handler: Data handler for market data
            initial_capital: Starting capital
            commission_rate: Commission rate (as fraction of trade value)
        """
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.logger = Logger.get_logger("portfolio")
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # symbol -> Position
        self.total_commission = 0.0
        
        # Performance tracking
        self.equity_curve = []
        self.returns = []
        self.trades = []
        
        # Current market data
        self.current_datetime = None
        self.current_prices = {}
        
    def update_timeindex(self, event: MarketEvent):
        """Update portfolio with new market data."""
        self.current_datetime = event.timestamp
        
        # Update current prices
        for symbol in event.symbols:
            latest_bar = self.data_handler.get_latest_bar(symbol)
            if latest_bar:
                self.current_prices[symbol] = latest_bar['close']
        
        # Calculate and record portfolio value
        portfolio_value = self.calculate_portfolio_value()
        self.equity_curve.append({
            'datetime': self.current_datetime,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash
        })
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            current_value = portfolio_value
            if prev_value > 0:
                period_return = (current_value - prev_value) / prev_value
                self.returns.append(period_return)
    
    def update_fill(self, event: FillEvent):
        """Update portfolio with fill event."""
        symbol = event.symbol
        quantity = event.quantity if event.direction == 'BUY' else -event.quantity
        price = event.fill_price
        commission = event.commission or self.calculate_commission(event)
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
            
        self.positions[symbol].update_position(quantity, price, commission)
        
        # Update cash
        trade_value = abs(quantity) * price
        if event.direction == 'BUY':
            self.cash -= trade_value + commission
        else:
            self.cash += trade_value - commission
            
        self.total_commission += commission
        
        # Record trade
        trade_record = {
            'datetime': event.timestamp,
            'symbol': symbol,
            'direction': event.direction,
            'quantity': abs(quantity),
            'price': price,
            'commission': commission,
            'trade_value': trade_value
        }
        self.trades.append(trade_record)
        
        self.logger.info(f"Fill executed: {event}")
        
    def calculate_commission(self, event: FillEvent) -> float:
        """Calculate commission for a fill event."""
        trade_value = abs(event.quantity) * event.fill_price
        return trade_value * self.commission_rate
        
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in self.current_prices:
                market_value = position.get_market_value(self.current_prices[symbol])
                total_value += market_value
                
        return total_value
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
        
    def get_position_quantity(self, symbol: str) -> int:
        """Get position quantity for a symbol."""
        position = self.get_position(symbol)
        return position.quantity if position else 0
        
    def get_unrealized_pnl(self, symbol: str = None) -> float:
        """Get unrealized P&L for a symbol or total portfolio."""
        if symbol:
            position = self.get_position(symbol)
            if position and symbol in self.current_prices:
                return position.get_unrealized_pnl(self.current_prices[symbol])
            return 0.0
        else:
            total_unrealized = 0.0
            for symbol, position in self.positions.items():
                if position.quantity != 0 and symbol in self.current_prices:
                    total_unrealized += position.get_unrealized_pnl(self.current_prices[symbol])
            return total_unrealized
            
    def get_realized_pnl(self, symbol: str = None) -> float:
        """Get realized P&L for a symbol or total portfolio."""
        if symbol:
            position = self.get_position(symbol)
            return position.realized_pnl if position else 0.0
        else:
            return sum(pos.realized_pnl for pos in self.positions.values())
            
    def get_total_pnl(self, symbol: str = None) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.get_realized_pnl(symbol) + self.get_unrealized_pnl(symbol)
        
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.equity_curve)
        df.set_index('datetime', inplace=True)
        return df
        
    def get_returns_series(self) -> pd.Series:
        """Get returns as pandas Series."""
        if not self.returns or not self.equity_curve:
            return pd.Series()
            
        dates = [entry['datetime'] for entry in self.equity_curve[1:]]  # Skip first entry
        return pd.Series(self.returns, index=dates)
        
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.trades)
        df.set_index('datetime', inplace=True)
        return df
        
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics."""
        current_value = self.calculate_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'cash': self.cash,
            'total_return': total_return,
            'total_pnl': self.get_total_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'total_commission': self.total_commission,
            'num_trades': len(self.trades),
            'num_positions': len([p for p in self.positions.values() if p.quantity != 0])
        }
        
        # Add position details
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                summary[f'{symbol}_quantity'] = position.quantity
                summary[f'{symbol}_avg_price'] = position.avg_price
                if symbol in self.current_prices:
                    summary[f'{symbol}_current_price'] = self.current_prices[symbol]
                    summary[f'{symbol}_unrealized_pnl'] = position.get_unrealized_pnl(self.current_prices[symbol])
                    
        return summary