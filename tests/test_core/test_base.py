"""
Tests for core base classes.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quantlib.core.base import BaseStrategy, BaseIndicator, BasePortfolio


class TestStrategy(BaseStrategy):
    """
    Test implementation of BaseStrategy.
    """
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Simple moving average crossover strategy
        short_ma = data['close'].rolling(10).mean()
        long_ma = data['close'].rolling(20).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        
        return signals
    
    def calculate_position_size(self, signal: float, current_price: float, 
                              portfolio_value: float) -> float:
        # Simple position sizing: 10% of portfolio per signal
        return (portfolio_value * 0.1 * signal) / current_price


class TestIndicator(BaseIndicator):
    """
    Test implementation of BaseIndicator.
    """
    
    def calculate(self, data: pd.Series) -> pd.Series:
        return data.rolling(self.period).mean()


class TestPortfolio(BasePortfolio):
    """
    Test implementation of BasePortfolio.
    """
    
    def add_position(self, symbol: str, quantity: float, price: float, 
                   timestamp: datetime) -> None:
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        
        current_qty = self.positions[symbol]['quantity']
        current_avg = self.positions[symbol]['avg_price']
        
        new_qty = current_qty + quantity
        new_avg = ((current_qty * current_avg) + (quantity * price)) / new_qty if new_qty != 0 else 0
        
        self.positions[symbol] = {'quantity': new_qty, 'avg_price': new_avg}
        
        self.trades.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'type': 'buy' if quantity > 0 else 'sell'
        })
    
    def close_position(self, symbol: str, quantity: float, price: float,
                      timestamp: datetime) -> None:
        if symbol in self.positions:
            self.positions[symbol]['quantity'] -= quantity
            
            if self.positions[symbol]['quantity'] <= 0:
                del self.positions[symbol]
        
        self.trades.append({
            'symbol': symbol,
            'quantity': -quantity,
            'price': price,
            'timestamp': timestamp,
            'type': 'sell'
        })
    
    def calculate_portfolio_value(self, current_prices: dict) -> float:
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['quantity'] * current_prices[symbol]
        
        return total_value


class TestBaseClasses:
    """
    Test cases for base classes.
    """
    
    def test_base_strategy(self, sample_price_data):
        """
        Test BaseStrategy implementation.
        """
        strategy = TestStrategy("test_strategy")
        
        # Test signal generation
        signals = strategy.generate_signals(sample_price_data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_price_data)
        assert signals.isin([-1, 0, 1]).all()
        
        # Test position sizing
        position_size = strategy.calculate_position_size(1.0, 100.0, 10000.0)
        assert isinstance(position_size, float)
        assert position_size > 0
        
        # Test performance update
        returns = pd.Series([0.01, -0.005, 0.02, -0.01], 
                           index=pd.date_range('2023-01-01', periods=4))
        strategy.update_performance(returns)
        
        assert 'total_return' in strategy.performance_metrics
        assert 'sharpe_ratio' in strategy.performance_metrics
        assert 'max_drawdown' in strategy.performance_metrics
        assert 'win_rate' in strategy.performance_metrics
    
    def test_base_indicator(self, sample_price_data):
        """
        Test BaseIndicator implementation.
        """
        indicator = TestIndicator("test_indicator", period=10)
        
        # Test calculation
        result = indicator.calculate(sample_price_data['close'])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # Test validation
        short_data = sample_price_data['close'].head(5)
        assert not indicator.validate_data(short_data)
        assert indicator.validate_data(sample_price_data['close'])
    
    def test_base_portfolio(self):
        """
        Test BasePortfolio implementation.
        """
        portfolio = TestPortfolio(10000.0, "test_portfolio")
        
        # Test adding positions
        portfolio.add_position("AAPL", 10, 150.0, datetime.now())
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"]['quantity'] == 10
        assert portfolio.positions["AAPL"]['avg_price'] == 150.0
        
        # Test portfolio value calculation
        current_prices = {"AAPL": 155.0}
        portfolio_value = portfolio.calculate_portfolio_value(current_prices)
        expected_value = 10000.0 + (10 * 155.0)  # Cash + position value
        assert portfolio_value == expected_value
        
        # Test closing position
        portfolio.close_position("AAPL", 5, 160.0, datetime.now())
        assert portfolio.positions["AAPL"]['quantity'] == 5
        
        # Test performance summary
        portfolio.performance_history = [
            {'portfolio_value': 10000, 'return': 0.0},
            {'portfolio_value': 10100, 'return': 0.01},
            {'portfolio_value': 10050, 'return': -0.005}
        ]
        
        summary = portfolio.get_performance_summary()
        assert 'total_return' in summary
        assert 'sharpe_ratio' in summary
        assert 'max_drawdown' in summary
        assert 'total_trades' in summary
        assert 'win_rate' in summary