"""Main backtesting engine.

Orchestrates the event-driven backtesting process.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from queue import Queue
import time

from .events import Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from .data_handler import DataHandler, HistoricalDataHandler
from .portfolio import BacktestPortfolio
from .broker import ExecutionHandler, SimulatedBroker
from .performance import BacktestResults
from quantlib.core.utils import Logger
from quantlib.strategy.base import BaseStrategy


class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(self, data_handler: DataHandler, portfolio: BacktestPortfolio,
                 execution_handler: ExecutionHandler, strategy: BaseStrategy,
                 heartbeat: float = 0.0):
        """
        Initialize backtesting engine.
        
        Args:
            data_handler: Handles market data
            portfolio: Tracks positions and performance
            execution_handler: Executes orders
            strategy: Trading strategy
            heartbeat: Sleep time between events (for live simulation)
        """
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.strategy = strategy
        self.heartbeat = heartbeat
        self.logger = Logger.get_logger("backtest_engine")
        
        # Event queue
        self.events = Queue()
        
        # Backtest state
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        
    def _generate_trading_instances(self):
        """Generate the trading instances objects from their class types."""
        self.logger.info("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        self.events = Queue()
        
    def _run_backtest(self):
        """Execute the backtest."""
        self.logger.info("Starting backtest...")
        self.start_time = datetime.now()
        
        # Initialize strategy with data
        if hasattr(self.strategy, 'initialize'):
            # Get initial data for strategy initialization
            initial_data = {}
            for symbol in self.data_handler.symbols:
                symbol_data = self.data_handler.data[symbol].copy()
                initial_data[symbol] = symbol_data
            self.strategy.initialize(initial_data)
        
        i = 0
        while True:
            i += 1
            
            # Update the market bars
            if self.data_handler.continue_backtest():
                market_event = self.data_handler.update_bars()
                if market_event:
                    self.events.put(market_event)
            else:
                break
                
            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except:
                    break
                else:
                    if event is not None:
                        self._handle_event(event)
                        
            # Optional heartbeat for live simulation
            if self.heartbeat > 0:
                time.sleep(self.heartbeat)
                
        self.end_time = datetime.now()
        self.logger.info(f"Backtest completed. Processed {i} time periods.")
        
    def _handle_event(self, event: Event):
        """Handle different types of events."""
        if event.type == EventType.MARKET:
            self._handle_market_event(event)
        elif event.type == EventType.SIGNAL:
            self._handle_signal_event(event)
        elif event.type == EventType.ORDER:
            self._handle_order_event(event)
        elif event.type == EventType.FILL:
            self._handle_fill_event(event)
            
    def _handle_market_event(self, event: MarketEvent):
        """Handle market data event."""
        # Update portfolio with new market data
        self.portfolio.update_timeindex(event)
        
        # Generate strategy signals
        signals = self.strategy.generate_signals(event.timestamp)
        
        if signals:
            for signal in signals:
                self.events.put(signal)
                
    def _handle_signal_event(self, event: SignalEvent):
        """Handle signal event from strategy."""
        self.signals += 1
        
        # Generate order from signal
        order = self._signal_to_order(event)
        if order:
            self.events.put(order)
            
    def _handle_order_event(self, event: OrderEvent):
        """Handle order event."""
        self.orders += 1
        
        # Execute order through broker
        fill_event = self.execution_handler.execute_order(event)
        if fill_event:
            self.events.put(fill_event)
            
    def _handle_fill_event(self, event: FillEvent):
        """Handle fill event."""
        self.fills += 1
        
        # Update portfolio with fill
        self.portfolio.update_fill(event)
        
    def _signal_to_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """Convert signal to order using position sizing."""
        symbol = signal.symbol
        signal_type = signal.signal_type
        
        if signal_type not in ['BUY', 'SELL']:
            return None
            
        # Get current position
        current_quantity = self.portfolio.get_position_quantity(symbol)
        
        # Calculate target position size
        if hasattr(self.strategy, 'calculate_position_size'):
            target_size = self.strategy.calculate_position_size(
                symbol, signal, self.portfolio.calculate_portfolio_value()
            )
        else:
            # Default position sizing (simple fixed size)
            portfolio_value = self.portfolio.calculate_portfolio_value()
            position_value = portfolio_value * 0.1  # 10% of portfolio
            
            latest_bar = self.data_handler.get_latest_bar(symbol)
            if not latest_bar:
                return None
                
            price = latest_bar['close']
            target_size = int(position_value / price) if price > 0 else 0
            
            if signal_type == 'SELL':
                target_size = -target_size
                
        # Calculate order quantity
        order_quantity = target_size - current_quantity
        
        if abs(order_quantity) < 1:  # Minimum order size
            return None
            
        # Determine order direction
        direction = 'BUY' if order_quantity > 0 else 'SELL'
        
        # Create order event
        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type='MKT',  # Market order
            quantity=abs(order_quantity),
            direction=direction
        )
        
        return order
        
    def run_backtest(self) -> BacktestResults:
        """Run the complete backtest and return results."""
        self._run_backtest()
        
        # Generate results
        results = BacktestResults(
            portfolio=self.portfolio,
            data_handler=self.data_handler,
            execution_handler=self.execution_handler,
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        # Log summary statistics
        self._log_summary(results)
        
        return results
        
    def _log_summary(self, results: BacktestResults):
        """Log backtest summary."""
        summary = results.get_summary_stats()
        
        self.logger.info("=== BACKTEST SUMMARY ===")
        self.logger.info(f"Total Return: {summary.get('total_return', 0):.2%}")
        self.logger.info(f"Annualized Return: {summary.get('annualized_return', 0):.2%}")
        self.logger.info(f"Volatility: {summary.get('volatility', 0):.2%}")
        self.logger.info(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        self.logger.info(f"Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
        self.logger.info(f"Total Trades: {summary.get('total_trades', 0)}")
        self.logger.info(f"Win Rate: {summary.get('win_rate', 0):.2%}")
        self.logger.info(f"Signals: {self.signals}, Orders: {self.orders}, Fills: {self.fills}")
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            self.logger.info(f"Backtest Duration: {duration}")


class BacktestBuilder:
    """Builder class for creating backtest configurations."""
    
    def __init__(self):
        self.data = None
        self.strategy = None
        self.initial_capital = 100000.0
        self.commission_rate = 0.001
        self.slippage_rate = 0.0005
        self.start_date = None
        self.end_date = None
        
    def set_data(self, data: Dict[str, pd.DataFrame]) -> 'BacktestBuilder':
        """Set market data."""
        self.data = data
        return self
        
    def set_strategy(self, strategy: BaseStrategy) -> 'BacktestBuilder':
        """Set trading strategy."""
        self.strategy = strategy
        return self
        
    def set_initial_capital(self, capital: float) -> 'BacktestBuilder':
        """Set initial capital."""
        self.initial_capital = capital
        return self
        
    def set_commission_rate(self, rate: float) -> 'BacktestBuilder':
        """Set commission rate."""
        self.commission_rate = rate
        return self
        
    def set_slippage_rate(self, rate: float) -> 'BacktestBuilder':
        """Set slippage rate."""
        self.slippage_rate = rate
        return self
        
    def set_date_range(self, start_date: datetime, end_date: datetime) -> 'BacktestBuilder':
        """Set backtest date range."""
        self.start_date = start_date
        self.end_date = end_date
        return self
        
    def build(self) -> BacktestEngine:
        """Build and return backtest engine."""
        if not self.data:
            raise ValueError("Market data is required")
        if not self.strategy:
            raise ValueError("Strategy is required")
            
        # Create components
        data_handler = HistoricalDataHandler(
            self.data, self.start_date, self.end_date
        )
        
        portfolio = BacktestPortfolio(
            data_handler, self.initial_capital, self.commission_rate
        )
        
        broker = SimulatedBroker(
            data_handler, self.commission_rate, self.slippage_rate
        )
        
        # Create engine
        engine = BacktestEngine(
            data_handler, portfolio, broker, self.strategy
        )
        
        return engine