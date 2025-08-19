"""Backtesting module for quantitative trading strategies.

Provides event-driven backtesting engine with portfolio simulation,
performance analysis, and execution modeling.
"""

from .events import Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from .data_handler import DataHandler, HistoricalDataHandler
from .portfolio import Position, BacktestPortfolio
from .broker import ExecutionHandler, SimulatedBroker, AdvancedBroker
from .engine import BacktestEngine, BacktestBuilder
from .performance import BacktestResults

__all__ = [
    'Event', 'EventType', 'MarketEvent', 'SignalEvent', 'OrderEvent', 'FillEvent',
    'DataHandler', 'HistoricalDataHandler',
    'Position', 'BacktestPortfolio',
    'ExecutionHandler', 'SimulatedBroker', 'AdvancedBroker',
    'BacktestEngine', 'BacktestBuilder',
    'BacktestResults'
]