#!/usr/bin/env python3
"""
QuantLib Strategy Development Framework

This module provides a comprehensive framework for developing and implementing
trading strategies with machine learning integration including:
- Base strategy classes and interfaces
- Signal generation and combination
- Machine learning model integration
- Strategy backtesting and evaluation
- Risk management and position sizing
- Multi-asset and multi-timeframe strategies
"""

from .base import (
    BaseStrategy,
    StrategySignal,
    StrategyResult,
    StrategyManager
)

from .signals import (
    SignalGenerator,
    TechnicalSignalGenerator,
    FundamentalSignalGenerator,
    SentimentSignalGenerator,
    MacroSignalGenerator,
    SignalCombiner
)

from .ml import (
    MLModelResult,
    ModelPerformance,
    FeatureEngineer,
    BaseMLModel,
    RandomForestModel,
    LogisticRegressionModel,
    MLSignalGenerator,
    OnlineLearningModel
)

from .risk import (
    RiskLimitType,
    RiskAction,
    RiskLimit,
    RiskMetrics,
    RiskAdjustment,
    BaseRiskManager,
    PositionSizeManager,
    StopLossManager,
    PortfolioRiskManager,
    RiskManagerComposite
)

from .execution import (
    Order, Fill, ExecutionResult, OrderType, OrderStatus, OrderSide,
    MarketDataProvider, SimulatedMarketData,
    TransactionCostModel, SimpleTransactionCostModel,
    BaseExecutionAlgorithm, MarketOrderExecution, TWAPExecution, VWAPExecution,
    ExecutionManager
)

__all__ = [
    # Base Strategy Framework
    'BaseStrategy',
    'StrategySignal',
    'StrategyResult',
    'StrategyManager',
    
    # Signal Generation
    'SignalGenerator',
    'TechnicalSignalGenerator',
    'FundamentalSignalGenerator',
    'SentimentSignalGenerator',
    'MacroSignalGenerator',
    'SignalCombiner',
    
    # ML Integration
    'MLModelResult',
    'ModelPerformance',
    'FeatureEngineer',
    'BaseMLModel',
    'RandomForestModel',
    'LogisticRegressionModel',
    'MLSignalGenerator',
    'OnlineLearningModel',
    
    # Risk Management
    'RiskLimitType',
    'RiskAction',
    'RiskLimit',
    'RiskMetrics',
    'RiskAdjustment',
    'BaseRiskManager',
    'PositionSizeManager',
    'StopLossManager',
    'PortfolioRiskManager',
    'RiskManagerComposite',
    
    # Execution
    'Order', 'Fill', 'ExecutionResult', 'OrderType', 'OrderStatus', 'OrderSide',
    'MarketDataProvider', 'SimulatedMarketData',
    'TransactionCostModel', 'SimpleTransactionCostModel',
    'BaseExecutionAlgorithm', 'MarketOrderExecution', 'TWAPExecution', 'VWAPExecution',
    'ExecutionManager'
]

__version__ = '1.0.0'
__author__ = 'QuantLib Team'