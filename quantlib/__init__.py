"""
QuantLib - A Comprehensive Quantitative Analysis Library

A Python library for quantitative finance, providing tools for:
- Market data processing and analysis
- Risk metrics and performance attribution
- Technical analysis and pattern recognition
- Portfolio optimization and allocation
- Strategy development and backtesting
- Live trading integration
"""

__version__ = "0.1.0"
__author__ = "QuantLib Team"
__email__ = "contact@quantlib.com"

# Core imports
from quantlib.core.base import BaseStrategy, BaseIndicator, BasePortfolio
from quantlib.core.data import DataManager, MarketData
from quantlib.core.utils import Logger, Config, DateUtils

# Analytics imports
from quantlib.analytics.risk import VaRCalculator, RiskMetrics
from quantlib.analytics.performance import PerformanceAnalyzer
from quantlib.analytics.statistics import DistributionAnalyzer, HypothesisTests, RegressionAnalysis, TimeSeriesAnalysis, MonteCarloSimulation

__all__ = [
    # Core
    "BaseStrategy",
    "BaseIndicator", 
    "BasePortfolio",
    "DataManager",
    "MarketData",
    "Logger",
    "Config",
    "DateUtils",
    # Analytics
    "VaRCalculator",
    "RiskMetrics",
    "PerformanceAnalyzer",
    "DistributionAnalyzer",
    "HypothesisTests",
    "RegressionAnalysis", 
    "TimeSeriesAnalysis",
    "MonteCarloSimulation",
]