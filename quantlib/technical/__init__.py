"""Technical Analysis Module

Provides technical indicators, pattern recognition, and signal generation tools.

Modules:
- indicators: Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- patterns: Chart pattern recognition (Head & Shoulders, Triangles, etc.)
- signals: Signal generation and filtering
- oscillators: Momentum oscillators and divergence analysis
"""

from quantlib.technical.indicators import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    RelativeStrengthIndex,
    MACD,
    BollingerBands,
    StochasticOscillator,
    AverageTrueRange,
    CommodityChannelIndex,
    WilliamsR,
    OnBalanceVolume
)

from quantlib.technical.patterns import (
    SupportResistanceDetector,
    TrendlineDetector,
    HeadAndShouldersDetector,
    DoubleTopBottomDetector,
    TriangleDetector,
    CandlestickPatterns
)

from quantlib.technical.signals import (
    SignalType,
    TradingSignal,
    SignalGenerator,
    MovingAverageCrossoverSignals,
    RSIMACDSignals,
    BollingerBandSignals,
    MultiSignalCombiner,
    PatternBasedSignals,
    SignalFilter
)

from quantlib.technical.oscillators import (
    RateOfChange,
    MomentumOscillator,
    UltimateOscillator,
    ChaikinOscillator,
    AccumulationDistributionLine,
    VolumeRateOfChange,
    ChaikinVolatility,
    PriceChannelOscillator,
    AroonOscillator,
    MoneyFlowIndex,
    CompositeOscillator
)

__all__ = [
    # Indicators
    "SimpleMovingAverage",
    "ExponentialMovingAverage", 
    "RelativeStrengthIndex",
    "MACD",
    "BollingerBands",
    "StochasticOscillator",
    "AverageTrueRange",
    "CommodityChannelIndex",
    "WilliamsR",
    "OnBalanceVolume",
    
    # Patterns
    "SupportResistanceDetector",
    "TrendlineDetector",
    "HeadAndShouldersDetector",
    "DoubleTopBottomDetector",
    "TriangleDetector",
    "CandlestickPatterns",
    
    # Signals
    "SignalType",
    "TradingSignal",
    "SignalGenerator",
    "MovingAverageCrossoverSignals",
    "RSIMACDSignals",
    "BollingerBandSignals",
    "MultiSignalCombiner",
    "PatternBasedSignals",
    "SignalFilter",
    
    # Oscillators
    "RateOfChange",
    "MomentumOscillator",
    "UltimateOscillator",
    "ChaikinOscillator",
    "AccumulationDistributionLine",
    "VolumeRateOfChange",
    "ChaikinVolatility",
    "PriceChannelOscillator",
    "AroonOscillator",
    "MoneyFlowIndex",
    "CompositeOscillator",
]