"""Technical Indicators Module

Implements popular technical analysis indicators including:
- Moving averages (SMA, EMA, WMA)
- Momentum indicators (RSI, Stochastic, Williams %R)
- Trend indicators (MACD, ADX)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, CCI)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import warnings

from quantlib.core.base import BaseIndicator
from quantlib.core.utils import Logger


class SimpleMovingAverage(BaseIndicator):
    """Simple Moving Average (SMA) indicator"""
    
    def __init__(self, period: int = 20):
        super().__init__(name="SMA", period=period)
        self.period = period
        self.logger = Logger.get_logger("sma")
        
    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate Simple Moving Average"""
        if len(data) < self.period:
            self.logger.warning(f"Insufficient data for SMA({self.period}): {len(data)} < {self.period}")
            
        sma = data.rolling(window=self.period, min_periods=1).mean()
        return sma
    
    def generate_signals(self, data: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on price crossing SMA"""
        sma = self.calculate(data)
        signals = pd.Series(0, index=price.index)
        
        # Buy when price crosses above SMA
        signals[(price > sma) & (price.shift(1) <= sma.shift(1))] = 1
        # Sell when price crosses below SMA
        signals[(price < sma) & (price.shift(1) >= sma.shift(1))] = -1
        
        return signals


class ExponentialMovingAverage(BaseIndicator):
    """Exponential Moving Average (EMA) indicator"""
    
    def __init__(self, period: int = 20, alpha: Optional[float] = None):
        super().__init__(name="EMA", period=period)
        self.period = period
        self.alpha = alpha or (2.0 / (period + 1))
        self.logger = Logger.get_logger("ema")
        
    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate Exponential Moving Average"""
        ema = data.ewm(alpha=self.alpha, adjust=False).mean()
        return ema
    
    def generate_signals(self, data: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on price crossing EMA"""
        ema = self.calculate(data)
        signals = pd.Series(0, index=price.index)
        
        signals[(price > ema) & (price.shift(1) <= ema.shift(1))] = 1
        signals[(price < ema) & (price.shift(1) >= ema.shift(1))] = -1
        
        return signals


class RelativeStrengthIndex(BaseIndicator):
    """Relative Strength Index (RSI) indicator"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__(name="RSI", period=period)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.logger = Logger.get_logger("rsi")
        
    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=1).mean()
        
        # Use exponential smoothing for more accurate RSI
        avg_gain = gain.ewm(alpha=1/self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on RSI levels"""
        rsi = self.calculate(data)
        signals = pd.Series(0, index=price.index)
        
        # Buy when RSI crosses above oversold level
        signals[(rsi > self.oversold) & (rsi.shift(1) <= self.oversold)] = 1
        # Sell when RSI crosses below overbought level
        signals[(rsi < self.overbought) & (rsi.shift(1) >= self.overbought)] = -1
        
        return signals


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence (MACD) indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(name="MACD", period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.logger = Logger.get_logger("macd")
        
    def calculate(self, data: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD, Signal line, and Histogram"""
        ema_fast = data.ewm(span=self.fast_period).mean()
        ema_slow = data.ewm(span=self.slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def generate_signals(self, data: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on MACD crossovers"""
        macd_data = self.calculate(data)
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        
        signals = pd.Series(0, index=price.index)
        
        # Buy when MACD crosses above signal line
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        # Sell when MACD crosses below signal line
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
        
        return signals


class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(name="BollingerBands", period=period)
        self.period = period
        self.std_dev = std_dev
        self.logger = Logger.get_logger("bollinger")
        
    def calculate(self, data: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=self.period).mean()
        std = data.rolling(window=self.period).std()
        
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bandwidth': (upper_band - lower_band) / sma,
            'percent_b': (data - lower_band) / (upper_band - lower_band)
        }
    
    def generate_signals(self, data: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on Bollinger Band touches"""
        bands = self.calculate(data)
        signals = pd.Series(0, index=price.index)
        
        # Buy when price touches lower band
        signals[price <= bands['lower']] = 1
        # Sell when price touches upper band
        signals[price >= bands['upper']] = -1
        
        return signals


class StochasticOscillator(BaseIndicator):
    """Stochastic Oscillator indicator"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, overbought: float = 80, oversold: float = 20):
        super().__init__(name="Stochastic", period=k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
        self.logger = Logger.get_logger("stochastic")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on Stochastic levels"""
        stoch = self.calculate(high, low, close)
        k_percent = stoch['k_percent']
        d_percent = stoch['d_percent']
        
        signals = pd.Series(0, index=close.index)
        
        # Buy when %K crosses above %D in oversold territory
        oversold_condition = (k_percent < self.oversold) & (d_percent < self.oversold)
        buy_condition = (k_percent > d_percent) & (k_percent.shift(1) <= d_percent.shift(1)) & oversold_condition
        signals[buy_condition] = 1
        
        # Sell when %K crosses below %D in overbought territory
        overbought_condition = (k_percent > self.overbought) & (d_percent > self.overbought)
        sell_condition = (k_percent < d_percent) & (k_percent.shift(1) >= d_percent.shift(1)) & overbought_condition
        signals[sell_condition] = -1
        
        return signals


class AverageTrueRange(BaseIndicator):
    """Average True Range (ATR) indicator"""
    
    def __init__(self, period: int = 14):
        super().__init__(name="ATR", period=period)
        self.period = period
        self.logger = Logger.get_logger("atr")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Average True Range"""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        
        return atr
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """ATR is primarily used for volatility measurement, not signal generation"""
        return pd.Series(0, index=close.index)


class CommodityChannelIndex(BaseIndicator):
    """Commodity Channel Index (CCI) indicator"""
    
    def __init__(self, period: int = 20, constant: float = 0.015):
        super().__init__(name="CCI", period=period)
        self.period = period
        self.constant = constant
        self.logger = Logger.get_logger("cci")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=self.period).mean()
        mean_deviation = typical_price.rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (self.constant * mean_deviation)
        return cci
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on CCI levels"""
        cci = self.calculate(high, low, close)
        signals = pd.Series(0, index=close.index)
        
        # Buy when CCI crosses above -100
        signals[(cci > -100) & (cci.shift(1) <= -100)] = 1
        # Sell when CCI crosses below 100
        signals[(cci < 100) & (cci.shift(1) >= 100)] = -1
        
        return signals


class WilliamsR(BaseIndicator):
    """Williams %R indicator"""
    
    def __init__(self, period: int = 14, overbought: float = -20, oversold: float = -80):
        super().__init__(name="WilliamsR", period=period)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.logger = Logger.get_logger("williams_r")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=self.period).max()
        lowest_low = low.rolling(window=self.period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on Williams %R levels"""
        williams_r = self.calculate(high, low, close)
        signals = pd.Series(0, index=close.index)
        
        # Buy when Williams %R crosses above oversold level
        signals[(williams_r > self.oversold) & (williams_r.shift(1) <= self.oversold)] = 1
        # Sell when Williams %R crosses below overbought level
        signals[(williams_r < self.overbought) & (williams_r.shift(1) >= self.overbought)] = -1
        
        return signals


class OnBalanceVolume(BaseIndicator):
    """On Balance Volume (OBV) indicator"""
    
    def __init__(self):
        super().__init__(name="OBV", period=1)
        self.logger = Logger.get_logger("obv")
        
    def calculate(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume"""
        price_change = close.diff()
        
        obv = pd.Series(0.0, index=close.index)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def generate_signals(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on OBV trend"""
        obv = self.calculate(close, volume)
        obv_sma = obv.rolling(window=20).mean()
        
        signals = pd.Series(0, index=close.index)
        
        # Buy when OBV crosses above its moving average
        signals[(obv > obv_sma) & (obv.shift(1) <= obv_sma.shift(1))] = 1
        # Sell when OBV crosses below its moving average
        signals[(obv < obv_sma) & (obv.shift(1) >= obv_sma.shift(1))] = -1
        
        return signals