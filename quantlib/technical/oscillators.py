"""Technical Oscillators Module

Implements various oscillators and momentum indicators including:
- Momentum oscillators (ROC, Momentum, Ultimate Oscillator)
- Volatility oscillators (Chaikin Volatility, Price Channel)
- Volume oscillators (Volume Rate of Change, Accumulation/Distribution)
- Custom composite oscillators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import warnings

from quantlib.core.base import BaseIndicator
from quantlib.core.utils import Logger


class RateOfChange(BaseIndicator):
    """Rate of Change (ROC) oscillator"""
    
    def __init__(self, period: int = 12):
        super().__init__(name="ROC", period=period)
        self.period = period
        self.logger = Logger.get_logger("roc")
        
    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate Rate of Change"""
        roc = ((data - data.shift(self.period)) / data.shift(self.period)) * 100
        return roc
    
    def generate_signals(self, data: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on ROC zero-line crossings"""
        roc = self.calculate(data)
        signals = pd.Series(0, index=price.index)
        
        # Buy when ROC crosses above zero
        signals[(roc > 0) & (roc.shift(1) <= 0)] = 1
        # Sell when ROC crosses below zero
        signals[(roc < 0) & (roc.shift(1) >= 0)] = -1
        
        return signals


class MomentumOscillator(BaseIndicator):
    """Momentum oscillator"""
    
    def __init__(self, period: int = 10):
        super().__init__(name="Momentum", period=period)
        self.period = period
        self.logger = Logger.get_logger("momentum")
        
    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate Momentum"""
        momentum = data - data.shift(self.period)
        return momentum
    
    def generate_signals(self, data: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on momentum zero-line crossings"""
        momentum = self.calculate(data)
        signals = pd.Series(0, index=price.index)
        
        # Buy when momentum crosses above zero
        signals[(momentum > 0) & (momentum.shift(1) <= 0)] = 1
        # Sell when momentum crosses below zero
        signals[(momentum < 0) & (momentum.shift(1) >= 0)] = -1
        
        return signals


class UltimateOscillator(BaseIndicator):
    """Ultimate Oscillator"""
    
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28,
                 weight1: float = 4.0, weight2: float = 2.0, weight3: float = 1.0,
                 overbought: float = 70, oversold: float = 30):
        super().__init__(name="UltimateOscillator", period=period3)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.overbought = overbought
        self.oversold = oversold
        self.logger = Logger.get_logger("ultimate_oscillator")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        prev_close = close.shift(1)
        
        # True Low and Buying Pressure
        true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
        buying_pressure = close - true_low
        
        # True Range
        true_high = pd.concat([high, prev_close], axis=1).max(axis=1)
        true_range = true_high - true_low
        
        # Calculate averages for each period
        bp_sum1 = buying_pressure.rolling(window=self.period1).sum()
        tr_sum1 = true_range.rolling(window=self.period1).sum()
        avg1 = bp_sum1 / tr_sum1
        
        bp_sum2 = buying_pressure.rolling(window=self.period2).sum()
        tr_sum2 = true_range.rolling(window=self.period2).sum()
        avg2 = bp_sum2 / tr_sum2
        
        bp_sum3 = buying_pressure.rolling(window=self.period3).sum()
        tr_sum3 = true_range.rolling(window=self.period3).sum()
        avg3 = bp_sum3 / tr_sum3
        
        # Ultimate Oscillator
        uo = 100 * ((self.weight1 * avg1 + self.weight2 * avg2 + self.weight3 * avg3) / 
                    (self.weight1 + self.weight2 + self.weight3))
        
        return uo
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on Ultimate Oscillator levels"""
        uo = self.calculate(high, low, close)
        signals = pd.Series(0, index=close.index)
        
        # Buy when UO crosses above oversold level
        signals[(uo > self.oversold) & (uo.shift(1) <= self.oversold)] = 1
        # Sell when UO crosses below overbought level
        signals[(uo < self.overbought) & (uo.shift(1) >= self.overbought)] = -1
        
        return signals


class ChaikinOscillator(BaseIndicator):
    """Chaikin Oscillator (MACD of Accumulation/Distribution Line)"""
    
    def __init__(self, fast_period: int = 3, slow_period: int = 10):
        super().__init__(name="ChaikinOscillator", period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.logger = Logger.get_logger("chaikin_oscillator")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Chaikin Oscillator"""
        # Accumulation/Distribution Line
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        
        # MACD of A/D Line
        ema_fast = ad_line.ewm(span=self.fast_period).mean()
        ema_slow = ad_line.ewm(span=self.slow_period).mean()
        
        chaikin_osc = ema_fast - ema_slow
        return chaikin_osc
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on Chaikin Oscillator zero-line crossings"""
        chaikin_osc = self.calculate(high, low, close, volume)
        signals = pd.Series(0, index=close.index)
        
        # Buy when oscillator crosses above zero
        signals[(chaikin_osc > 0) & (chaikin_osc.shift(1) <= 0)] = 1
        # Sell when oscillator crosses below zero
        signals[(chaikin_osc < 0) & (chaikin_osc.shift(1) >= 0)] = -1
        
        return signals


class AccumulationDistributionLine(BaseIndicator):
    """Accumulation/Distribution Line"""
    
    def __init__(self):
        super().__init__(name="ADL", period=1)
        self.logger = Logger.get_logger("ad_line")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        # Close Location Value
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero when high == low
        
        # A/D Line
        ad_line = (clv * volume).cumsum()
        return ad_line
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on A/D Line trend"""
        ad_line = self.calculate(high, low, close, volume)
        ad_sma = ad_line.rolling(window=20).mean()
        
        signals = pd.Series(0, index=close.index)
        
        # Buy when A/D Line crosses above its moving average
        signals[(ad_line > ad_sma) & (ad_line.shift(1) <= ad_sma.shift(1))] = 1
        # Sell when A/D Line crosses below its moving average
        signals[(ad_line < ad_sma) & (ad_line.shift(1) >= ad_sma.shift(1))] = -1
        
        return signals


class VolumeRateOfChange(BaseIndicator):
    """Volume Rate of Change"""
    
    def __init__(self, period: int = 12):
        super().__init__(name="VolumeROC", period=period)
        self.period = period
        self.logger = Logger.get_logger("volume_roc")
        
    def calculate(self, volume: pd.Series) -> pd.Series:
        """Calculate Volume Rate of Change"""
        volume_roc = ((volume - volume.shift(self.period)) / volume.shift(self.period)) * 100
        return volume_roc
    
    def generate_signals(self, volume: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on volume expansion"""
        volume_roc = self.calculate(volume)
        signals = pd.Series(0, index=price.index)
        
        # Buy when volume expansion is significant (>50%)
        signals[volume_roc > 50] = 1
        # Sell when volume contraction is significant (<-30%)
        signals[volume_roc < -30] = -1
        
        return signals


class ChaikinVolatility(BaseIndicator):
    """Chaikin Volatility"""
    
    def __init__(self, period: int = 10, roc_period: int = 10):
        super().__init__(name="ChaikinVolatility", period=period)
        self.period = period
        self.roc_period = roc_period
        self.logger = Logger.get_logger("chaikin_volatility")
        
    def calculate(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Chaikin Volatility"""
        # High-Low spread
        hl_spread = high - low
        
        # Exponential moving average of the spread
        ema_spread = hl_spread.ewm(span=self.period).mean()
        
        # Rate of change of the EMA
        chaikin_vol = ((ema_spread - ema_spread.shift(self.roc_period)) / 
                      ema_spread.shift(self.roc_period)) * 100
        
        return chaikin_vol
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate signals based on volatility changes"""
        chaikin_vol = self.calculate(high, low)
        signals = pd.Series(0, index=close.index)
        
        # High volatility periods - potential reversal signals
        # This is more of a filter than a direct signal generator
        return signals


class PriceChannelOscillator(BaseIndicator):
    """Price Channel Oscillator"""
    
    def __init__(self, period: int = 20):
        super().__init__(name="PriceChannel", period=period)
        self.period = period
        self.logger = Logger.get_logger("price_channel")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Price Channel Oscillator"""
        highest_high = high.rolling(window=self.period).max()
        lowest_low = low.rolling(window=self.period).min()
        
        # Price position within channel (0-100)
        price_position = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        return {
            'highest_high': highest_high,
            'lowest_low': lowest_low,
            'price_position': price_position,
            'channel_width': highest_high - lowest_low
        }
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on price channel position"""
        channel_data = self.calculate(high, low, close)
        price_position = channel_data['price_position']
        
        signals = pd.Series(0, index=close.index)
        
        # Buy when price is near bottom of channel
        signals[price_position < 20] = 1
        # Sell when price is near top of channel
        signals[price_position > 80] = -1
        
        return signals


class DetrендedPriceOscillator(BaseIndicator):
    """Detrended Price Oscillator (DPO)"""
    
    def __init__(self, period: int = 20):
        super().__init__(name="DPO", period=period)
        self.period = period
        self.logger = Logger.get_logger("dpo")
        
    def calculate(self, close: pd.Series) -> pd.Series:
        """Calculate Detrended Price Oscillator"""
        # Simple moving average
        sma = close.rolling(window=self.period).mean()
        
        # Shift the SMA back by (period/2 + 1) periods
        shift_periods = int(self.period / 2) + 1
        shifted_sma = sma.shift(shift_periods)
        
        # DPO = Close - Shifted SMA
        dpo = close - shifted_sma
        
        return dpo
    
    def generate_signals(self, close: pd.Series, price: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on DPO zero-line crossings"""
        dpo = self.calculate(close)
        signals = pd.Series(0, index=price.index)
        
        # Buy when DPO crosses above zero
        signals[(dpo > 0) & (dpo.shift(1) <= 0)] = 1
        # Sell when DPO crosses below zero
        signals[(dpo < 0) & (dpo.shift(1) >= 0)] = -1
        
        return signals


class AroonOscillator(BaseIndicator):
    """Aroon Oscillator"""
    
    def __init__(self, period: int = 25):
        super().__init__(name="Aroon", period=period)
        self.period = period
        self.logger = Logger.get_logger("aroon")
        
    def calculate(self, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Aroon Up, Aroon Down, and Aroon Oscillator"""
        aroon_up = pd.Series(index=high.index, dtype=float)
        aroon_down = pd.Series(index=low.index, dtype=float)
        
        for i in range(self.period, len(high)):
            # Find the number of periods since the highest high
            high_slice = high.iloc[i-self.period+1:i+1]
            high_max_idx = high_slice.idxmax()
            if isinstance(high_max_idx, pd.Timestamp):
                periods_since_high = len(high_slice) - 1 - high_slice.index.get_loc(high_max_idx)
            else:
                periods_since_high = self.period - 1 - high_max_idx
            
            # Find the number of periods since the lowest low
            low_slice = low.iloc[i-self.period+1:i+1]
            low_min_idx = low_slice.idxmin()
            if isinstance(low_min_idx, pd.Timestamp):
                periods_since_low = len(low_slice) - 1 - low_slice.index.get_loc(low_min_idx)
            else:
                periods_since_low = self.period - 1 - low_min_idx
            
            # Calculate Aroon Up and Aroon Down
            aroon_up.iloc[i] = ((self.period - periods_since_high) / self.period) * 100
            aroon_down.iloc[i] = ((self.period - periods_since_low) / self.period) * 100
        
        # Aroon Oscillator
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on Aroon Oscillator"""
        aroon_data = self.calculate(high, low)
        aroon_osc = aroon_data['aroon_oscillator']
        
        signals = pd.Series(0, index=close.index)
        
        # Buy when Aroon Oscillator crosses above zero
        signals[(aroon_osc > 0) & (aroon_osc.shift(1) <= 0)] = 1
        # Sell when Aroon Oscillator crosses below zero
        signals[(aroon_osc < 0) & (aroon_osc.shift(1) >= 0)] = -1
        
        return signals


class MoneyFlowIndex(BaseIndicator):
    """Money Flow Index (MFI)"""
    
    def __init__(self, period: int = 14, overbought: float = 80, oversold: float = 20):
        super().__init__(name="MFI", period=period)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.logger = Logger.get_logger("mfi")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Money Flow Index"""
        # Typical Price
        typical_price = (high + low + close) / 3
        
        # Raw Money Flow
        raw_money_flow = typical_price * volume
        
        # Positive and Negative Money Flow
        positive_flow = pd.Series(0.0, index=close.index)
        negative_flow = pd.Series(0.0, index=close.index)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = raw_money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = raw_money_flow.iloc[i]
        
        # Money Flow Ratio
        positive_mf = positive_flow.rolling(window=self.period).sum()
        negative_mf = negative_flow.rolling(window=self.period).sum()
        
        money_flow_ratio = positive_mf / negative_mf
        money_flow_ratio = money_flow_ratio.replace([np.inf, -np.inf], 100)  # Handle division by zero
        
        # Money Flow Index
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on MFI levels"""
        mfi = self.calculate(high, low, close, volume)
        signals = pd.Series(0, index=close.index)
        
        # Buy when MFI crosses above oversold level
        signals[(mfi > self.oversold) & (mfi.shift(1) <= self.oversold)] = 1
        # Sell when MFI crosses below overbought level
        signals[(mfi < self.overbought) & (mfi.shift(1) >= self.overbought)] = -1
        
        return signals


class CompositeOscillator(BaseIndicator):
    """Composite oscillator combining multiple indicators"""
    
    def __init__(self, rsi_period: int = 14, stoch_k: int = 14, stoch_d: int = 3, 
                 williams_period: int = 14):
        super().__init__(name="CompositeOscillator", period=rsi_period)
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.williams_period = williams_period
        self.logger = Logger.get_logger("composite_oscillator")
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate composite oscillator from multiple indicators"""
        # RSI component
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic component
        lowest_low = low.rolling(window=self.stoch_k).min()
        highest_high = high.rolling(window=self.stoch_k).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=self.stoch_d).mean()
        
        # Williams %R component
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        # Composite score (average of normalized indicators)
        composite = (rsi + k_percent + (williams_r + 100)) / 3
        
        return {
            'rsi': rsi,
            'stoch_k': k_percent,
            'stoch_d': d_percent,
            'williams_r': williams_r,
            'composite': composite
        }
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate buy/sell signals based on composite oscillator"""
        oscillator_data = self.calculate(high, low, close)
        composite = oscillator_data['composite']
        
        signals = pd.Series(0, index=close.index)
        
        # Buy when composite oscillator is oversold (<30)
        signals[(composite > 30) & (composite.shift(1) <= 30)] = 1
        # Sell when composite oscillator is overbought (>70)
        signals[(composite < 70) & (composite.shift(1) >= 70)] = -1
        
        return signals