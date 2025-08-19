"""Trading Signals Module

Combines technical indicators and pattern recognition to generate
actionable trading signals with confidence scoring and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum
import warnings

from quantlib.core.base import BaseIndicator
from quantlib.core.utils import Logger
from quantlib.technical.indicators import (
    SimpleMovingAverage, ExponentialMovingAverage, RelativeStrengthIndex,
    MACD, BollingerBands, StochasticOscillator
)
from quantlib.technical.patterns import (
    SupportResistanceDetector, TrendlineDetector, HeadAndShouldersDetector,
    DoubleTopBottomDetector, CandlestickPatterns
)


class SignalType(Enum):
    """Enumeration of signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


class SignalStrength(Enum):
    """Enumeration of signal strength levels"""
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


class TradingSignal:
    """Represents a trading signal with metadata"""
    
    def __init__(self, signal_type: SignalType, confidence: float, 
                 timestamp: pd.Timestamp, price: float, 
                 indicators: Dict = None, patterns: Dict = None,
                 stop_loss: float = None, take_profit: float = None):
        self.signal_type = signal_type
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
        self.timestamp = timestamp
        self.price = price
        self.indicators = indicators or {}
        self.patterns = patterns or {}
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
    def __repr__(self):
        return f"TradingSignal({self.signal_type.name}, {self.confidence:.2f}, {self.timestamp})"
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return {
            'signal_type': self.signal_type.name,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'price': self.price,
            'indicators': self.indicators,
            'patterns': self.patterns,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class SignalGenerator(ABC):
    """Base class for signal generation strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger.get_logger(f"signal_{name.lower()}")
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from market data"""
        pass
    
    def _calculate_confidence(self, indicators: Dict, patterns: Dict) -> float:
        """Calculate signal confidence based on indicators and patterns"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on indicator agreement
        indicator_signals = [v for v in indicators.values() if isinstance(v, (int, float))]
        if indicator_signals:
            agreement = abs(np.mean(indicator_signals))
            confidence += agreement * 0.3
        
        # Adjust based on pattern strength
        pattern_strengths = [p.get('confidence', 0) for p in patterns.values() if isinstance(p, dict)]
        if pattern_strengths:
            confidence += np.mean(pattern_strengths) * 0.2
        
        return min(1.0, confidence)


class MovingAverageCrossoverSignals(SignalGenerator):
    """Generate signals based on moving average crossovers"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20, 
                 signal_period: int = 9, use_ema: bool = True):
        super().__init__("ma_crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.use_ema = use_ema
        
        if use_ema:
            self.fast_ma = ExponentialMovingAverage(fast_period)
            self.slow_ma = ExponentialMovingAverage(slow_period)
        else:
            self.fast_ma = SimpleMovingAverage(fast_period)
            self.slow_ma = SimpleMovingAverage(slow_period)
            
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on MA crossovers"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        fast_ma = self.fast_ma.calculate(close)
        slow_ma = self.slow_ma.calculate(close)
        
        signals = []
        
        for i in range(1, len(data)):
            current_fast = fast_ma.iloc[i]
            current_slow = slow_ma.iloc[i]
            prev_fast = fast_ma.iloc[i-1]
            prev_slow = slow_ma.iloc[i-1]
            
            # Golden cross (bullish)
            if current_fast > current_slow and prev_fast <= prev_slow:
                confidence = self._calculate_crossover_confidence(fast_ma, slow_ma, i)
                signals.append(TradingSignal(
                    SignalType.BUY,
                    confidence,
                    data.index[i],
                    close.iloc[i],
                    indicators={'fast_ma': current_fast, 'slow_ma': current_slow}
                ))
            
            # Death cross (bearish)
            elif current_fast < current_slow and prev_fast >= prev_slow:
                confidence = self._calculate_crossover_confidence(fast_ma, slow_ma, i)
                signals.append(TradingSignal(
                    SignalType.SELL,
                    confidence,
                    data.index[i],
                    close.iloc[i],
                    indicators={'fast_ma': current_fast, 'slow_ma': current_slow}
                ))
        
        return signals
    
    def _calculate_crossover_confidence(self, fast_ma: pd.Series, slow_ma: pd.Series, index: int) -> float:
        """Calculate confidence based on MA separation and trend strength"""
        separation = abs(fast_ma.iloc[index] - slow_ma.iloc[index]) / slow_ma.iloc[index]
        
        # Higher separation = higher confidence
        confidence = min(0.8, separation * 10)
        
        # Add trend consistency bonus
        if index >= 5:
            recent_trend = np.mean(np.diff(fast_ma.iloc[index-5:index+1]))
            trend_strength = min(0.2, abs(recent_trend) * 100)
            confidence += trend_strength
        
        return min(1.0, confidence + 0.3)  # Base confidence of 0.3


class RSIMACDSignals(SignalGenerator):
    """Generate signals combining RSI and MACD indicators"""
    
    def __init__(self, rsi_period: int = 14, rsi_overbought: float = 70, rsi_oversold: float = 30,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        super().__init__("rsi_macd")
        self.rsi = RelativeStrengthIndex(rsi_period, rsi_overbought, rsi_oversold)
        self.macd = MACD(macd_fast, macd_slow, macd_signal)
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on RSI and MACD combination"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        rsi = self.rsi.calculate(close)
        macd_data = self.macd.calculate(close)
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        histogram = macd_data['histogram']
        
        signals = []
        
        for i in range(1, len(data)):
            current_rsi = rsi.iloc[i]
            current_macd = macd_line.iloc[i]
            current_signal = signal_line.iloc[i]
            prev_macd = macd_line.iloc[i-1]
            prev_signal = signal_line.iloc[i-1]
            
            indicators = {
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'macd_histogram': histogram.iloc[i]
            }
            
            # Bullish signal: RSI oversold + MACD bullish crossover
            if (current_rsi < self.rsi_oversold and 
                current_macd > current_signal and prev_macd <= prev_signal):
                
                confidence = self._calculate_rsi_macd_confidence(current_rsi, current_macd, current_signal, True)
                signals.append(TradingSignal(
                    SignalType.BUY,
                    confidence,
                    data.index[i],
                    close.iloc[i],
                    indicators=indicators
                ))
            
            # Bearish signal: RSI overbought + MACD bearish crossover
            elif (current_rsi > self.rsi_overbought and 
                  current_macd < current_signal and prev_macd >= prev_signal):
                
                confidence = self._calculate_rsi_macd_confidence(current_rsi, current_macd, current_signal, False)
                signals.append(TradingSignal(
                    SignalType.SELL,
                    confidence,
                    data.index[i],
                    close.iloc[i],
                    indicators=indicators
                ))
        
        return signals
    
    def _calculate_rsi_macd_confidence(self, rsi: float, macd: float, signal: float, is_bullish: bool) -> float:
        """Calculate confidence for RSI+MACD signals"""
        confidence = 0.4  # Base confidence
        
        if is_bullish:
            # More oversold = higher confidence
            rsi_strength = max(0, (self.rsi_oversold - rsi) / self.rsi_oversold)
            confidence += rsi_strength * 0.3
        else:
            # More overbought = higher confidence
            rsi_strength = max(0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
            confidence += rsi_strength * 0.3
        
        # MACD separation strength
        macd_separation = abs(macd - signal) / max(abs(macd), abs(signal), 0.001)
        confidence += min(0.3, macd_separation)
        
        return min(1.0, confidence)


class BollingerBandSignals(SignalGenerator):
    """Generate signals based on Bollinger Bands"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, rsi_period: int = 14):
        super().__init__("bollinger_bands")
        self.bb = BollingerBands(period, std_dev)
        self.rsi = RelativeStrengthIndex(rsi_period)
        
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on Bollinger Band touches with RSI confirmation"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        bb_data = self.bb.calculate(close)
        rsi = self.rsi.calculate(close)
        
        signals = []
        
        for i in range(1, len(data)):
            current_price = close.iloc[i]
            prev_price = close.iloc[i-1]
            
            upper_band = bb_data['upper'].iloc[i]
            lower_band = bb_data['lower'].iloc[i]
            middle_band = bb_data['middle'].iloc[i]
            percent_b = bb_data['percent_b'].iloc[i]
            bandwidth = bb_data['bandwidth'].iloc[i]
            
            current_rsi = rsi.iloc[i]
            
            indicators = {
                'bb_upper': upper_band,
                'bb_lower': lower_band,
                'bb_middle': middle_band,
                'bb_percent_b': percent_b,
                'bb_bandwidth': bandwidth,
                'rsi': current_rsi
            }
            
            # Bullish signal: Price bounces off lower band + RSI oversold
            if (prev_price <= lower_band and current_price > lower_band and 
                current_rsi < 30):
                
                confidence = self._calculate_bb_confidence(percent_b, bandwidth, current_rsi, True)
                signals.append(TradingSignal(
                    SignalType.BUY,
                    confidence,
                    data.index[i],
                    current_price,
                    indicators=indicators,
                    stop_loss=lower_band * 0.98,
                    take_profit=middle_band
                ))
            
            # Bearish signal: Price bounces off upper band + RSI overbought
            elif (prev_price >= upper_band and current_price < upper_band and 
                  current_rsi > 70):
                
                confidence = self._calculate_bb_confidence(percent_b, bandwidth, current_rsi, False)
                signals.append(TradingSignal(
                    SignalType.SELL,
                    confidence,
                    data.index[i],
                    current_price,
                    indicators=indicators,
                    stop_loss=upper_band * 1.02,
                    take_profit=middle_band
                ))
        
        return signals
    
    def _calculate_bb_confidence(self, percent_b: float, bandwidth: float, rsi: float, is_bullish: bool) -> float:
        """Calculate confidence for Bollinger Band signals"""
        confidence = 0.4  # Base confidence
        
        # Extreme %B values increase confidence
        if is_bullish:
            confidence += max(0, (0.2 - percent_b)) * 2  # Below 0.2 is strong
            confidence += max(0, (30 - rsi) / 30) * 0.3  # RSI oversold
        else:
            confidence += max(0, (percent_b - 0.8)) * 2  # Above 0.8 is strong
            confidence += max(0, (rsi - 70) / 30) * 0.3  # RSI overbought
        
        # Higher bandwidth = more reliable signals
        confidence += min(0.2, bandwidth * 5)
        
        return min(1.0, confidence)


class MultiSignalCombiner(SignalGenerator):
    """Combine multiple signal generators with weighted voting"""
    
    def __init__(self, generators: List[Tuple[SignalGenerator, float]]):
        super().__init__("multi_signal")
        self.generators = generators  # List of (generator, weight) tuples
        self.total_weight = sum(weight for _, weight in generators)
        
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate combined signals from multiple generators"""
        all_signals = {}
        
        # Collect signals from all generators
        for generator, weight in self.generators:
            try:
                signals = generator.generate_signals(data)
                for signal in signals:
                    timestamp = signal.timestamp
                    if timestamp not in all_signals:
                        all_signals[timestamp] = []
                    all_signals[timestamp].append((signal, weight))
            except Exception as e:
                self.logger.warning(f"Error in generator {generator.name}: {e}")
        
        # Combine signals at each timestamp
        combined_signals = []
        for timestamp, signal_list in all_signals.items():
            combined_signal = self._combine_signals_at_timestamp(signal_list, timestamp, data)
            if combined_signal:
                combined_signals.append(combined_signal)
        
        return sorted(combined_signals, key=lambda x: x.timestamp)
    
    def _combine_signals_at_timestamp(self, signal_list: List[Tuple[TradingSignal, float]], 
                                    timestamp: pd.Timestamp, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Combine multiple signals at the same timestamp"""
        if not signal_list:
            return None
        
        # Calculate weighted vote
        weighted_vote = 0
        total_confidence = 0
        combined_indicators = {}
        combined_patterns = {}
        
        for signal, weight in signal_list:
            vote = signal.signal_type.value * signal.confidence * weight
            weighted_vote += vote
            total_confidence += signal.confidence * weight
            
            # Merge indicators and patterns
            combined_indicators.update(signal.indicators)
            combined_patterns.update(signal.patterns)
        
        # Normalize by total weight
        weighted_vote /= self.total_weight
        total_confidence /= self.total_weight
        
        # Determine final signal type
        if weighted_vote > 0.5:
            signal_type = SignalType.STRONG_BUY if weighted_vote > 1.0 else SignalType.BUY
        elif weighted_vote < -0.5:
            signal_type = SignalType.STRONG_SELL if weighted_vote < -1.0 else SignalType.SELL
        else:
            return None  # No clear signal
        
        # Get price from data
        try:
            price_idx = data.index.get_loc(timestamp)
            price = data['close'].iloc[price_idx] if 'close' in data.columns else data.iloc[price_idx, 0]
        except (KeyError, IndexError):
            price = signal_list[0][0].price  # Fallback to first signal's price
        
        return TradingSignal(
            signal_type,
            min(1.0, total_confidence),
            timestamp,
            price,
            indicators=combined_indicators,
            patterns=combined_patterns
        )


class PatternBasedSignals(SignalGenerator):
    """Generate signals based on chart patterns"""
    
    def __init__(self):
        super().__init__("pattern_based")
        self.support_resistance = SupportResistanceDetector()
        self.trendlines = TrendlineDetector()
        self.head_shoulders = HeadAndShouldersDetector()
        self.double_patterns = DoubleTopBottomDetector()
        self.candlesticks = CandlestickPatterns()
        
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on detected patterns"""
        signals = []
        
        try:
            # Detect patterns
            sr_patterns = self.support_resistance.detect(data)
            trendline_patterns = self.trendlines.detect(data)
            hs_patterns = self.head_shoulders.detect(data)
            double_patterns = self.double_patterns.detect(data)
            candlestick_patterns = self.candlesticks.detect(data)
            
            # Generate signals from support/resistance breaks
            signals.extend(self._generate_sr_signals(data, sr_patterns))
            
            # Generate signals from trendline breaks
            signals.extend(self._generate_trendline_signals(data, trendline_patterns))
            
            # Generate signals from reversal patterns
            signals.extend(self._generate_reversal_signals(data, hs_patterns, double_patterns))
            
            # Generate signals from candlestick patterns
            signals.extend(self._generate_candlestick_signals(data, candlestick_patterns))
            
        except Exception as e:
            self.logger.error(f"Error generating pattern signals: {e}")
        
        return signals
    
    def _generate_sr_signals(self, data: pd.DataFrame, patterns: Dict) -> List[TradingSignal]:
        """Generate signals from support/resistance level breaks"""
        signals = []
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        # Check for resistance breaks (bullish)
        for resistance in patterns.get('resistance_levels', []):
            level = resistance['level']
            strength = resistance['strength']
            
            # Find recent breaks above resistance
            breaks = (close > level) & (close.shift(1) <= level)
            for idx in close.index[breaks]:
                signals.append(TradingSignal(
                    SignalType.BUY,
                    strength * 0.8,  # Confidence based on level strength
                    idx,
                    close.loc[idx],
                    patterns={'resistance_break': resistance}
                ))
        
        # Check for support breaks (bearish)
        for support in patterns.get('support_levels', []):
            level = support['level']
            strength = support['strength']
            
            # Find recent breaks below support
            breaks = (close < level) & (close.shift(1) >= level)
            for idx in close.index[breaks]:
                signals.append(TradingSignal(
                    SignalType.SELL,
                    strength * 0.8,
                    idx,
                    close.loc[idx],
                    patterns={'support_break': support}
                ))
        
        return signals
    
    def _generate_trendline_signals(self, data: pd.DataFrame, patterns: Dict) -> List[TradingSignal]:
        """Generate signals from trendline breaks"""
        signals = []
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        # Uptrend line breaks (bearish)
        for trendline in patterns.get('uptrend_lines', []):
            # Calculate trendline values
            start_idx = trendline['start_index']
            end_idx = trendline['end_index']
            slope = trendline['slope']
            intercept = trendline['intercept']
            
            # Check for breaks below uptrend line
            for i in range(end_idx, len(data)):
                expected_value = slope * i + intercept
                if close.iloc[i] < expected_value * 0.98:  # 2% break threshold
                    signals.append(TradingSignal(
                        SignalType.SELL,
                        trendline['strength'] * 0.7,
                        data.index[i],
                        close.iloc[i],
                        patterns={'uptrend_break': trendline}
                    ))
                    break
        
        return signals
    
    def _generate_reversal_signals(self, data: pd.DataFrame, hs_patterns: Dict, double_patterns: Dict) -> List[TradingSignal]:
        """Generate signals from reversal patterns"""
        signals = []
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        # Head and shoulders patterns
        for pattern in hs_patterns.get('head_and_shoulders', []):
            neckline = pattern['neckline']
            confidence = pattern['confidence']
            
            # Look for neckline break
            breaks = (close < neckline) & (close.shift(1) >= neckline)
            for idx in close.index[breaks]:
                signals.append(TradingSignal(
                    SignalType.SELL,
                    confidence * 0.9,
                    idx,
                    close.loc[idx],
                    patterns={'head_shoulders': pattern},
                    take_profit=pattern['target_price']
                ))
        
        return signals
    
    def _generate_candlestick_signals(self, data: pd.DataFrame, patterns: Dict) -> List[TradingSignal]:
        """Generate signals from candlestick patterns"""
        signals = []
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        # Bullish patterns
        for pattern_name in ['hammer', 'morning_star']:
            for idx in patterns.get(pattern_name, []):
                if idx < len(data):
                    signals.append(TradingSignal(
                        SignalType.BUY,
                        0.6,  # Moderate confidence for candlestick patterns
                        data.index[idx],
                        close.iloc[idx],
                        patterns={pattern_name: True}
                    ))
        
        # Bearish patterns
        for pattern_name in ['shooting_star', 'evening_star']:
            for idx in patterns.get(pattern_name, []):
                if idx < len(data):
                    signals.append(TradingSignal(
                        SignalType.SELL,
                        0.6,
                        data.index[idx],
                        close.iloc[idx],
                        patterns={pattern_name: True}
                    ))
        
        return signals


class SignalFilter:
    """Filter and validate trading signals"""
    
    def __init__(self, min_confidence: float = 0.5, max_signals_per_day: int = 3):
        self.min_confidence = min_confidence
        self.max_signals_per_day = max_signals_per_day
        self.logger = Logger.get_logger("signal_filter")
        
    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on confidence and frequency"""
        # Filter by minimum confidence
        filtered_signals = [s for s in signals if s.confidence >= self.min_confidence]
        
        # Group by date and limit signals per day
        daily_signals = {}
        for signal in filtered_signals:
            date = signal.timestamp.date()
            if date not in daily_signals:
                daily_signals[date] = []
            daily_signals[date].append(signal)
        
        # Keep only top signals per day
        final_signals = []
        for date, day_signals in daily_signals.items():
            # Sort by confidence and keep top N
            day_signals.sort(key=lambda x: x.confidence, reverse=True)
            final_signals.extend(day_signals[:self.max_signals_per_day])
        
        return sorted(final_signals, key=lambda x: x.timestamp)