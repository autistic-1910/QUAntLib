#!/usr/bin/env python3
"""
Signal Generation Module

Provides various signal generators for different types of market analysis:
- Technical signal generation
- Fundamental signal generation
- Sentiment signal generation
- Macro signal generation
- Signal combination and filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

from quantlib.core.utils import Logger
from quantlib.technical.indicators import (
    SimpleMovingAverage, ExponentialMovingAverage, RelativeStrengthIndex, 
    MACD, BollingerBands
)
from quantlib.technical.oscillators import RateOfChange, MomentumOscillator
from quantlib.strategy.base import StrategySignal, SignalType


class SignalGenerator(ABC):
    """Base class for signal generators"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger.get_logger(f"signal_{name.lower()}")
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, 
                        timestamp: datetime,
                        assets: List[str]) -> List[StrategySignal]:
        """Generate trading signals"""
        pass
    
    def _normalize_signal_strength(self, value: float, 
                                  min_val: float = -1.0, 
                                  max_val: float = 1.0) -> float:
        """Normalize signal strength to [0, 1]"""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))


class TechnicalSignalGenerator(SignalGenerator):
    """Technical analysis signal generator"""
    
    def __init__(self, indicators_config: Optional[Dict] = None):
        super().__init__("TechnicalSignals")
        
        # Default indicator configuration
        self.config = indicators_config or {
            'sma_short': 20,
            'sma_long': 50,
            'ema_period': 12,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0,
            'momentum_period': 10
        }
        
        # Initialize indicators
        self.indicators = {
            'sma_short': SimpleMovingAverage(self.config['sma_short']),
            'sma_long': SimpleMovingAverage(self.config['sma_long']),
            'ema': ExponentialMovingAverage(self.config['ema_period']),
            'rsi': RelativeStrengthIndex(self.config['rsi_period']),
            'macd': MACD(self.config['macd_fast'], 
                        self.config['macd_slow'], self.config['macd_signal']),
            'bb': BollingerBands(self.config['bb_period'], self.config['bb_std']),
            'momentum': MomentumOscillator(self.config['momentum_period'])
        }
    
    def generate_signals(self, data: pd.DataFrame, 
                        timestamp: datetime,
                        assets: List[str]) -> List[StrategySignal]:
        """Generate technical analysis signals"""
        signals = []
        
        for asset in assets:
            if asset not in data.columns:
                continue
                
            asset_data = data[asset].dropna()
            if len(asset_data) < max(self.config.values()):
                continue
            
            # Calculate indicators
            indicator_values = {}
            for name, indicator in self.indicators.items():
                try:
                    values = indicator.calculate(asset_data)
                    if len(values) > 0:
                        indicator_values[name] = values.iloc[-1] if hasattr(values, 'iloc') else values[-1]
                except Exception as e:
                    self.logger.warning(f"Error calculating {name} for {asset}: {e}")
                    continue
            
            if not indicator_values:
                continue
            
            # Generate signals based on technical conditions
            signal = self._analyze_technical_conditions(asset, asset_data, indicator_values, timestamp)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_technical_conditions(self, asset: str, price_data: pd.Series,
                                    indicators: Dict, timestamp: datetime) -> Optional[StrategySignal]:
        """Analyze technical conditions and generate signal"""
        current_price = price_data.iloc[-1]
        
        # Initialize signal components
        signal_strength = 0.0
        signal_count = 0
        confidence_factors = []
        
        # Moving Average Crossover
        if 'sma_short' in indicators and 'sma_long' in indicators:
            sma_short = indicators['sma_short']
            sma_long = indicators['sma_long']
            
            if sma_short > sma_long:
                signal_strength += 1
            else:
                signal_strength -= 1
            signal_count += 1
            
            # Confidence based on separation
            separation = abs(sma_short - sma_long) / current_price
            confidence_factors.append(min(1.0, separation * 10))
        
        # RSI Conditions
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            
            if rsi < 30:  # Oversold
                signal_strength += 1
                confidence_factors.append(min(1.0, (30 - rsi) / 30))
            elif rsi > 70:  # Overbought
                signal_strength -= 1
                confidence_factors.append(min(1.0, (rsi - 70) / 30))
            else:
                confidence_factors.append(0.5)
            signal_count += 1
        
        # MACD Signal
        if 'macd' in indicators:
            macd_data = indicators['macd']
            if isinstance(macd_data, dict):
                macd_line = macd_data.get('macd', 0)
                signal_line = macd_data.get('signal', 0)
                
                if macd_line > signal_line:
                    signal_strength += 1
                else:
                    signal_strength -= 1
                signal_count += 1
                
                # Confidence based on MACD strength
                macd_strength = abs(macd_line - signal_line)
                confidence_factors.append(min(1.0, macd_strength * 100))
        
        # Bollinger Bands
        if 'bb' in indicators:
            bb_data = indicators['bb']
            if isinstance(bb_data, dict):
                upper_band = bb_data.get('upper', current_price)
                lower_band = bb_data.get('lower', current_price)
                
                if current_price < lower_band:  # Below lower band
                    signal_strength += 1
                    confidence_factors.append(min(1.0, (lower_band - current_price) / current_price))
                elif current_price > upper_band:  # Above upper band
                    signal_strength -= 1
                    confidence_factors.append(min(1.0, (current_price - upper_band) / current_price))
                else:
                    confidence_factors.append(0.3)
                signal_count += 1
        
        # Momentum
        if 'momentum' in indicators:
            momentum = indicators['momentum']
            
            if momentum > 0:
                signal_strength += 1
            else:
                signal_strength -= 1
            signal_count += 1
            
            confidence_factors.append(min(1.0, abs(momentum) * 10))
        
        if signal_count == 0:
            return None
        
        # Calculate final signal
        avg_signal = signal_strength / signal_count
        avg_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Determine signal type
        if avg_signal > 0.3:
            signal_type = SignalType.BUY
        elif avg_signal < -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Skip weak signals
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate target position (simplified)
        target_position = avg_signal * avg_confidence
        
        return StrategySignal(
            timestamp=timestamp,
            asset=asset,
            signal_type=signal_type,
            strength=abs(avg_signal),
            confidence=avg_confidence,
            target_position=target_position,
            metadata={
                'price': current_price,
                'indicators': indicators,
                'signal_components': signal_count
            }
        )


class FundamentalSignalGenerator(SignalGenerator):
    """Fundamental analysis signal generator"""
    
    def __init__(self, fundamental_factors: Optional[List[str]] = None):
        super().__init__("FundamentalSignals")
        self.factors = fundamental_factors or [
            'pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'revenue_growth'
        ]
    
    def generate_signals(self, data: pd.DataFrame, 
                        timestamp: datetime,
                        assets: List[str]) -> List[StrategySignal]:
        """Generate fundamental analysis signals"""
        signals = []
        
        # For demonstration, generate simple fundamental signals
        # In practice, this would use actual fundamental data
        
        for asset in assets:
            # Simulate fundamental analysis
            fundamental_score = self._calculate_fundamental_score(asset)
            
            if fundamental_score > 0.6:
                signal_type = SignalType.BUY
            elif fundamental_score < 0.4:
                signal_type = SignalType.SELL
            else:
                continue
            
            signal = StrategySignal(
                timestamp=timestamp,
                asset=asset,
                signal_type=signal_type,
                strength=abs(fundamental_score - 0.5) * 2,
                confidence=0.7,  # Fundamental analysis typically has high confidence
                target_position=fundamental_score - 0.5,
                metadata={
                    'fundamental_score': fundamental_score,
                    'analysis_type': 'fundamental'
                }
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_fundamental_score(self, asset: str) -> float:
        """Calculate fundamental score (simplified simulation)"""
        # In practice, this would analyze actual fundamental data
        # For now, return a random score for demonstration
        np.random.seed(hash(asset) % 2**32)
        return np.random.uniform(0.2, 0.8)


class SentimentSignalGenerator(SignalGenerator):
    """Sentiment analysis signal generator"""
    
    def __init__(self, sentiment_sources: Optional[List[str]] = None):
        super().__init__("SentimentSignals")
        self.sources = sentiment_sources or ['news', 'social_media', 'analyst_ratings']
    
    def generate_signals(self, data: pd.DataFrame, 
                        timestamp: datetime,
                        assets: List[str]) -> List[StrategySignal]:
        """Generate sentiment-based signals"""
        signals = []
        
        for asset in assets:
            # Simulate sentiment analysis
            sentiment_score = self._analyze_sentiment(asset)
            
            if sentiment_score > 0.6:
                signal_type = SignalType.BUY
            elif sentiment_score < 0.4:
                signal_type = SignalType.SELL
            else:
                continue
            
            # Sentiment signals typically have lower confidence
            confidence = 0.5
            
            signal = StrategySignal(
                timestamp=timestamp,
                asset=asset,
                signal_type=signal_type,
                strength=abs(sentiment_score - 0.5) * 2,
                confidence=confidence,
                target_position=(sentiment_score - 0.5) * 0.5,  # Smaller positions for sentiment
                metadata={
                    'sentiment_score': sentiment_score,
                    'analysis_type': 'sentiment'
                }
            )
            signals.append(signal)
        
        return signals
    
    def _analyze_sentiment(self, asset: str) -> float:
        """Analyze sentiment for asset (simplified simulation)"""
        # In practice, this would analyze news, social media, etc.
        np.random.seed((hash(asset) + 1) % 2**32)
        return np.random.uniform(0.3, 0.7)


class MacroSignalGenerator(SignalGenerator):
    """Macroeconomic signal generator"""
    
    def __init__(self, macro_indicators: Optional[List[str]] = None):
        super().__init__("MacroSignals")
        self.indicators = macro_indicators or [
            'gdp_growth', 'inflation', 'interest_rates', 'unemployment', 'vix'
        ]
    
    def generate_signals(self, data: pd.DataFrame, 
                        timestamp: datetime,
                        assets: List[str]) -> List[StrategySignal]:
        """Generate macro-based signals"""
        signals = []
        
        # Calculate macro environment score
        macro_score = self._calculate_macro_score()
        
        # Apply macro signals to all assets (systematic risk)
        for asset in assets:
            if macro_score > 0.6:
                signal_type = SignalType.BUY
            elif macro_score < 0.4:
                signal_type = SignalType.SELL
            else:
                continue
            
            signal = StrategySignal(
                timestamp=timestamp,
                asset=asset,
                signal_type=signal_type,
                strength=abs(macro_score - 0.5) * 2,
                confidence=0.6,
                target_position=(macro_score - 0.5) * 0.3,  # Moderate positions for macro
                metadata={
                    'macro_score': macro_score,
                    'analysis_type': 'macro'
                }
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_macro_score(self) -> float:
        """Calculate macroeconomic environment score"""
        # In practice, this would analyze actual macro indicators
        np.random.seed(42)  # Fixed seed for consistent macro environment
        return np.random.uniform(0.4, 0.6)


class SignalCombiner:
    """Combines signals from multiple generators"""
    
    def __init__(self, generators: List[SignalGenerator],
                 weights: Optional[Dict[str, float]] = None):
        self.generators = generators
        self.weights = weights or {gen.name: 1.0 for gen in generators}
        self.logger = Logger.get_logger("signal_combiner")
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def generate_combined_signals(self, data: pd.DataFrame,
                                timestamp: datetime,
                                assets: List[str]) -> List[StrategySignal]:
        """Generate combined signals from all generators"""
        # Collect signals from all generators
        all_signals = {}
        
        for generator in self.generators:
            try:
                signals = generator.generate_signals(data, timestamp, assets)
                for signal in signals:
                    key = signal.asset
                    if key not in all_signals:
                        all_signals[key] = []
                    all_signals[key].append((generator.name, signal))
            except Exception as e:
                self.logger.error(f"Error generating signals from {generator.name}: {e}")
        
        # Combine signals for each asset
        combined_signals = []
        
        for asset, asset_signals in all_signals.items():
            combined_signal = self._combine_asset_signals(asset, asset_signals, timestamp)
            if combined_signal:
                combined_signals.append(combined_signal)
        
        return combined_signals
    
    def _combine_asset_signals(self, asset: str, 
                             signals: List[Tuple[str, StrategySignal]],
                             timestamp: datetime) -> Optional[StrategySignal]:
        """Combine signals for a single asset"""
        if not signals:
            return None
        
        # Calculate weighted averages
        total_strength = 0.0
        total_confidence = 0.0
        total_position = 0.0
        total_weight = 0.0
        
        signal_types = []
        metadata = {'component_signals': {}}
        
        for generator_name, signal in signals:
            weight = self.weights.get(generator_name, 0.0)
            
            # Convert signal type to numeric value
            signal_value = signal.signal_type.value
            
            total_strength += signal.strength * weight
            total_confidence += signal.confidence * weight
            total_position += signal.target_position * weight
            total_weight += weight
            
            signal_types.append(signal_value)
            metadata['component_signals'][generator_name] = {
                'type': signal.signal_type.name,
                'strength': signal.strength,
                'confidence': signal.confidence
            }
        
        if total_weight == 0:
            return None
        
        # Calculate final values
        avg_strength = total_strength / total_weight
        avg_confidence = total_confidence / total_weight
        avg_position = total_position / total_weight
        
        # Determine combined signal type
        avg_signal_value = np.mean(signal_types)
        
        if avg_signal_value > 0.5:
            combined_signal_type = SignalType.BUY
        elif avg_signal_value < -0.5:
            combined_signal_type = SignalType.SELL
        else:
            combined_signal_type = SignalType.HOLD
        
        # Skip weak combined signals
        if combined_signal_type == SignalType.HOLD or avg_strength < 0.3:
            return None
        
        return StrategySignal(
            timestamp=timestamp,
            asset=asset,
            signal_type=combined_signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            target_position=avg_position,
            metadata=metadata
        )