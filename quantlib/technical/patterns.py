"""Chart Pattern Recognition Module

Implements algorithms to detect common chart patterns including:
- Trend patterns (support/resistance, trendlines)
- Reversal patterns (head and shoulders, double tops/bottoms)
- Continuation patterns (triangles, flags, pennants)
- Candlestick patterns (doji, hammer, engulfing)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from scipy import stats
from scipy.signal import find_peaks, find_peaks_cwt
import warnings

from quantlib.core.base import BaseIndicator
from quantlib.core.utils import Logger


class PatternDetector(ABC):
    """Base class for pattern detection algorithms"""
    
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__.lower())
        
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect patterns in price data"""
        pass
    
    def _find_peaks_and_troughs(self, data: pd.Series, prominence: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and troughs in price data"""
        # Normalize data for peak detection
        normalized_data = (data - data.min()) / (data.max() - data.min())
        
        # Find peaks (local maxima)
        peaks, _ = find_peaks(normalized_data, prominence=prominence)
        
        # Find troughs (local minima) by inverting the data
        troughs, _ = find_peaks(-normalized_data, prominence=prominence)
        
        return peaks, troughs


class SupportResistanceDetector(PatternDetector):
    """Detect support and resistance levels"""
    
    def __init__(self, min_touches: int = 3, tolerance: float = 0.02):
        super().__init__()
        self.min_touches = min_touches
        self.tolerance = tolerance
        
    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect support and resistance levels"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        high = data['high'] if 'high' in data.columns else close
        low = data['low'] if 'low' in data.columns else close
        
        peaks, troughs = self._find_peaks_and_troughs(close)
        
        # Find resistance levels from peaks
        resistance_levels = self._find_levels(high.iloc[peaks], high, is_resistance=True)
        
        # Find support levels from troughs
        support_levels = self._find_levels(low.iloc[troughs], low, is_resistance=False)
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'peaks': peaks,
            'troughs': troughs
        }
    
    def _find_levels(self, candidate_levels: pd.Series, price_series: pd.Series, is_resistance: bool) -> List[Dict]:
        """Find significant support or resistance levels"""
        levels = []
        
        for level in candidate_levels:
            touches = self._count_touches(level, price_series, is_resistance)
            
            if touches >= self.min_touches:
                strength = min(touches / 10.0, 1.0)  # Normalize strength
                levels.append({
                    'level': level,
                    'touches': touches,
                    'strength': strength,
                    'type': 'resistance' if is_resistance else 'support'
                })
        
        # Sort by strength
        levels.sort(key=lambda x: x['strength'], reverse=True)
        return levels
    
    def _count_touches(self, level: float, price_series: pd.Series, is_resistance: bool) -> int:
        """Count how many times price touched a level"""
        tolerance_range = level * self.tolerance
        
        if is_resistance:
            # Count touches from below
            touches = ((price_series >= level - tolerance_range) & 
                      (price_series <= level + tolerance_range)).sum()
        else:
            # Count touches from above
            touches = ((price_series >= level - tolerance_range) & 
                      (price_series <= level + tolerance_range)).sum()
        
        return touches


class TrendlineDetector(PatternDetector):
    """Detect trendlines in price data"""
    
    def __init__(self, min_points: int = 3, max_deviation: float = 0.05):
        super().__init__()
        self.min_points = min_points
        self.max_deviation = max_deviation
        
    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect uptrend and downtrend lines"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        high = data['high'] if 'high' in data.columns else close
        low = data['low'] if 'low' in data.columns else close
        
        peaks, troughs = self._find_peaks_and_troughs(close)
        
        # Detect uptrend lines using troughs
        uptrend_lines = self._detect_trendlines(troughs, low.iloc[troughs], low.index[troughs], 'uptrend')
        
        # Detect downtrend lines using peaks
        downtrend_lines = self._detect_trendlines(peaks, high.iloc[peaks], high.index[peaks], 'downtrend')
        
        return {
            'uptrend_lines': uptrend_lines,
            'downtrend_lines': downtrend_lines
        }
    
    def _detect_trendlines(self, indices: np.ndarray, values: pd.Series, timestamps: pd.Index, trend_type: str) -> List[Dict]:
        """Detect trendlines from peaks or troughs"""
        trendlines = []
        
        if len(indices) < self.min_points:
            return trendlines
        
        # Try different combinations of points
        for i in range(len(indices) - self.min_points + 1):
            for j in range(i + self.min_points - 1, len(indices)):
                # Get points for trendline
                x_points = np.arange(i, j + 1)
                y_points = values.iloc[i:j+1].values
                
                # Fit linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)
                
                # Check if trendline is valid
                if self._is_valid_trendline(x_points, y_points, slope, intercept, trend_type):
                    # Ensure indices are within bounds
                    start_idx = min(indices[i], len(timestamps) - 1)
                    end_idx = min(indices[j], len(timestamps) - 1)
                    
                    trendlines.append({
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'start_time': timestamps[start_idx],
                        'end_time': timestamps[end_idx],
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value ** 2,
                        'type': trend_type,
                        'strength': abs(r_value)
                    })
        
        # Sort by strength and return top trendlines
        trendlines.sort(key=lambda x: x['strength'], reverse=True)
        return trendlines[:5]  # Return top 5 trendlines
    
    def _is_valid_trendline(self, x_points: np.ndarray, y_points: np.ndarray, slope: float, intercept: float, trend_type: str) -> bool:
        """Check if trendline meets validity criteria"""
        # Calculate predicted values
        predicted = slope * x_points + intercept
        
        # Calculate maximum deviation
        max_deviation = np.max(np.abs(y_points - predicted)) / np.mean(y_points)
        
        # Check slope direction
        if trend_type == 'uptrend' and slope <= 0:
            return False
        elif trend_type == 'downtrend' and slope >= 0:
            return False
        
        # Check deviation threshold
        return max_deviation <= self.max_deviation


class HeadAndShouldersDetector(PatternDetector):
    """Detect Head and Shoulders patterns"""
    
    def __init__(self, shoulder_tolerance: float = 0.05, head_min_height: float = 0.1):
        super().__init__()
        self.shoulder_tolerance = shoulder_tolerance
        self.head_min_height = head_min_height
        
    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect Head and Shoulders patterns"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        high = data['high'] if 'high' in data.columns else close
        
        peaks, _ = self._find_peaks_and_troughs(close)
        
        patterns = []
        
        # Need at least 3 peaks for head and shoulders
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                pattern = self._analyze_head_shoulders(
                    left_shoulder, head, right_shoulder,
                    high.iloc[left_shoulder], high.iloc[head], high.iloc[right_shoulder],
                    close.index
                )
                
                if pattern:
                    patterns.append(pattern)
        
        return {'head_and_shoulders': patterns}
    
    def _analyze_head_shoulders(self, ls_idx: int, h_idx: int, rs_idx: int, 
                               ls_price: float, h_price: float, rs_price: float,
                               timestamps: pd.Index) -> Optional[Dict]:
        """Analyze if three peaks form a head and shoulders pattern"""
        
        # Check if head is higher than both shoulders
        if h_price <= max(ls_price, rs_price):
            return None
        
        # Check if shoulders are approximately equal
        shoulder_diff = abs(ls_price - rs_price) / max(ls_price, rs_price)
        if shoulder_diff > self.shoulder_tolerance:
            return None
        
        # Check if head is significantly higher
        head_height = (h_price - max(ls_price, rs_price)) / max(ls_price, rs_price)
        if head_height < self.head_min_height:
            return None
        
        # Calculate neckline (support level)
        neckline = min(ls_price, rs_price)
        
        return {
            'type': 'head_and_shoulders',
            'left_shoulder': {'index': ls_idx, 'price': ls_price, 'time': timestamps[ls_idx]},
            'head': {'index': h_idx, 'price': h_price, 'time': timestamps[h_idx]},
            'right_shoulder': {'index': rs_idx, 'price': rs_price, 'time': timestamps[rs_idx]},
            'neckline': neckline,
            'target_price': neckline - (h_price - neckline),  # Price target
            'confidence': 1 - shoulder_diff  # Higher confidence for more equal shoulders
        }


class DoubleTopBottomDetector(PatternDetector):
    """Detect Double Top and Double Bottom patterns"""
    
    def __init__(self, price_tolerance: float = 0.03, min_separation: int = 10):
        super().__init__()
        self.price_tolerance = price_tolerance
        self.min_separation = min_separation
        
    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect Double Top and Double Bottom patterns"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        high = data['high'] if 'high' in data.columns else close
        low = data['low'] if 'low' in data.columns else close
        
        peaks, troughs = self._find_peaks_and_troughs(close)
        
        double_tops = self._find_double_patterns(peaks, high.iloc[peaks], close.index, 'double_top')
        double_bottoms = self._find_double_patterns(troughs, low.iloc[troughs], close.index, 'double_bottom')
        
        return {
            'double_tops': double_tops,
            'double_bottoms': double_bottoms
        }
    
    def _find_double_patterns(self, indices: np.ndarray, prices: pd.Series, timestamps: pd.Index, pattern_type: str) -> List[Dict]:
        """Find double top or double bottom patterns"""
        patterns = []
        
        for i in range(len(indices) - 1):
            for j in range(i + 1, len(indices)):
                # Check minimum separation
                if indices[j] - indices[i] < self.min_separation:
                    continue
                
                price1 = prices.iloc[i]
                price2 = prices.iloc[j]
                
                # Check if prices are approximately equal
                price_diff = abs(price1 - price2) / max(price1, price2)
                
                if price_diff <= self.price_tolerance:
                    patterns.append({
                        'type': pattern_type,
                        'first_peak': {'index': indices[i], 'price': price1, 'time': timestamps[indices[i]]},
                        'second_peak': {'index': indices[j], 'price': price2, 'time': timestamps[indices[j]]},
                        'confidence': 1 - price_diff,
                        'separation': indices[j] - indices[i]
                    })
        
        # Sort by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        return patterns[:3]  # Return top 3 patterns


class TriangleDetector(PatternDetector):
    """Detect Triangle patterns (ascending, descending, symmetrical)"""
    
    def __init__(self, min_touches: int = 4, convergence_threshold: float = 0.1):
        super().__init__()
        self.min_touches = min_touches
        self.convergence_threshold = convergence_threshold
        
    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect triangle patterns"""
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        high = data['high'] if 'high' in data.columns else close
        low = data['low'] if 'low' in data.columns else close
        
        peaks, troughs = self._find_peaks_and_troughs(close)
        
        triangles = []
        
        # Analyze recent peaks and troughs for triangle formation
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Get recent peaks and troughs
            recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
            recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
            
            triangle = self._analyze_triangle_formation(
                recent_peaks, recent_troughs,
                high.iloc[recent_peaks], low.iloc[recent_troughs],
                close.index
            )
            
            if triangle:
                triangles.append(triangle)
        
        return {'triangles': triangles}
    
    def _analyze_triangle_formation(self, peaks: np.ndarray, troughs: np.ndarray,
                                   peak_prices: pd.Series, trough_prices: pd.Series,
                                   timestamps: pd.Index) -> Optional[Dict]:
        """Analyze if peaks and troughs form a triangle pattern"""
        
        if len(peaks) < 2 or len(troughs) < 2:
            return None
        
        # Fit trendlines to peaks and troughs
        peak_slope, peak_intercept, peak_r, _, _ = stats.linregress(peaks, peak_prices)
        trough_slope, trough_intercept, trough_r, _, _ = stats.linregress(troughs, trough_prices)
        
        # Check if lines are converging
        slope_diff = abs(peak_slope - trough_slope)
        
        if slope_diff < self.convergence_threshold:
            return None
        
        # Determine triangle type
        triangle_type = self._classify_triangle(peak_slope, trough_slope)
        
        # Calculate convergence point
        if peak_slope != trough_slope:
            convergence_x = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            convergence_y = peak_slope * convergence_x + peak_intercept
        else:
            convergence_x = convergence_y = None
        
        return {
            'type': triangle_type,
            'peak_trendline': {'slope': peak_slope, 'intercept': peak_intercept, 'r_squared': peak_r**2},
            'trough_trendline': {'slope': trough_slope, 'intercept': trough_intercept, 'r_squared': trough_r**2},
            'convergence_point': {'x': convergence_x, 'y': convergence_y},
            'peaks': [{'index': idx, 'price': price, 'time': timestamps[idx]} 
                     for idx, price in zip(peaks, peak_prices)],
            'troughs': [{'index': idx, 'price': price, 'time': timestamps[idx]} 
                       for idx, price in zip(troughs, trough_prices)],
            'confidence': min(abs(peak_r), abs(trough_r))
        }
    
    def _classify_triangle(self, peak_slope: float, trough_slope: float) -> str:
        """Classify triangle type based on trendline slopes"""
        if abs(peak_slope) < 0.001:  # Horizontal resistance
            return 'ascending_triangle'
        elif abs(trough_slope) < 0.001:  # Horizontal support
            return 'descending_triangle'
        elif peak_slope < 0 and trough_slope > 0:  # Converging lines
            return 'symmetrical_triangle'
        else:
            return 'unknown_triangle'


class CandlestickPatterns(PatternDetector):
    """Detect common candlestick patterns"""
    
    def __init__(self):
        super().__init__()
        
    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect various candlestick patterns"""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            self.logger.warning("OHLC data required for candlestick pattern detection")
            return {}
        
        patterns = {
            'doji': self._detect_doji(data),
            'hammer': self._detect_hammer(data),
            'shooting_star': self._detect_shooting_star(data),
            'engulfing': self._detect_engulfing(data),
            'morning_star': self._detect_morning_star(data),
            'evening_star': self._detect_evening_star(data)
        }
        
        return patterns
    
    def _detect_doji(self, data: pd.DataFrame) -> List[int]:
        """Detect Doji candlestick patterns"""
        open_price = data['open']
        close_price = data['close']
        high_price = data['high']
        low_price = data['low']
        
        # Body size relative to range
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        # Doji: very small body relative to range
        doji_threshold = 0.1
        doji_mask = (body_size / total_range) < doji_threshold
        
        return data.index[doji_mask].tolist()
    
    def _detect_hammer(self, data: pd.DataFrame) -> List[int]:
        """Detect Hammer candlestick patterns"""
        open_price = data['open']
        close_price = data['close']
        high_price = data['high']
        low_price = data['low']
        
        body_top = np.maximum(open_price, close_price)
        body_bottom = np.minimum(open_price, close_price)
        
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        body_size = abs(close_price - open_price)
        
        # Hammer: long lower shadow, short upper shadow, small body
        hammer_mask = (
            (lower_shadow > 2 * body_size) &
            (upper_shadow < 0.5 * body_size) &
            (body_size > 0)
        )
        
        return data.index[hammer_mask].tolist()
    
    def _detect_shooting_star(self, data: pd.DataFrame) -> List[int]:
        """Detect Shooting Star candlestick patterns"""
        open_price = data['open']
        close_price = data['close']
        high_price = data['high']
        low_price = data['low']
        
        body_top = np.maximum(open_price, close_price)
        body_bottom = np.minimum(open_price, close_price)
        
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        body_size = abs(close_price - open_price)
        
        # Shooting Star: long upper shadow, short lower shadow, small body
        shooting_star_mask = (
            (upper_shadow > 2 * body_size) &
            (lower_shadow < 0.5 * body_size) &
            (body_size > 0)
        )
        
        return data.index[shooting_star_mask].tolist()
    
    def _detect_engulfing(self, data: pd.DataFrame) -> List[int]:
        """Detect Bullish/Bearish Engulfing patterns"""
        open_price = data['open']
        close_price = data['close']
        
        # Current candle
        curr_body_top = np.maximum(open_price, close_price)
        curr_body_bottom = np.minimum(open_price, close_price)
        curr_bullish = close_price > open_price
        
        # Previous candle
        prev_body_top = np.maximum(open_price.shift(1), close_price.shift(1))
        prev_body_bottom = np.minimum(open_price.shift(1), close_price.shift(1))
        prev_bullish = close_price.shift(1) > open_price.shift(1)
        
        # Bullish engulfing: current bullish candle engulfs previous bearish candle
        bullish_engulfing = (
            curr_bullish & ~prev_bullish &
            (curr_body_bottom < prev_body_bottom) &
            (curr_body_top > prev_body_top)
        )
        
        # Bearish engulfing: current bearish candle engulfs previous bullish candle
        bearish_engulfing = (
            ~curr_bullish & prev_bullish &
            (curr_body_bottom < prev_body_bottom) &
            (curr_body_top > prev_body_top)
        )
        
        engulfing_mask = bullish_engulfing | bearish_engulfing
        return data.index[engulfing_mask].tolist()
    
    def _detect_morning_star(self, data: pd.DataFrame) -> List[int]:
        """Detect Morning Star patterns (3-candle bullish reversal)"""
        if len(data) < 3:
            return []
        
        open_price = data['open']
        close_price = data['close']
        high_price = data['high']
        low_price = data['low']
        
        patterns = []
        
        for i in range(2, len(data)):
            # First candle: bearish
            first_bearish = close_price.iloc[i-2] < open_price.iloc[i-2]
            
            # Second candle: small body (star)
            second_body = abs(close_price.iloc[i-1] - open_price.iloc[i-1])
            second_range = high_price.iloc[i-1] - low_price.iloc[i-1]
            second_small = second_body < 0.3 * second_range
            
            # Third candle: bullish
            third_bullish = close_price.iloc[i] > open_price.iloc[i]
            
            # Gap conditions
            gap_down = high_price.iloc[i-1] < min(open_price.iloc[i-2], close_price.iloc[i-2])
            gap_up = low_price.iloc[i] > max(open_price.iloc[i-1], close_price.iloc[i-1])
            
            if first_bearish and second_small and third_bullish and gap_down and gap_up:
                patterns.append(i)
        
        return patterns
    
    def _detect_evening_star(self, data: pd.DataFrame) -> List[int]:
        """Detect Evening Star patterns (3-candle bearish reversal)"""
        if len(data) < 3:
            return []
        
        open_price = data['open']
        close_price = data['close']
        high_price = data['high']
        low_price = data['low']
        
        patterns = []
        
        for i in range(2, len(data)):
            # First candle: bullish
            first_bullish = close_price.iloc[i-2] > open_price.iloc[i-2]
            
            # Second candle: small body (star)
            second_body = abs(close_price.iloc[i-1] - open_price.iloc[i-1])
            second_range = high_price.iloc[i-1] - low_price.iloc[i-1]
            second_small = second_body < 0.3 * second_range
            
            # Third candle: bearish
            third_bearish = close_price.iloc[i] < open_price.iloc[i]
            
            # Gap conditions
            gap_up = low_price.iloc[i-1] > max(open_price.iloc[i-2], close_price.iloc[i-2])
            gap_down = high_price.iloc[i] < min(open_price.iloc[i-1], close_price.iloc[i-1])
            
            if first_bullish and second_small and third_bearish and gap_up and gap_down:
                patterns.append(i)
        
        return patterns