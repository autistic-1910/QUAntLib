#!/usr/bin/env python3
"""
Portfolio Allocation Module

Implements various portfolio allocation strategies including:
- Strategic Asset Allocation
- Tactical Asset Allocation
- Dynamic Asset Allocation
- Risk-based allocation methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from quantlib.core.utils import Logger
from quantlib.portfolio.optimization import (
    MeanVarianceOptimizer, RiskBudgetOptimizer, 
    HierarchicalRiskParity, EfficientFrontier
)


class BaseAllocator(ABC):
    """Base class for portfolio allocators"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger.get_logger(f"allocator_{name.lower()}")
        
    @abstractmethod
    def allocate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Generate portfolio allocation"""
        pass
        
    def validate_allocation(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize allocation weights"""
        total_weight = sum(weights.values())
        
        if abs(total_weight - 1.0) > 1e-6:
            self.logger.warning(f"Weights sum to {total_weight:.6f}, normalizing to 1.0")
            weights = {asset: weight / total_weight for asset, weight in weights.items()}
        
        # Remove negative weights (set to zero)
        weights = {asset: max(0, weight) for asset, weight in weights.items()}
        
        return weights


class StrategicAssetAllocator(BaseAllocator):
    """Strategic Asset Allocation - Long-term target allocation"""
    
    def __init__(self, target_allocation: Dict[str, float], 
                 rebalance_frequency: str = 'quarterly'):
        super().__init__("StrategicAssetAllocation")
        self.target_allocation = self.validate_allocation(target_allocation)
        self.rebalance_frequency = rebalance_frequency
        
    def allocate(self, data: pd.DataFrame, 
                current_weights: Optional[Dict[str, float]] = None,
                **kwargs) -> Dict[str, float]:
        """Return strategic allocation weights"""
        
        # If no current weights provided, return target allocation
        if current_weights is None:
            return self.target_allocation.copy()
        
        # Check if rebalancing is needed
        if self._should_rebalance(current_weights, **kwargs):
            self.logger.info("Rebalancing to strategic allocation")
            return self.target_allocation.copy()
        else:
            return current_weights
    
    def _should_rebalance(self, current_weights: Dict[str, float], 
                         last_rebalance: Optional[datetime] = None,
                         threshold: float = 0.05, **kwargs) -> bool:
        """Determine if rebalancing is needed"""
        
        # Time-based rebalancing
        if last_rebalance is not None:
            days_since_rebalance = (datetime.now() - last_rebalance).days
            
            if self.rebalance_frequency == 'monthly' and days_since_rebalance >= 30:
                return True
            elif self.rebalance_frequency == 'quarterly' and days_since_rebalance >= 90:
                return True
            elif self.rebalance_frequency == 'annually' and days_since_rebalance >= 365:
                return True
        
        # Threshold-based rebalancing
        for asset, target_weight in self.target_allocation.items():
            current_weight = current_weights.get(asset, 0)
            if abs(current_weight - target_weight) > threshold:
                return True
        
        return False


class TacticalAssetAllocator(BaseAllocator):
    """Tactical Asset Allocation - Short-term deviations from strategic allocation"""
    
    def __init__(self, strategic_allocation: Dict[str, float],
                 max_deviation: float = 0.1,
                 lookback_period: int = 252):
        super().__init__("TacticalAssetAllocation")
        self.strategic_allocation = strategic_allocation
        self.max_deviation = max_deviation
        self.lookback_period = lookback_period
        
    def allocate(self, data: pd.DataFrame, 
                signals: Optional[Dict[str, float]] = None,
                **kwargs) -> Dict[str, float]:
        """Generate tactical allocation based on market signals"""
        
        if signals is None:
            signals = self._generate_momentum_signals(data)
        
        tactical_weights = {}
        
        for asset, strategic_weight in self.strategic_allocation.items():
            signal = signals.get(asset, 0)
            
            # Apply tactical adjustment
            adjustment = signal * self.max_deviation
            tactical_weight = strategic_weight + adjustment
            
            # Ensure weights stay within reasonable bounds
            tactical_weight = max(0, min(1, tactical_weight))
            tactical_weights[asset] = tactical_weight
        
        return self.validate_allocation(tactical_weights)
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate momentum-based signals"""
        signals = {}
        
        for asset in data.columns:
            if len(data[asset].dropna()) < self.lookback_period:
                signals[asset] = 0
                continue
            
            # Calculate momentum (price change over lookback period)
            recent_data = data[asset].dropna().tail(self.lookback_period)
            momentum = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1
            
            # Normalize signal to [-1, 1]
            signals[asset] = np.tanh(momentum)
        
        return signals


class DynamicAssetAllocator(BaseAllocator):
    """Dynamic Asset Allocation - Continuous optimization based on market conditions"""
    
    def __init__(self, optimizer_type: str = 'mean_variance',
                 lookback_period: int = 252,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0):
        super().__init__("DynamicAssetAllocation")
        self.optimizer_type = optimizer_type
        self.lookback_period = lookback_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize optimizer
        if optimizer_type == 'mean_variance':
            self.optimizer = MeanVarianceOptimizer()
        elif optimizer_type == 'risk_budget':
            self.optimizer = RiskBudgetOptimizer()
        elif optimizer_type == 'hrp':
            self.optimizer = HierarchicalRiskParity()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def allocate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Generate dynamic allocation using optimization"""
        
        # Use recent data for optimization
        if len(data) > self.lookback_period:
            recent_data = data.tail(self.lookback_period)
        else:
            recent_data = data
        
        # Calculate returns
        returns = recent_data.pct_change().dropna()
        
        if len(returns) < 30:  # Minimum data requirement
            self.logger.warning("Insufficient data for optimization, using equal weights")
            n_assets = len(data.columns)
            return {asset: 1.0 / n_assets for asset in data.columns}
        
        # Set constraints
        constraints = {
            'min_weights': self.min_weight,
            'max_weights': self.max_weight
        }
        
        # Optimize
        try:
            result = self.optimizer.optimize(returns, constraints=constraints)
            
            if result.get('success', False):
                weights = dict(zip(data.columns, result['weights']))
                return self.validate_allocation(weights)
            else:
                self.logger.warning("Optimization failed, using equal weights")
                n_assets = len(data.columns)
                return {asset: 1.0 / n_assets for asset in data.columns}
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            n_assets = len(data.columns)
            return {asset: 1.0 / n_assets for asset in data.columns}


class RiskBasedAllocator(BaseAllocator):
    """Risk-based allocation strategies"""
    
    def __init__(self, method: str = 'equal_risk', 
                 lookback_period: int = 252):
        super().__init__(f"RiskBased_{method}")
        self.method = method
        self.lookback_period = lookback_period
        
    def allocate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Generate risk-based allocation"""
        
        if self.method == 'equal_risk':
            return self._equal_risk_allocation(data)
        elif self.method == 'inverse_volatility':
            return self._inverse_volatility_allocation(data)
        elif self.method == 'minimum_variance':
            return self._minimum_variance_allocation(data)
        else:
            raise ValueError(f"Unknown risk-based method: {self.method}")
    
    def _equal_risk_allocation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Equal risk contribution allocation"""
        optimizer = RiskBudgetOptimizer()
        
        # Use recent data
        if len(data) > self.lookback_period:
            recent_data = data.tail(self.lookback_period)
        else:
            recent_data = data
        
        returns = recent_data.pct_change().dropna()
        
        if len(returns) < 30:
            n_assets = len(data.columns)
            return {asset: 1.0 / n_assets for asset in data.columns}
        
        try:
            result = optimizer.optimize(returns)
            weights = dict(zip(data.columns, result['weights']))
            return self.validate_allocation(weights)
        except:
            n_assets = len(data.columns)
            return {asset: 1.0 / n_assets for asset in data.columns}
    
    def _inverse_volatility_allocation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Inverse volatility allocation"""
        
        # Use recent data
        if len(data) > self.lookback_period:
            recent_data = data.tail(self.lookback_period)
        else:
            recent_data = data
        
        returns = recent_data.pct_change().dropna()
        
        if len(returns) < 30:
            n_assets = len(data.columns)
            return {asset: 1.0 / n_assets for asset in data.columns}
        
        # Calculate volatilities
        volatilities = returns.std()
        
        # Inverse volatility weights
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return dict(zip(data.columns, weights))
    
    def _minimum_variance_allocation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Minimum variance allocation"""
        
        # Use recent data
        if len(data) > self.lookback_period:
            recent_data = data.tail(self.lookback_period)
        else:
            recent_data = data
        
        returns = recent_data.pct_change().dropna()
        
        if len(returns) < 30:
            n_assets = len(data.columns)
            return {asset: 1.0 / n_assets for asset in data.columns}
        
        try:
            # Use efficient frontier to find minimum variance portfolio
            frontier = EfficientFrontier(n_points=1)
            cov = returns.cov().values
            
            # Calculate minimum variance weights directly
            n = cov.shape[0]
            ones = np.ones((n, 1))
            
            try:
                cov_inv = np.linalg.inv(cov)
                weights = cov_inv @ ones / (ones.T @ cov_inv @ ones)
                weights = weights.flatten()
                
                # Ensure non-negative weights
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
                
                return dict(zip(data.columns, weights))
            except np.linalg.LinAlgError:
                # Fallback to equal weights
                n_assets = len(data.columns)
                return {asset: 1.0 / n_assets for asset in data.columns}
                
        except Exception as e:
            self.logger.error(f"Minimum variance calculation failed: {e}")
            n_assets = len(data.columns)
            return {asset: 1.0 / n_assets for asset in data.columns}


class AdaptiveAllocator(BaseAllocator):
    """Adaptive allocation that switches between strategies based on market conditions"""
    
    def __init__(self, strategies: Dict[str, BaseAllocator],
                 regime_detector: Optional[Callable] = None):
        super().__init__("AdaptiveAllocation")
        self.strategies = strategies
        self.regime_detector = regime_detector or self._default_regime_detector
        
    def allocate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Generate adaptive allocation based on market regime"""
        
        # Detect current market regime
        regime = self.regime_detector(data)
        
        # Select appropriate strategy
        if regime in self.strategies:
            strategy = self.strategies[regime]
            self.logger.info(f"Using {strategy.name} for regime: {regime}")
            return strategy.allocate(data, **kwargs)
        else:
            # Default to first strategy
            default_strategy = list(self.strategies.values())[0]
            self.logger.warning(f"Unknown regime {regime}, using default strategy: {default_strategy.name}")
            return default_strategy.allocate(data, **kwargs)
    
    def _default_regime_detector(self, data: pd.DataFrame) -> str:
        """Simple regime detection based on volatility"""
        
        if len(data) < 60:  # Need at least 60 days
            return 'normal'
        
        # Calculate rolling volatility
        returns = data.pct_change().dropna()
        recent_vol = returns.tail(30).std().mean()
        long_term_vol = returns.std().mean()
        
        vol_ratio = recent_vol / long_term_vol
        
        if vol_ratio > 1.5:
            return 'high_volatility'
        elif vol_ratio < 0.7:
            return 'low_volatility'
        else:
            return 'normal'


class AllocationManager:
    """Manager class for coordinating different allocation strategies"""
    
    def __init__(self):
        self.logger = Logger.get_logger("allocation_manager")
        self.allocators = {}
        
    def add_allocator(self, name: str, allocator: BaseAllocator):
        """Add an allocator to the manager"""
        self.allocators[name] = allocator
        self.logger.info(f"Added allocator: {name}")
    
    def get_allocation(self, allocator_name: str, data: pd.DataFrame, 
                     **kwargs) -> Dict[str, float]:
        """Get allocation from specific allocator"""
        if allocator_name not in self.allocators:
            raise ValueError(f"Allocator {allocator_name} not found")
        
        return self.allocators[allocator_name].allocate(data, **kwargs)
    
    def compare_allocations(self, data: pd.DataFrame, 
                          allocator_names: Optional[List[str]] = None,
                          **kwargs) -> pd.DataFrame:
        """Compare allocations from multiple allocators"""
        
        if allocator_names is None:
            allocator_names = list(self.allocators.keys())
        
        results = {}
        
        for name in allocator_names:
            if name in self.allocators:
                try:
                    allocation = self.get_allocation(name, data, **kwargs)
                    results[name] = allocation
                except Exception as e:
                    self.logger.error(f"Error getting allocation from {name}: {e}")
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results).fillna(0)
            return df
        else:
            return pd.DataFrame()