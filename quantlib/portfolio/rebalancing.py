#!/usr/bin/env python3
"""
Portfolio Rebalancing Module

Implements various portfolio rebalancing strategies including:
- Calendar-based rebalancing
- Threshold-based rebalancing
- Volatility-based rebalancing
- Transaction cost optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass

from quantlib.core.utils import Logger


@dataclass
class RebalanceTransaction:
    """Represents a rebalancing transaction"""
    asset: str
    current_weight: float
    target_weight: float
    weight_change: float
    transaction_cost: float = 0.0
    
    @property
    def trade_amount(self) -> float:
        """Amount to trade (positive = buy, negative = sell)"""
        return self.weight_change


@dataclass
class RebalanceResult:
    """Results of a rebalancing operation"""
    timestamp: datetime
    transactions: List[RebalanceTransaction]
    total_cost: float
    turnover: float
    rebalance_needed: bool
    
    @property
    def net_trades(self) -> Dict[str, float]:
        """Net trades by asset"""
        return {t.asset: t.trade_amount for t in self.transactions}


class BaseRebalancer(ABC):
    """Base class for portfolio rebalancers"""
    
    def __init__(self, name: str, transaction_cost: float = 0.001):
        self.name = name
        self.transaction_cost = transaction_cost  # As fraction of trade value
        self.logger = Logger.get_logger(f"rebalancer_{name.lower()}")
        
    @abstractmethod
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        **kwargs) -> bool:
        """Determine if rebalancing is needed"""
        pass
    
    def rebalance(self, current_weights: Dict[str, float],
                 target_weights: Dict[str, float],
                 portfolio_value: float = 1.0,
                 **kwargs) -> RebalanceResult:
        """Execute rebalancing if needed"""
        
        timestamp = kwargs.get('timestamp', datetime.now())
        
        # Check if rebalancing is needed
        rebalance_needed = self.should_rebalance(current_weights, target_weights, **kwargs)
        
        if not rebalance_needed:
            return RebalanceResult(
                timestamp=timestamp,
                transactions=[],
                total_cost=0.0,
                turnover=0.0,
                rebalance_needed=False
            )
        
        # Calculate transactions
        transactions = self._calculate_transactions(
            current_weights, target_weights, portfolio_value
        )
        
        # Calculate costs and turnover
        total_cost = sum(t.transaction_cost for t in transactions)
        turnover = sum(abs(t.weight_change) for t in transactions) / 2  # Divide by 2 to avoid double counting
        
        self.logger.info(f"Rebalancing: {len(transactions)} transactions, "
                        f"turnover: {turnover:.4f}, cost: {total_cost:.6f}")
        
        return RebalanceResult(
            timestamp=timestamp,
            transactions=transactions,
            total_cost=total_cost,
            turnover=turnover,
            rebalance_needed=True
        )
    
    def _calculate_transactions(self, current_weights: Dict[str, float],
                              target_weights: Dict[str, float],
                              portfolio_value: float) -> List[RebalanceTransaction]:
        """Calculate required transactions"""
        transactions = []
        
        # Get all assets
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            weight_change = target_weight - current_weight
            
            if abs(weight_change) > 1e-6:  # Only create transaction if meaningful change
                trade_value = abs(weight_change) * portfolio_value
                transaction_cost = trade_value * self.transaction_cost
                
                transaction = RebalanceTransaction(
                    asset=asset,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    weight_change=weight_change,
                    transaction_cost=transaction_cost
                )
                transactions.append(transaction)
        
        return transactions


class CalendarRebalancer(BaseRebalancer):
    """Calendar-based rebalancing (monthly, quarterly, etc.)"""
    
    def __init__(self, frequency: str = 'quarterly', 
                 transaction_cost: float = 0.001):
        super().__init__(f"Calendar_{frequency}", transaction_cost)
        self.frequency = frequency
        self.last_rebalance = None
        
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        timestamp: Optional[datetime] = None,
                        **kwargs) -> bool:
        """Check if enough time has passed for rebalancing"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.last_rebalance is None:
            self.last_rebalance = timestamp
            return True
        
        days_since_rebalance = (timestamp - self.last_rebalance).days
        
        if self.frequency == 'daily':
            threshold = 1
        elif self.frequency == 'weekly':
            threshold = 7
        elif self.frequency == 'monthly':
            threshold = 30
        elif self.frequency == 'quarterly':
            threshold = 90
        elif self.frequency == 'annually':
            threshold = 365
        else:
            raise ValueError(f"Unknown frequency: {self.frequency}")
        
        if days_since_rebalance >= threshold:
            self.last_rebalance = timestamp
            return True
        
        return False


class ThresholdRebalancer(BaseRebalancer):
    """Threshold-based rebalancing (rebalance when weights deviate too much)"""
    
    def __init__(self, threshold: float = 0.05, 
                 transaction_cost: float = 0.001):
        super().__init__(f"Threshold_{threshold}", transaction_cost)
        self.threshold = threshold
        
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        **kwargs) -> bool:
        """Check if any weight deviates beyond threshold"""
        
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > self.threshold:
                self.logger.debug(f"Asset {asset} deviation: {deviation:.4f} > {self.threshold}")
                return True
        
        return False


class VolatilityRebalancer(BaseRebalancer):
    """Volatility-based rebalancing (rebalance more frequently in high volatility periods)"""
    
    def __init__(self, base_threshold: float = 0.05,
                 volatility_multiplier: float = 2.0,
                 lookback_period: int = 30,
                 transaction_cost: float = 0.001):
        super().__init__("Volatility", transaction_cost)
        self.base_threshold = base_threshold
        self.volatility_multiplier = volatility_multiplier
        self.lookback_period = lookback_period
        
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        returns_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> bool:
        """Check if rebalancing needed based on volatility-adjusted threshold"""
        
        # Calculate dynamic threshold based on volatility
        if returns_data is not None and len(returns_data) >= self.lookback_period:
            recent_returns = returns_data.tail(self.lookback_period)
            portfolio_returns = self._calculate_portfolio_returns(
                recent_returns, current_weights
            )
            current_volatility = portfolio_returns.std()
            
            # Adjust threshold based on volatility
            # Higher volatility -> lower threshold (more frequent rebalancing)
            volatility_factor = min(self.volatility_multiplier, 
                                  1 / (1 + current_volatility * 10))
            dynamic_threshold = self.base_threshold * volatility_factor
        else:
            dynamic_threshold = self.base_threshold
        
        # Check deviations against dynamic threshold
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > dynamic_threshold:
                self.logger.debug(f"Asset {asset} deviation: {deviation:.4f} > {dynamic_threshold:.4f}")
                return True
        
        return False
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame,
                                   weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns given weights"""
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        
        for asset, weight in weights.items():
            if asset in returns_data.columns:
                portfolio_returns += weight * returns_data[asset]
        
        return portfolio_returns


class CostOptimizedRebalancer(BaseRebalancer):
    """Cost-optimized rebalancing that considers transaction costs"""
    
    def __init__(self, cost_threshold: float = 0.001,
                 max_deviation: float = 0.10,
                 transaction_cost: float = 0.001):
        super().__init__("CostOptimized", transaction_cost)
        self.cost_threshold = cost_threshold
        self.max_deviation = max_deviation
        
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        portfolio_value: float = 1.0,
                        **kwargs) -> bool:
        """Check if rebalancing benefits outweigh costs"""
        
        # Calculate potential transaction costs
        transactions = self._calculate_transactions(
            current_weights, target_weights, portfolio_value
        )
        total_cost = sum(t.transaction_cost for t in transactions)
        
        # Calculate maximum deviation
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        max_deviation = 0.0
        
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            deviation = abs(current_weight - target_weight)
            max_deviation = max(max_deviation, deviation)
        
        # Rebalance if:
        # 1. Cost is below threshold, OR
        # 2. Deviation is above maximum allowed
        should_rebalance = (total_cost < self.cost_threshold or 
                           max_deviation > self.max_deviation)
        
        if should_rebalance:
            self.logger.debug(f"Rebalancing: cost={total_cost:.6f}, "
                            f"max_deviation={max_deviation:.4f}")
        
        return should_rebalance


class BandRebalancer(BaseRebalancer):
    """Band-based rebalancing with upper and lower bounds for each asset"""
    
    def __init__(self, bands: Dict[str, Tuple[float, float]],
                 transaction_cost: float = 0.001):
        super().__init__("Band", transaction_cost)
        self.bands = bands  # {asset: (lower_bound, upper_bound)}
        
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        **kwargs) -> bool:
        """Check if any weight is outside its band"""
        
        for asset, (lower_bound, upper_bound) in self.bands.items():
            current_weight = current_weights.get(asset, 0.0)
            
            if current_weight < lower_bound or current_weight > upper_bound:
                self.logger.debug(f"Asset {asset} outside band: {current_weight:.4f} "
                                f"not in [{lower_bound:.4f}, {upper_bound:.4f}]")
                return True
        
        return False


class RebalancingManager:
    """Manager class for coordinating rebalancing strategies"""
    
    def __init__(self):
        self.logger = Logger.get_logger("rebalancing_manager")
        self.rebalancers = {}
        self.rebalance_history = []
        
    def add_rebalancer(self, name: str, rebalancer: BaseRebalancer):
        """Add a rebalancer to the manager"""
        self.rebalancers[name] = rebalancer
        self.logger.info(f"Added rebalancer: {name}")
    
    def execute_rebalancing(self, rebalancer_name: str,
                          current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          portfolio_value: float = 1.0,
                          **kwargs) -> RebalanceResult:
        """Execute rebalancing using specified rebalancer"""
        
        if rebalancer_name not in self.rebalancers:
            raise ValueError(f"Rebalancer {rebalancer_name} not found")
        
        rebalancer = self.rebalancers[rebalancer_name]
        result = rebalancer.rebalance(
            current_weights, target_weights, portfolio_value, **kwargs
        )
        
        # Store in history
        self.rebalance_history.append(result)
        
        return result
    
    def get_rebalance_history(self, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[RebalanceResult]:
        """Get rebalancing history within date range"""
        
        filtered_history = self.rebalance_history
        
        if start_date:
            filtered_history = [r for r in filtered_history if r.timestamp >= start_date]
        
        if end_date:
            filtered_history = [r for r in filtered_history if r.timestamp <= end_date]
        
        return filtered_history
    
    def analyze_rebalancing_costs(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, float]:
        """Analyze rebalancing costs over time"""
        
        history = self.get_rebalance_history(start_date, end_date)
        
        if not history:
            return {}
        
        total_cost = sum(r.total_cost for r in history)
        total_turnover = sum(r.turnover for r in history)
        avg_cost_per_rebalance = total_cost / len(history)
        avg_turnover_per_rebalance = total_turnover / len(history)
        
        return {
            'total_cost': total_cost,
            'total_turnover': total_turnover,
            'average_cost_per_rebalance': avg_cost_per_rebalance,
            'average_turnover_per_rebalance': avg_turnover_per_rebalance,
            'number_of_rebalances': len(history)
        }
    
    def simulate_rebalancing_strategies(self, 
                                      current_weights: Dict[str, float],
                                      target_weights: Dict[str, float],
                                      portfolio_value: float = 1.0,
                                      **kwargs) -> Dict[str, RebalanceResult]:
        """Simulate all rebalancing strategies"""
        
        results = {}
        
        for name, rebalancer in self.rebalancers.items():
            try:
                result = rebalancer.rebalance(
                    current_weights, target_weights, portfolio_value, **kwargs
                )
                results[name] = result
            except Exception as e:
                self.logger.error(f"Error simulating {name}: {e}")
        
        return results