#!/usr/bin/env python3
"""
Strategy Risk Management Module

Provides comprehensive risk management for trading strategies:
- Position sizing and risk limits
- Stop-loss and take-profit management
- Portfolio-level risk controls
- Dynamic risk adjustment
- Risk monitoring and alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from quantlib.core.utils import Logger
from quantlib.strategy.base import StrategySignal, SignalType, PositionType


class RiskLimitType(Enum):
    """Types of risk limits"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_VAR = "portfolio_var"
    DRAWDOWN = "drawdown"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    VOLATILITY = "volatility"


class RiskAction(Enum):
    """Risk management actions"""
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    CLOSE = "close"
    ALERT = "alert"


@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_type: RiskLimitType
    threshold: float
    action: RiskAction
    enabled: bool = True
    metadata: Optional[Dict] = None


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    portfolio_value: float
    total_exposure: float
    leverage: float
    var_1d: Optional[float] = None
    var_5d: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    concentration_risk: Optional[Dict[str, float]] = None
    position_sizes: Optional[Dict[str, float]] = None


@dataclass
class RiskAdjustment:
    """Risk adjustment result"""
    original_signal: StrategySignal
    adjusted_signal: Optional[StrategySignal]
    action_taken: RiskAction
    reason: str
    risk_metrics: Optional[RiskMetrics] = None


class BaseRiskManager(ABC):
    """Base class for risk managers"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger.get_logger(f"risk_{name.lower()}")
        self.limits = []
        self.enabled = True
    
    def add_limit(self, limit: RiskLimit):
        """Add a risk limit"""
        self.limits.append(limit)
        self.logger.info(f"Added {limit.limit_type.value} limit: {limit.threshold}")
    
    def remove_limit(self, limit_type: RiskLimitType):
        """Remove a risk limit"""
        self.limits = [l for l in self.limits if l.limit_type != limit_type]
        self.logger.info(f"Removed {limit_type.value} limit")
    
    @abstractmethod
    def assess_risk(self, signal: StrategySignal, 
                   portfolio_data: Dict, 
                   market_data: pd.DataFrame) -> RiskAdjustment:
        """Assess risk and adjust signal if necessary"""
        pass
    
    def calculate_risk_metrics(self, portfolio_data: Dict, 
                             market_data: pd.DataFrame) -> RiskMetrics:
        """Calculate current risk metrics"""
        # Extract portfolio information
        positions = portfolio_data.get('positions', {})
        portfolio_value = portfolio_data.get('total_value', 0.0)
        
        # Calculate basic metrics
        total_exposure = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Position sizes as percentage of portfolio
        position_sizes = {}
        for asset, pos in positions.items():
            market_value = abs(pos.get('market_value', 0))
            position_sizes[asset] = market_value / portfolio_value if portfolio_value > 0 else 0
        
        # Concentration risk (largest positions)
        concentration_risk = dict(sorted(position_sizes.items(), 
                                       key=lambda x: x[1], reverse=True)[:5])
        
        # Calculate VaR and volatility if we have price data
        var_1d = None
        var_5d = None
        volatility = None
        
        if not market_data.empty and len(market_data) > 20:
            returns = market_data.pct_change().dropna()
            if not returns.empty:
                # Portfolio returns (simplified)
                portfolio_returns = returns.mean(axis=1)  # Equal weight for simplicity
                
                # VaR calculation (95% confidence)
                var_1d = np.percentile(portfolio_returns, 5)
                var_5d = var_1d * np.sqrt(5)
                
                # Volatility
                volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Drawdown calculation
        max_drawdown = None
        current_drawdown = None
        
        if 'equity_curve' in portfolio_data:
            equity_curve = portfolio_data['equity_curve']
            if len(equity_curve) > 1:
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - peak) / peak
                max_drawdown = drawdown.min()
                current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            leverage=leverage,
            var_1d=var_1d,
            var_5d=var_5d,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            concentration_risk=concentration_risk,
            position_sizes=position_sizes
        )


class PositionSizeManager(BaseRiskManager):
    """Manages position sizing based on risk"""
    
    def __init__(self, max_position_size: float = 0.1, 
                 volatility_target: float = 0.15):
        super().__init__("PositionSizing")
        self.max_position_size = max_position_size
        self.volatility_target = volatility_target
        
        # Add default limits
        self.add_limit(RiskLimit(
            RiskLimitType.POSITION_SIZE, 
            max_position_size, 
            RiskAction.REDUCE
        ))
    
    def assess_risk(self, signal: StrategySignal, 
                   portfolio_data: Dict, 
                   market_data: pd.DataFrame) -> RiskAdjustment:
        """Assess and adjust position size"""
        if not self.enabled:
            return RiskAdjustment(signal, signal, RiskAction.ALLOW, "Risk manager disabled")
        
        risk_metrics = self.calculate_risk_metrics(portfolio_data, market_data)
        
        # Calculate optimal position size
        optimal_size = self._calculate_optimal_position_size(
            signal, portfolio_data, market_data, risk_metrics
        )
        
        # Check against limits
        for limit in self.limits:
            if not limit.enabled:
                continue
                
            if limit.limit_type == RiskLimitType.POSITION_SIZE:
                if abs(optimal_size) > limit.threshold:
                    if limit.action == RiskAction.REDUCE:
                        optimal_size = np.sign(optimal_size) * limit.threshold
                        action = RiskAction.REDUCE
                        reason = f"Position size reduced to {limit.threshold:.2%} limit"
                    elif limit.action == RiskAction.BLOCK:
                        return RiskAdjustment(
                            signal, None, RiskAction.BLOCK,
                            f"Position size exceeds {limit.threshold:.2%} limit",
                            risk_metrics
                        )
        
        # Create adjusted signal
        if abs(optimal_size - signal.target_position) > 1e-6:
            adjusted_signal = StrategySignal(
                timestamp=signal.timestamp,
                asset=signal.asset,
                signal_type=signal.signal_type,
                strength=signal.strength,
                confidence=signal.confidence,
                target_position=optimal_size,
                metadata={**(signal.metadata or {}), 'risk_adjusted': True}
            )
            action = RiskAction.REDUCE
            reason = f"Position size adjusted from {signal.target_position:.2%} to {optimal_size:.2%}"
        else:
            adjusted_signal = signal
            action = RiskAction.ALLOW
            reason = "Position size within limits"
        
        return RiskAdjustment(signal, adjusted_signal, action, reason, risk_metrics)
    
    def _calculate_optimal_position_size(self, signal: StrategySignal,
                                       portfolio_data: Dict,
                                       market_data: pd.DataFrame,
                                       risk_metrics: RiskMetrics) -> float:
        """Calculate optimal position size based on volatility targeting"""
        asset = signal.asset
        
        # Get asset volatility
        asset_volatility = self._estimate_asset_volatility(asset, market_data)
        
        if asset_volatility is None or asset_volatility == 0:
            # Fallback to signal target if no volatility data
            return min(abs(signal.target_position), self.max_position_size) * np.sign(signal.target_position)
        
        # Volatility-based position sizing
        volatility_scalar = self.volatility_target / asset_volatility
        base_position = signal.target_position * volatility_scalar
        
        # Apply confidence adjustment
        confidence_adjusted = base_position * signal.confidence
        
        # Apply maximum position size limit
        final_position = np.sign(confidence_adjusted) * min(
            abs(confidence_adjusted), self.max_position_size
        )
        
        return final_position
    
    def _estimate_asset_volatility(self, asset: str, 
                                 market_data: pd.DataFrame) -> Optional[float]:
        """Estimate asset volatility"""
        if asset not in market_data.columns or len(market_data) < 20:
            return None
        
        returns = market_data[asset].pct_change().dropna()
        if len(returns) < 10:
            return None
        
        # Annualized volatility
        return returns.std() * np.sqrt(252)


class StopLossManager(BaseRiskManager):
    """Manages stop-loss and take-profit orders"""
    
    def __init__(self, default_stop_loss: float = 0.05,
                 default_take_profit: float = 0.10,
                 trailing_stop: bool = True):
        super().__init__("StopLoss")
        self.default_stop_loss = default_stop_loss
        self.default_take_profit = default_take_profit
        self.trailing_stop = trailing_stop
        self.stop_levels = {}  # Track stop levels for each position
    
    def assess_risk(self, signal: StrategySignal, 
                   portfolio_data: Dict, 
                   market_data: pd.DataFrame) -> RiskAdjustment:
        """Assess stop-loss conditions"""
        if not self.enabled:
            return RiskAdjustment(signal, signal, RiskAction.ALLOW, "Risk manager disabled")
        
        asset = signal.asset
        positions = portfolio_data.get('positions', {})
        
        # Check if we have a position in this asset
        if asset not in positions:
            return RiskAdjustment(signal, signal, RiskAction.ALLOW, "No existing position")
        
        position = positions[asset]
        current_price = self._get_current_price(asset, market_data)
        
        if current_price is None:
            return RiskAdjustment(signal, signal, RiskAction.ALLOW, "No price data")
        
        # Check stop-loss conditions
        stop_action = self._check_stop_conditions(asset, position, current_price)
        
        if stop_action == RiskAction.CLOSE:
            # Create close signal
            close_signal = StrategySignal(
                timestamp=signal.timestamp,
                asset=asset,
                signal_type=SignalType.SELL if position.get('quantity', 0) > 0 else SignalType.BUY,
                strength=1.0,
                confidence=1.0,
                target_position=0.0,
                metadata={'stop_loss_triggered': True, 'reason': 'stop_loss'}
            )
            
            return RiskAdjustment(
                signal, close_signal, RiskAction.CLOSE,
                f"Stop-loss triggered for {asset}"
            )
        
        # Update trailing stops if enabled
        if self.trailing_stop:
            self._update_trailing_stop(asset, position, current_price)
        
        return RiskAdjustment(signal, signal, RiskAction.ALLOW, "Stop-loss conditions OK")
    
    def _check_stop_conditions(self, asset: str, position: Dict, 
                             current_price: float) -> RiskAction:
        """Check if stop-loss should be triggered"""
        entry_price = position.get('avg_price', current_price)
        quantity = position.get('quantity', 0)
        
        if quantity == 0:
            return RiskAction.ALLOW
        
        # Calculate P&L percentage
        if quantity > 0:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop-loss
        if pnl_pct <= -self.default_stop_loss:
            self.logger.warning(f"Stop-loss triggered for {asset}: {pnl_pct:.2%}")
            return RiskAction.CLOSE
        
        # Check take-profit
        if pnl_pct >= self.default_take_profit:
            self.logger.info(f"Take-profit triggered for {asset}: {pnl_pct:.2%}")
            return RiskAction.CLOSE
        
        # Check trailing stop
        if asset in self.stop_levels:
            stop_level = self.stop_levels[asset]
            if quantity > 0 and current_price <= stop_level:
                self.logger.warning(f"Trailing stop triggered for {asset}")
                return RiskAction.CLOSE
            elif quantity < 0 and current_price >= stop_level:
                self.logger.warning(f"Trailing stop triggered for {asset}")
                return RiskAction.CLOSE
        
        return RiskAction.ALLOW
    
    def _update_trailing_stop(self, asset: str, position: Dict, current_price: float):
        """Update trailing stop level"""
        quantity = position.get('quantity', 0)
        
        if quantity == 0:
            return
        
        if asset not in self.stop_levels:
            # Initialize trailing stop
            if quantity > 0:  # Long position
                self.stop_levels[asset] = current_price * (1 - self.default_stop_loss)
            else:  # Short position
                self.stop_levels[asset] = current_price * (1 + self.default_stop_loss)
        else:
            # Update trailing stop
            current_stop = self.stop_levels[asset]
            
            if quantity > 0:  # Long position
                new_stop = current_price * (1 - self.default_stop_loss)
                self.stop_levels[asset] = max(current_stop, new_stop)
            else:  # Short position
                new_stop = current_price * (1 + self.default_stop_loss)
                self.stop_levels[asset] = min(current_stop, new_stop)
    
    def _get_current_price(self, asset: str, market_data: pd.DataFrame) -> Optional[float]:
        """Get current price for asset"""
        if asset not in market_data.columns or market_data.empty:
            return None
        
        return market_data[asset].iloc[-1]


class PortfolioRiskManager(BaseRiskManager):
    """Portfolio-level risk management"""
    
    def __init__(self, max_leverage: float = 2.0,
                 max_concentration: float = 0.2,
                 max_drawdown: float = 0.15):
        super().__init__("PortfolioRisk")
        
        # Add default limits
        self.add_limit(RiskLimit(RiskLimitType.LEVERAGE, max_leverage, RiskAction.BLOCK))
        self.add_limit(RiskLimit(RiskLimitType.CONCENTRATION, max_concentration, RiskAction.REDUCE))
        self.add_limit(RiskLimit(RiskLimitType.DRAWDOWN, max_drawdown, RiskAction.ALERT))
    
    def assess_risk(self, signal: StrategySignal, 
                   portfolio_data: Dict, 
                   market_data: pd.DataFrame) -> RiskAdjustment:
        """Assess portfolio-level risk"""
        if not self.enabled:
            return RiskAdjustment(signal, signal, RiskAction.ALLOW, "Risk manager disabled")
        
        risk_metrics = self.calculate_risk_metrics(portfolio_data, market_data)
        
        # Check each limit
        for limit in self.limits:
            if not limit.enabled:
                continue
            
            violation = self._check_limit_violation(limit, signal, risk_metrics)
            
            if violation:
                if limit.action == RiskAction.BLOCK:
                    return RiskAdjustment(
                        signal, None, RiskAction.BLOCK,
                        f"{limit.limit_type.value} limit exceeded: {violation}",
                        risk_metrics
                    )
                elif limit.action == RiskAction.REDUCE:
                    adjusted_signal = self._reduce_signal(signal, limit, risk_metrics)
                    return RiskAdjustment(
                        signal, adjusted_signal, RiskAction.REDUCE,
                        f"{limit.limit_type.value} limit: signal reduced",
                        risk_metrics
                    )
                elif limit.action == RiskAction.ALERT:
                    self.logger.warning(f"{limit.limit_type.value} limit exceeded: {violation}")
        
        return RiskAdjustment(signal, signal, RiskAction.ALLOW, "Portfolio risk OK", risk_metrics)
    
    def _check_limit_violation(self, limit: RiskLimit, 
                             signal: StrategySignal,
                             risk_metrics: RiskMetrics) -> Optional[str]:
        """Check if a limit is violated"""
        if limit.limit_type == RiskLimitType.LEVERAGE:
            if risk_metrics.leverage > limit.threshold:
                return f"Leverage {risk_metrics.leverage:.2f} > {limit.threshold:.2f}"
        
        elif limit.limit_type == RiskLimitType.CONCENTRATION:
            if risk_metrics.concentration_risk:
                max_concentration = max(risk_metrics.concentration_risk.values())
                if max_concentration > limit.threshold:
                    return f"Max concentration {max_concentration:.2%} > {limit.threshold:.2%}"
        
        elif limit.limit_type == RiskLimitType.DRAWDOWN:
            if risk_metrics.current_drawdown and abs(risk_metrics.current_drawdown) > limit.threshold:
                return f"Drawdown {risk_metrics.current_drawdown:.2%} > {limit.threshold:.2%}"
        
        return None
    
    def _reduce_signal(self, signal: StrategySignal, 
                      limit: RiskLimit,
                      risk_metrics: RiskMetrics) -> StrategySignal:
        """Reduce signal to comply with limit"""
        reduction_factor = 0.5  # Simple reduction
        
        if limit.limit_type == RiskLimitType.CONCENTRATION:
            # Reduce position size to stay under concentration limit
            current_concentration = risk_metrics.position_sizes.get(signal.asset, 0)
            if current_concentration > 0:
                max_additional = limit.threshold - current_concentration
                reduction_factor = max(0.1, max_additional / abs(signal.target_position))
        
        adjusted_position = signal.target_position * reduction_factor
        
        return StrategySignal(
            timestamp=signal.timestamp,
            asset=signal.asset,
            signal_type=signal.signal_type,
            strength=signal.strength * reduction_factor,
            confidence=signal.confidence,
            target_position=adjusted_position,
            metadata={**(signal.metadata or {}), 'risk_reduced': True}
        )


class RiskManagerComposite:
    """Composite risk manager that combines multiple risk managers"""
    
    def __init__(self, managers: List[BaseRiskManager]):
        self.managers = managers
        self.logger = Logger.get_logger("risk_composite")
    
    def assess_risk(self, signal: StrategySignal, 
                   portfolio_data: Dict, 
                   market_data: pd.DataFrame) -> RiskAdjustment:
        """Assess risk through all managers"""
        current_signal = signal
        adjustments = []
        
        for manager in self.managers:
            if not manager.enabled:
                continue
            
            try:
                adjustment = manager.assess_risk(current_signal, portfolio_data, market_data)
                adjustments.append(adjustment)
                
                # If signal is blocked or closed, stop processing
                if adjustment.action in [RiskAction.BLOCK, RiskAction.CLOSE]:
                    return adjustment
                
                # Update current signal for next manager
                if adjustment.adjusted_signal:
                    current_signal = adjustment.adjusted_signal
                    
            except Exception as e:
                self.logger.error(f"Error in {manager.name}: {e}")
        
        # Combine all adjustments
        if adjustments:
            final_adjustment = adjustments[-1]
            final_adjustment.metadata = {
                'adjustments': [adj.reason for adj in adjustments],
                'managers_applied': [adj.original_signal.__class__.__name__ for adj in adjustments]
            }
            return final_adjustment
        
        return RiskAdjustment(signal, signal, RiskAction.ALLOW, "No risk managers applied")