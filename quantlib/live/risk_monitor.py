"""Real-time risk monitoring system.

This module provides comprehensive risk monitoring for live trading:
- Position limits
- Loss limits
- Exposure monitoring
- Risk metrics calculation
- Alert generation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..backtesting.events import OrderEvent, FillEvent
from ..analytics.risk import VaRCalculator, RiskMetrics
from ..core.base import BaseComponent


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert type enumeration."""
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    VAR_LIMIT = "var_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"
    VOLATILITY_SPIKE = "volatility_spike"
    ORDER_SIZE = "order_size"
    TRADING_FREQUENCY = "trading_frequency"


@dataclass
class RiskAlert:
    """Risk alert."""
    alert_id: str
    alert_type: AlertType
    level: RiskLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged
        }


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Position limits
    max_position_size: float = 0.1  # 10% of portfolio
    max_symbol_exposure: float = 0.2  # 20% per symbol
    max_sector_exposure: float = 0.3  # 30% per sector
    
    # Loss limits
    max_daily_loss: float = 0.05  # 5% daily loss
    max_weekly_loss: float = 0.10  # 10% weekly loss
    max_monthly_loss: float = 0.15  # 15% monthly loss
    max_drawdown: float = 0.20  # 20% max drawdown
    
    # Risk metrics limits
    max_var_95: float = 0.03  # 3% VaR at 95% confidence
    max_var_99: float = 0.05  # 5% VaR at 99% confidence
    max_portfolio_volatility: float = 0.25  # 25% annualized volatility
    
    # Trading limits
    max_order_size: float = 50000.0  # $50k per order
    max_orders_per_minute: int = 10
    max_orders_per_hour: int = 100
    max_orders_per_day: int = 500
    
    # Concentration limits
    max_correlation: float = 0.8  # Maximum correlation between positions
    min_diversification_ratio: float = 0.5  # Minimum diversification
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert limits to dictionary."""
        return {
            'max_position_size': self.max_position_size,
            'max_symbol_exposure': self.max_symbol_exposure,
            'max_sector_exposure': self.max_sector_exposure,
            'max_daily_loss': self.max_daily_loss,
            'max_weekly_loss': self.max_weekly_loss,
            'max_monthly_loss': self.max_monthly_loss,
            'max_drawdown': self.max_drawdown,
            'max_var_95': self.max_var_95,
            'max_var_99': self.max_var_99,
            'max_portfolio_volatility': self.max_portfolio_volatility,
            'max_order_size': self.max_order_size,
            'max_orders_per_minute': self.max_orders_per_minute,
            'max_orders_per_hour': self.max_orders_per_hour,
            'max_orders_per_day': self.max_orders_per_day,
            'max_correlation': self.max_correlation,
            'min_diversification_ratio': self.min_diversification_ratio
        }


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Portfolio metrics
    total_value: float = 0.0
    total_exposure: float = 0.0
    cash_balance: float = 0.0
    leverage: float = 0.0
    
    # Position metrics
    num_positions: int = 0
    largest_position_pct: float = 0.0
    concentration_ratio: float = 0.0
    
    # Risk metrics
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    monthly_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # VaR metrics
    var_95_1d: float = 0.0
    var_99_1d: float = 0.0
    portfolio_volatility: float = 0.0
    
    # Trading metrics
    orders_last_minute: int = 0
    orders_last_hour: int = 0
    orders_today: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'total_exposure': self.total_exposure,
            'cash_balance': self.cash_balance,
            'leverage': self.leverage,
            'num_positions': self.num_positions,
            'largest_position_pct': self.largest_position_pct,
            'concentration_ratio': self.concentration_ratio,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'weekly_pnl_pct': self.weekly_pnl_pct,
            'monthly_pnl_pct': self.monthly_pnl_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'current_drawdown_pct': self.current_drawdown_pct,
            'var_95_1d': self.var_95_1d,
            'var_99_1d': self.var_99_1d,
            'portfolio_volatility': self.portfolio_volatility,
            'orders_last_minute': self.orders_last_minute,
            'orders_last_hour': self.orders_last_hour,
            'orders_today': self.orders_today
        }


class RiskMonitor(BaseComponent):
    """Real-time risk monitoring system."""
    
    def __init__(self, limits: RiskLimits, initial_capital: float = 100000.0, name: str = "RiskMonitor"):
        super().__init__(name)
        self.limits = limits
        self.initial_capital = initial_capital
        self.current_metrics = RiskMetrics()
        self.var_calculator = VaRCalculator()
        self.risk_metrics_calc = RiskMetrics()
        
        # Alert management
        self.alerts: Dict[str, RiskAlert] = {}
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        
        # Historical data
        self.portfolio_values: List[float] = [initial_capital]
        self.portfolio_timestamps: List[datetime] = [datetime.now()]
        self.returns: List[float] = []
        self.positions: Dict[str, float] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Monitoring state
        self._monitoring = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Risk calculation cache
        self._last_risk_calc = datetime.now()
        self._risk_calc_interval = timedelta(minutes=1)
    
    async def start(self) -> bool:
        """Start risk monitoring."""
        try:
            self._monitoring = True
            
            # Start monitoring tasks
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._risk_calculation_loop())
            
            self.logger.info("Risk monitor started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start risk monitor: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop risk monitoring."""
        self._monitoring = False
        self.logger.info("Risk monitor stopped")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    async def check_order(self, order_event: OrderEvent) -> bool:
        """Check if order passes risk checks."""
        try:
            # Check order size
            order_value = order_event.quantity * getattr(order_event, 'price', 100.0)
            if order_value > self.limits.max_order_size:
                await self._create_alert(
                    AlertType.ORDER_SIZE,
                    RiskLevel.HIGH,
                    f"Order size ${order_value:,.2f} exceeds limit ${self.limits.max_order_size:,.2f}",
                    symbol=order_event.symbol,
                    current_value=order_value,
                    limit_value=self.limits.max_order_size
                )
                return False
            
            # Check trading frequency
            if not self._check_trading_frequency():
                await self._create_alert(
                    AlertType.TRADING_FREQUENCY,
                    RiskLevel.MEDIUM,
                    "Trading frequency limits exceeded",
                    symbol=order_event.symbol
                )
                return False
            
            # Check position limits (simplified)
            current_position = self.positions.get(order_event.symbol, 0.0)
            new_position_value = abs(current_position + order_value)
            position_pct = new_position_value / max(self.current_metrics.total_value, 1)
            
            if position_pct > self.limits.max_position_size:
                await self._create_alert(
                    AlertType.POSITION_LIMIT,
                    RiskLevel.HIGH,
                    f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_size:.1%}",
                    symbol=order_event.symbol,
                    current_value=position_pct,
                    limit_value=self.limits.max_position_size
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking order: {e}")
            return False
    
    async def check_limits(self) -> List[RiskAlert]:
        """Check all risk limits and generate alerts."""
        alerts = []
        
        try:
            # Update current metrics
            await self._update_metrics()
            
            # Check loss limits
            alerts.extend(await self._check_loss_limits())
            
            # Check position limits
            alerts.extend(await self._check_position_limits())
            
            # Check risk metrics
            alerts.extend(await self._check_risk_metrics())
            
            # Check concentration risk
            alerts.extend(await self._check_concentration_risk())
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking limits: {e}")
            return []
    
    def update_portfolio(self, total_value: float, positions: Dict[str, float], cash: float) -> None:
        """Update portfolio information."""
        self.portfolio_values.append(total_value)
        self.portfolio_timestamps.append(datetime.now())
        self.positions = positions.copy()
        
        # Calculate returns
        if len(self.portfolio_values) > 1:
            returns = (total_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns.append(returns)
        
        # Keep only recent data (last 252 trading days)
        if len(self.portfolio_values) > 252:
            self.portfolio_values = self.portfolio_values[-252:]
            self.portfolio_timestamps = self.portfolio_timestamps[-252:]
            self.returns = self.returns[-251:]
        
        # Update current metrics
        self.current_metrics.total_value = total_value
        self.current_metrics.cash_balance = cash
        self.current_metrics.total_exposure = sum(abs(pos) for pos in positions.values())
        self.current_metrics.leverage = self.current_metrics.total_exposure / max(total_value, 1)
        self.current_metrics.num_positions = len([p for p in positions.values() if p != 0])
    
    def update_fill(self, fill_event: FillEvent) -> None:
        """Update with fill information."""
        # Record order
        self.order_history.append({
            'timestamp': fill_event.timestamp,
            'symbol': fill_event.symbol,
            'quantity': fill_event.quantity,
            'price': fill_event.fill_cost,
            'direction': fill_event.direction
        })
        
        # Keep only recent orders (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.order_history = [
            order for order in self.order_history
            if order['timestamp'] > cutoff_time
        ]
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Check limits every 30 seconds
                await self.check_limits()
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _risk_calculation_loop(self) -> None:
        """Risk calculation loop."""
        while self._monitoring:
            try:
                # Calculate risk metrics every minute
                if datetime.now() - self._last_risk_calc > self._risk_calc_interval:
                    await self._calculate_risk_metrics()
                    self._last_risk_calc = datetime.now()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in risk calculation loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self) -> None:
        """Update current risk metrics."""
        try:
            # Update trading frequency metrics
            now = datetime.now()
            
            # Count orders in different time windows
            self.current_metrics.orders_last_minute = len([
                order for order in self.order_history
                if order['timestamp'] > now - timedelta(minutes=1)
            ])
            
            self.current_metrics.orders_last_hour = len([
                order for order in self.order_history
                if order['timestamp'] > now - timedelta(hours=1)
            ])
            
            self.current_metrics.orders_today = len([
                order for order in self.order_history
                if order['timestamp'].date() == now.date()
            ])
            
            # Update position metrics
            if self.positions:
                position_values = [abs(pos) for pos in self.positions.values()]
                self.current_metrics.largest_position_pct = max(position_values) / max(self.current_metrics.total_value, 1)
                
                # Calculate concentration ratio (HHI)
                total_exposure = sum(position_values)
                if total_exposure > 0:
                    weights = [pos / total_exposure for pos in position_values]
                    self.current_metrics.concentration_ratio = sum(w**2 for w in weights)
            
            # Update P&L metrics
            if len(self.portfolio_values) > 1:
                # Daily P&L
                self.current_metrics.daily_pnl = self.portfolio_values[-1] - self.portfolio_values[-2]
                self.current_metrics.daily_pnl_pct = self.current_metrics.daily_pnl / self.portfolio_values[-2]
                
                # Weekly P&L (if we have enough data)
                if len(self.portfolio_values) >= 5:
                    week_start_value = self.portfolio_values[-5]
                    self.current_metrics.weekly_pnl_pct = (self.portfolio_values[-1] - week_start_value) / week_start_value
                
                # Monthly P&L (if we have enough data)
                if len(self.portfolio_values) >= 21:
                    month_start_value = self.portfolio_values[-21]
                    self.current_metrics.monthly_pnl_pct = (self.portfolio_values[-1] - month_start_value) / month_start_value
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    async def _calculate_risk_metrics(self) -> None:
        """Calculate advanced risk metrics."""
        try:
            if len(self.returns) < 10:
                return
            
            returns_array = np.array(self.returns)
            
            # Calculate VaR
            var_results = self.var_calculator.historical_var(returns_array, [0.95, 0.99])
            self.current_metrics.var_95_1d = abs(var_results.get(0.95, 0.0))
            self.current_metrics.var_99_1d = abs(var_results.get(0.99, 0.0))
            
            # Calculate portfolio volatility (annualized)
            self.current_metrics.portfolio_volatility = np.std(returns_array) * np.sqrt(252)
            
            # Calculate drawdown
            if len(self.portfolio_values) > 1:
                values_array = np.array(self.portfolio_values)
                peak = np.maximum.accumulate(values_array)
                drawdown = (values_array - peak) / peak
                
                self.current_metrics.max_drawdown_pct = abs(np.min(drawdown))
                self.current_metrics.current_drawdown_pct = abs(drawdown[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
    
    async def _check_loss_limits(self) -> List[RiskAlert]:
        """Check loss limits."""
        alerts = []
        
        try:
            # Daily loss limit
            if self.current_metrics.daily_pnl_pct < -self.limits.max_daily_loss:
                alert = await self._create_alert(
                    AlertType.LOSS_LIMIT,
                    RiskLevel.CRITICAL,
                    f"Daily loss {self.current_metrics.daily_pnl_pct:.1%} exceeds limit {self.limits.max_daily_loss:.1%}",
                    current_value=abs(self.current_metrics.daily_pnl_pct),
                    limit_value=self.limits.max_daily_loss
                )
                alerts.append(alert)
            
            # Weekly loss limit
            if self.current_metrics.weekly_pnl_pct < -self.limits.max_weekly_loss:
                alert = await self._create_alert(
                    AlertType.LOSS_LIMIT,
                    RiskLevel.CRITICAL,
                    f"Weekly loss {self.current_metrics.weekly_pnl_pct:.1%} exceeds limit {self.limits.max_weekly_loss:.1%}",
                    current_value=abs(self.current_metrics.weekly_pnl_pct),
                    limit_value=self.limits.max_weekly_loss
                )
                alerts.append(alert)
            
            # Drawdown limit
            if self.current_metrics.current_drawdown_pct > self.limits.max_drawdown:
                alert = await self._create_alert(
                    AlertType.DRAWDOWN_LIMIT,
                    RiskLevel.HIGH,
                    f"Current drawdown {self.current_metrics.current_drawdown_pct:.1%} exceeds limit {self.limits.max_drawdown:.1%}",
                    current_value=self.current_metrics.current_drawdown_pct,
                    limit_value=self.limits.max_drawdown
                )
                alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking loss limits: {e}")
        
        return alerts
    
    async def _check_position_limits(self) -> List[RiskAlert]:
        """Check position limits."""
        alerts = []
        
        try:
            # Check individual position limits
            if self.current_metrics.largest_position_pct > self.limits.max_position_size:
                alert = await self._create_alert(
                    AlertType.POSITION_LIMIT,
                    RiskLevel.HIGH,
                    f"Largest position {self.current_metrics.largest_position_pct:.1%} exceeds limit {self.limits.max_position_size:.1%}",
                    current_value=self.current_metrics.largest_position_pct,
                    limit_value=self.limits.max_position_size
                )
                alerts.append(alert)
            
            # Check symbol exposure limits
            for symbol, position in self.positions.items():
                exposure_pct = abs(position) / max(self.current_metrics.total_value, 1)
                if exposure_pct > self.limits.max_symbol_exposure:
                    alert = await self._create_alert(
                        AlertType.EXPOSURE_LIMIT,
                        RiskLevel.MEDIUM,
                        f"Symbol {symbol} exposure {exposure_pct:.1%} exceeds limit {self.limits.max_symbol_exposure:.1%}",
                        symbol=symbol,
                        current_value=exposure_pct,
                        limit_value=self.limits.max_symbol_exposure
                    )
                    alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
        
        return alerts
    
    async def _check_risk_metrics(self) -> List[RiskAlert]:
        """Check risk metrics limits."""
        alerts = []
        
        try:
            # VaR limits
            if self.current_metrics.var_95_1d > self.limits.max_var_95:
                alert = await self._create_alert(
                    AlertType.VAR_LIMIT,
                    RiskLevel.HIGH,
                    f"VaR 95% {self.current_metrics.var_95_1d:.1%} exceeds limit {self.limits.max_var_95:.1%}",
                    current_value=self.current_metrics.var_95_1d,
                    limit_value=self.limits.max_var_95
                )
                alerts.append(alert)
            
            if self.current_metrics.var_99_1d > self.limits.max_var_99:
                alert = await self._create_alert(
                    AlertType.VAR_LIMIT,
                    RiskLevel.CRITICAL,
                    f"VaR 99% {self.current_metrics.var_99_1d:.1%} exceeds limit {self.limits.max_var_99:.1%}",
                    current_value=self.current_metrics.var_99_1d,
                    limit_value=self.limits.max_var_99
                )
                alerts.append(alert)
            
            # Volatility limit
            if self.current_metrics.portfolio_volatility > self.limits.max_portfolio_volatility:
                alert = await self._create_alert(
                    AlertType.VOLATILITY_SPIKE,
                    RiskLevel.MEDIUM,
                    f"Portfolio volatility {self.current_metrics.portfolio_volatility:.1%} exceeds limit {self.limits.max_portfolio_volatility:.1%}",
                    current_value=self.current_metrics.portfolio_volatility,
                    limit_value=self.limits.max_portfolio_volatility
                )
                alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking risk metrics: {e}")
        
        return alerts
    
    async def _check_concentration_risk(self) -> List[RiskAlert]:
        """Check concentration risk."""
        alerts = []
        
        try:
            # Concentration ratio (HHI)
            if self.current_metrics.concentration_ratio > (1 - self.limits.min_diversification_ratio):
                alert = await self._create_alert(
                    AlertType.CONCENTRATION_RISK,
                    RiskLevel.MEDIUM,
                    f"Portfolio concentration {self.current_metrics.concentration_ratio:.2f} indicates insufficient diversification",
                    current_value=self.current_metrics.concentration_ratio,
                    limit_value=1 - self.limits.min_diversification_ratio
                )
                alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking concentration risk: {e}")
        
        return alerts
    
    def _check_trading_frequency(self) -> bool:
        """Check trading frequency limits."""
        return (
            self.current_metrics.orders_last_minute <= self.limits.max_orders_per_minute and
            self.current_metrics.orders_last_hour <= self.limits.max_orders_per_hour and
            self.current_metrics.orders_today <= self.limits.max_orders_per_day
        )
    
    async def _create_alert(
        self,
        alert_type: AlertType,
        level: RiskLevel,
        message: str,
        symbol: Optional[str] = None,
        current_value: Optional[float] = None,
        limit_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskAlert:
        """Create and process risk alert."""
        alert_id = f"{alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            level=level,
            message=message,
            symbol=symbol,
            current_value=current_value,
            limit_value=limit_value,
            metadata=metadata or {}
        )
        
        # Store alert
        self.alerts[alert_id] = alert
        
        # Log alert
        log_level = {
            RiskLevel.LOW: logging.INFO,
            RiskLevel.MEDIUM: logging.WARNING,
            RiskLevel.HIGH: logging.ERROR,
            RiskLevel.CRITICAL: logging.CRITICAL
        }.get(level, logging.WARNING)
        
        self.logger.log(log_level, f"Risk Alert [{level.value.upper()}]: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        return alert
    
    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        level: Optional[RiskLevel] = None,
        acknowledged: Optional[bool] = None
    ) -> List[RiskAlert]:
        """Get alerts with optional filters."""
        alerts = list(self.alerts.values())
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary."""
        unacknowledged_alerts = len([a for a in self.alerts.values() if not a.acknowledged])
        critical_alerts = len([a for a in self.alerts.values() if a.level == RiskLevel.CRITICAL and not a.acknowledged])
        
        return {
            'current_metrics': self.current_metrics.to_dict(),
            'limits': self.limits.to_dict(),
            'alerts': {
                'total': len(self.alerts),
                'unacknowledged': unacknowledged_alerts,
                'critical': critical_alerts
            },
            'risk_score': self._calculate_risk_score()
        }
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        try:
            score = 0.0
            
            # Position risk (0-25 points)
            position_risk = min(self.current_metrics.largest_position_pct / self.limits.max_position_size, 1.0)
            score += position_risk * 25
            
            # Loss risk (0-25 points)
            loss_risk = min(abs(self.current_metrics.daily_pnl_pct) / self.limits.max_daily_loss, 1.0)
            score += loss_risk * 25
            
            # VaR risk (0-25 points)
            var_risk = min(self.current_metrics.var_95_1d / self.limits.max_var_95, 1.0)
            score += var_risk * 25
            
            # Concentration risk (0-25 points)
            concentration_risk = min(self.current_metrics.concentration_ratio / (1 - self.limits.min_diversification_ratio), 1.0)
            score += concentration_risk * 25
            
            return min(score, 100.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.0