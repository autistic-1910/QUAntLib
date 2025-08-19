#!/usr/bin/env python3
"""
Strategy Execution Module

Provides order execution and management capabilities:
- Order types and execution algorithms
- Market impact modeling
- Execution cost analysis
- Order routing and management
- Slippage and transaction cost modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from quantlib.core.utils import Logger
from quantlib.strategy.base import StrategySignal, SignalType


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    asset: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to fill"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return abs(self.remaining_quantity) < 1e-8
    
    @property
    def fill_ratio(self) -> float:
        """Ratio of filled quantity"""
        return self.filled_quantity / self.quantity if self.quantity != 0 else 0


@dataclass
class Fill:
    """Order fill representation"""
    fill_id: str
    order_id: str
    asset: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Execution result"""
    signal: StrategySignal
    orders: List[Order]
    total_quantity: float
    avg_price: Optional[float] = None
    total_commission: float = 0.0
    total_slippage: float = 0.0
    execution_time: Optional[timedelta] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class MarketDataProvider(ABC):
    """Abstract market data provider"""
    
    @abstractmethod
    def get_current_price(self, asset: str) -> Optional[float]:
        """Get current market price"""
        pass
    
    @abstractmethod
    def get_bid_ask(self, asset: str) -> Optional[Tuple[float, float]]:
        """Get current bid/ask prices"""
        pass
    
    @abstractmethod
    def get_volume(self, asset: str, period: str = '1d') -> Optional[float]:
        """Get trading volume"""
        pass


class SimulatedMarketData(MarketDataProvider):
    """Simulated market data for backtesting"""
    
    def __init__(self, price_data: pd.DataFrame, 
                 bid_ask_spread: float = 0.001):
        self.price_data = price_data
        self.bid_ask_spread = bid_ask_spread
        self.current_time = None
        
    def set_current_time(self, timestamp: datetime):
        """Set current time for simulation"""
        self.current_time = timestamp
    
    def get_current_price(self, asset: str) -> Optional[float]:
        """Get current market price"""
        if asset not in self.price_data.columns:
            return None
        
        if self.current_time is None:
            return self.price_data[asset].iloc[-1]
        
        # Find closest timestamp
        try:
            idx = self.price_data.index.get_loc(self.current_time, method='nearest')
            return self.price_data[asset].iloc[idx]
        except (KeyError, IndexError):
            return None
    
    def get_bid_ask(self, asset: str) -> Optional[Tuple[float, float]]:
        """Get current bid/ask prices"""
        mid_price = self.get_current_price(asset)
        if mid_price is None:
            return None
        
        spread = mid_price * self.bid_ask_spread
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2
        
        return bid, ask
    
    def get_volume(self, asset: str, period: str = '1d') -> Optional[float]:
        """Get trading volume (simulated)"""
        price = self.get_current_price(asset)
        if price is None:
            return None
        
        # Simulate volume based on price volatility
        if len(self.price_data) > 20:
            returns = self.price_data[asset].pct_change().dropna()
            volatility = returns.std()
            base_volume = 1000000  # Base volume
            volume_multiplier = 1 + volatility * 10  # Higher vol = higher volume
            return base_volume * volume_multiplier
        
        return 1000000  # Default volume


class TransactionCostModel(ABC):
    """Abstract transaction cost model"""
    
    @abstractmethod
    def calculate_cost(self, order: Order, market_data: MarketDataProvider) -> Tuple[float, float]:
        """Calculate commission and slippage"""
        pass


class SimpleTransactionCostModel(TransactionCostModel):
    """Simple transaction cost model"""
    
    def __init__(self, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 market_impact_factor: float = 0.1):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_impact_factor = market_impact_factor
    
    def calculate_cost(self, order: Order, market_data: MarketDataProvider) -> Tuple[float, float]:
        """Calculate commission and slippage"""
        # Commission calculation
        notional = abs(order.quantity * (order.price or 0))
        commission = notional * self.commission_rate
        
        # Slippage calculation
        base_slippage = notional * self.slippage_rate
        
        # Market impact (depends on order size relative to volume)
        volume = market_data.get_volume(order.asset)
        if volume and volume > 0:
            volume_ratio = abs(order.quantity) / volume
            market_impact = base_slippage * (1 + volume_ratio * self.market_impact_factor)
        else:
            market_impact = base_slippage
        
        return commission, market_impact


class BaseExecutionAlgorithm(ABC):
    """Base execution algorithm"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger.get_logger(f"execution_{name.lower()}")
    
    @abstractmethod
    def execute_signal(self, signal: StrategySignal,
                      market_data: MarketDataProvider,
                      cost_model: TransactionCostModel) -> ExecutionResult:
        """Execute a trading signal"""
        pass
    
    def _create_order(self, signal: StrategySignal, 
                     order_type: OrderType = OrderType.MARKET,
                     price: Optional[float] = None) -> Order:
        """Create an order from a signal"""
        # Determine order side and quantity
        if signal.signal_type == SignalType.BUY:
            side = OrderSide.BUY
            quantity = abs(signal.target_position)
        elif signal.signal_type == SignalType.SELL:
            side = OrderSide.SELL
            quantity = abs(signal.target_position)
        else:  # HOLD
            # For hold signals, we might need to close positions
            side = OrderSide.SELL  # Default to sell
            quantity = 0  # No quantity for hold
        
        return Order(
            order_id=str(uuid.uuid4()),
            asset=signal.asset,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=signal.timestamp,
            metadata={
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'original_signal': signal.metadata
            }
        )


class MarketOrderExecution(BaseExecutionAlgorithm):
    """Simple market order execution"""
    
    def __init__(self):
        super().__init__("MarketOrder")
    
    def execute_signal(self, signal: StrategySignal,
                      market_data: MarketDataProvider,
                      cost_model: TransactionCostModel) -> ExecutionResult:
        """Execute signal with market order"""
        start_time = datetime.now()
        
        try:
            # Skip hold signals with zero target position
            if signal.signal_type == SignalType.HOLD or abs(signal.target_position) < 1e-8:
                return ExecutionResult(
                    signal=signal,
                    orders=[],
                    total_quantity=0,
                    success=True,
                    metadata={'reason': 'hold_signal_skipped'}
                )
            
            # Get current market price
            current_price = market_data.get_current_price(signal.asset)
            if current_price is None:
                return ExecutionResult(
                    signal=signal,
                    orders=[],
                    total_quantity=0,
                    success=False,
                    error_message=f"No price data for {signal.asset}"
                )
            
            # Create market order
            order = self._create_order(signal, OrderType.MARKET, current_price)
            
            # Calculate transaction costs
            commission, slippage = cost_model.calculate_cost(order, market_data)
            
            # Simulate execution (immediate fill for market orders)
            bid, ask = market_data.get_bid_ask(signal.asset)
            if bid is None or ask is None:
                execution_price = current_price
            else:
                # Use bid for sells, ask for buys
                execution_price = ask if order.side == OrderSide.BUY else bid
            
            # Apply slippage
            if order.side == OrderSide.BUY:
                execution_price += slippage / order.quantity if order.quantity > 0 else 0
            else:
                execution_price -= slippage / order.quantity if order.quantity > 0 else 0
            
            # Fill the order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.commission = commission
            order.slippage = slippage
            
            execution_time = datetime.now() - start_time
            
            return ExecutionResult(
                signal=signal,
                orders=[order],
                total_quantity=order.quantity,
                avg_price=execution_price,
                total_commission=commission,
                total_slippage=slippage,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Execution failed for {signal.asset}: {e}")
            return ExecutionResult(
                signal=signal,
                orders=[],
                total_quantity=0,
                success=False,
                error_message=str(e)
            )


class TWAPExecution(BaseExecutionAlgorithm):
    """Time-Weighted Average Price execution"""
    
    def __init__(self, duration_minutes: int = 30, 
                 num_slices: int = 10):
        super().__init__("TWAP")
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
    
    def execute_signal(self, signal: StrategySignal,
                      market_data: MarketDataProvider,
                      cost_model: TransactionCostModel) -> ExecutionResult:
        """Execute signal using TWAP algorithm"""
        start_time = datetime.now()
        
        try:
            # Skip hold signals
            if signal.signal_type == SignalType.HOLD or abs(signal.target_position) < 1e-8:
                return ExecutionResult(
                    signal=signal,
                    orders=[],
                    total_quantity=0,
                    success=True,
                    metadata={'reason': 'hold_signal_skipped'}
                )
            
            # Calculate slice parameters
            slice_quantity = abs(signal.target_position) / self.num_slices
            slice_interval = timedelta(minutes=self.duration_minutes / self.num_slices)
            
            orders = []
            total_quantity = 0
            total_commission = 0
            total_slippage = 0
            total_value = 0
            
            # Create and execute slices
            for i in range(self.num_slices):
                # Create slice order
                slice_order = self._create_order(signal, OrderType.MARKET)
                slice_order.quantity = slice_quantity
                slice_order.timestamp = signal.timestamp + i * slice_interval
                
                # Get execution price for this slice
                current_price = market_data.get_current_price(signal.asset)
                if current_price is None:
                    continue
                
                # Calculate costs for this slice
                commission, slippage = cost_model.calculate_cost(slice_order, market_data)
                
                # Simulate execution
                bid, ask = market_data.get_bid_ask(signal.asset)
                if bid and ask:
                    execution_price = ask if slice_order.side == OrderSide.BUY else bid
                else:
                    execution_price = current_price
                
                # Apply slippage (reduced for smaller slices)
                slice_slippage = slippage * 0.5  # TWAP reduces market impact
                if slice_order.side == OrderSide.BUY:
                    execution_price += slice_slippage / slice_quantity if slice_quantity > 0 else 0
                else:
                    execution_price -= slice_slippage / slice_quantity if slice_quantity > 0 else 0
                
                # Fill the slice
                slice_order.status = OrderStatus.FILLED
                slice_order.filled_quantity = slice_quantity
                slice_order.avg_fill_price = execution_price
                slice_order.commission = commission
                slice_order.slippage = slice_slippage
                
                orders.append(slice_order)
                total_quantity += slice_quantity
                total_commission += commission
                total_slippage += slice_slippage
                total_value += slice_quantity * execution_price
            
            avg_price = total_value / total_quantity if total_quantity > 0 else None
            execution_time = datetime.now() - start_time
            
            return ExecutionResult(
                signal=signal,
                orders=orders,
                total_quantity=total_quantity,
                avg_price=avg_price,
                total_commission=total_commission,
                total_slippage=total_slippage,
                execution_time=execution_time,
                success=True,
                metadata={
                    'algorithm': 'TWAP',
                    'num_slices': len(orders),
                    'duration_minutes': self.duration_minutes
                }
            )
            
        except Exception as e:
            self.logger.error(f"TWAP execution failed for {signal.asset}: {e}")
            return ExecutionResult(
                signal=signal,
                orders=[],
                total_quantity=0,
                success=False,
                error_message=str(e)
            )


class VWAPExecution(BaseExecutionAlgorithm):
    """Volume-Weighted Average Price execution"""
    
    def __init__(self, participation_rate: float = 0.1,
                 duration_minutes: int = 60):
        super().__init__("VWAP")
        self.participation_rate = participation_rate  # Fraction of market volume
        self.duration_minutes = duration_minutes
    
    def execute_signal(self, signal: StrategySignal,
                      market_data: MarketDataProvider,
                      cost_model: TransactionCostModel) -> ExecutionResult:
        """Execute signal using VWAP algorithm"""
        start_time = datetime.now()
        
        try:
            # Skip hold signals
            if signal.signal_type == SignalType.HOLD or abs(signal.target_position) < 1e-8:
                return ExecutionResult(
                    signal=signal,
                    orders=[],
                    total_quantity=0,
                    success=True,
                    metadata={'reason': 'hold_signal_skipped'}
                )
            
            # Get market volume
            daily_volume = market_data.get_volume(signal.asset)
            if daily_volume is None:
                # Fallback to market order execution
                market_exec = MarketOrderExecution()
                return market_exec.execute_signal(signal, market_data, cost_model)
            
            # Calculate execution parameters
            target_volume = daily_volume * self.participation_rate
            execution_rate = target_volume / (self.duration_minutes * 60)  # Per second
            
            # Simulate VWAP execution (simplified)
            order = self._create_order(signal, OrderType.VWAP)
            
            # Get current price
            current_price = market_data.get_current_price(signal.asset)
            if current_price is None:
                return ExecutionResult(
                    signal=signal,
                    orders=[],
                    total_quantity=0,
                    success=False,
                    error_message=f"No price data for {signal.asset}"
                )
            
            # Calculate costs (reduced for VWAP)
            commission, slippage = cost_model.calculate_cost(order, market_data)
            vwap_slippage = slippage * 0.3  # VWAP significantly reduces market impact
            
            # Simulate VWAP execution price
            bid, ask = market_data.get_bid_ask(signal.asset)
            if bid and ask:
                vwap_price = (bid + ask) / 2  # VWAP typically gets mid-price
            else:
                vwap_price = current_price
            
            # Fill the order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = vwap_price
            order.commission = commission
            order.slippage = vwap_slippage
            
            execution_time = datetime.now() - start_time
            
            return ExecutionResult(
                signal=signal,
                orders=[order],
                total_quantity=order.quantity,
                avg_price=vwap_price,
                total_commission=commission,
                total_slippage=vwap_slippage,
                execution_time=execution_time,
                success=True,
                metadata={
                    'algorithm': 'VWAP',
                    'participation_rate': self.participation_rate,
                    'daily_volume': daily_volume
                }
            )
            
        except Exception as e:
            self.logger.error(f"VWAP execution failed for {signal.asset}: {e}")
            return ExecutionResult(
                signal=signal,
                orders=[],
                total_quantity=0,
                success=False,
                error_message=str(e)
            )


class ExecutionManager:
    """Manages order execution and routing"""
    
    def __init__(self, market_data: MarketDataProvider,
                 cost_model: TransactionCostModel,
                 default_algorithm: str = 'market'):
        self.market_data = market_data
        self.cost_model = cost_model
        self.default_algorithm = default_algorithm
        self.logger = Logger.get_logger("execution_manager")
        
        # Initialize execution algorithms
        self.algorithms = {
            'market': MarketOrderExecution(),
            'twap': TWAPExecution(),
            'vwap': VWAPExecution()
        }
        
        # Execution history
        self.execution_history = []
        self.active_orders = {}
    
    def execute_signal(self, signal: StrategySignal,
                      algorithm: Optional[str] = None) -> ExecutionResult:
        """Execute a trading signal"""
        algo_name = algorithm or self.default_algorithm
        
        if algo_name not in self.algorithms:
            self.logger.error(f"Unknown execution algorithm: {algo_name}")
            return ExecutionResult(
                signal=signal,
                orders=[],
                total_quantity=0,
                success=False,
                error_message=f"Unknown algorithm: {algo_name}"
            )
        
        algorithm_instance = self.algorithms[algo_name]
        
        try:
            result = algorithm_instance.execute_signal(
                signal, self.market_data, self.cost_model
            )
            
            # Store execution history
            self.execution_history.append(result)
            
            # Track active orders
            for order in result.orders:
                if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    self.active_orders[order.order_id] = order
            
            price_str = f"{result.avg_price:.4f}" if result.avg_price else "N/A"
            self.logger.info(
                f"Executed {signal.asset} signal with {algo_name}: "
                f"qty={result.total_quantity:.2f}, "
                f"price={price_str}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                signal=signal,
                orders=[],
                total_quantity=0,
                success=False,
                error_message=str(e)
            )
    
    def get_execution_statistics(self) -> Dict:
        """Get execution performance statistics"""
        if not self.execution_history:
            return {}
        
        successful_executions = [r for r in self.execution_history if r.success]
        
        if not successful_executions:
            return {'total_executions': len(self.execution_history), 'success_rate': 0.0}
        
        total_commission = sum(r.total_commission for r in successful_executions)
        total_slippage = sum(r.total_slippage for r in successful_executions)
        total_quantity = sum(r.total_quantity for r in successful_executions)
        
        execution_times = [r.execution_time.total_seconds() 
                          for r in successful_executions 
                          if r.execution_time]
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_quantity': total_quantity,
            'avg_commission_rate': total_commission / (total_quantity * 100) if total_quantity > 0 else 0,
            'avg_slippage_rate': total_slippage / (total_quantity * 100) if total_quantity > 0 else 0,
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'active_orders': len(self.active_orders)
        }