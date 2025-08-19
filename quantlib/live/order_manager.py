"""Order management system for live trading.

This module provides order management capabilities including:
- Order lifecycle management
- Broker API integration
- Order routing and execution
- Position tracking
- Risk checks
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from ..backtesting.events import OrderEvent, FillEvent
from ..core.base import BaseComponent


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTD = "gtd"  # Good Till Date


@dataclass
class LiveOrder:
    """Live trading order."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    created_time: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    broker_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    def update_fill(self, quantity: int, price: float, commission: float = 0.0) -> None:
        """Update order with fill information."""
        if quantity <= 0:
            return
        
        # Update average fill price
        total_value = self.avg_fill_price * self.filled_quantity + price * quantity
        self.filled_quantity += quantity
        self.avg_fill_price = total_value / self.filled_quantity
        self.commission += commission
        
        # Update status
        if self.is_filled:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'created_time': self.created_time.isoformat(),
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'broker_order_id': self.broker_order_id,
            'parent_order_id': self.parent_order_id,
            'metadata': self.metadata
        }


class LiveBroker(ABC):
    """Abstract base class for live brokers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: LiveOrder) -> bool:
        """Submit order to broker."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass


class SimulatedBroker(LiveBroker):
    """Simulated broker for testing."""
    
    def __init__(self, name: str = "simulated", initial_cash: float = 100000.0):
        super().__init__(name)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.orders: Dict[str, LiveOrder] = {}
        self.fill_delay = 0.1  # seconds
        self.commission_rate = 0.001  # 0.1%
        
    async def start(self) -> bool:
        """Start the simulated broker."""
        return await self.connect()
    
    async def stop(self) -> None:
        """Stop the simulated broker."""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """Connect to simulated broker."""
        self.connected = True
        self.logger.info("Connected to simulated broker")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from simulated broker."""
        self.connected = False
        self.logger.info("Disconnected from simulated broker")
    
    async def submit_order(self, order: LiveOrder) -> bool:
        """Submit order to simulated broker."""
        if not self.connected:
            return False
        
        try:
            order.status = OrderStatus.SUBMITTED
            order.broker_order_id = f"sim_{uuid4().hex[:8]}"
            self.orders[order.order_id] = order
            
            self.logger.info(f"Order submitted: {order.order_id}")
            
            # Simulate order processing
            asyncio.create_task(self._process_order(order))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                self.logger.info(f"Order cancelled: {order_id}")
                return True
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return None
    
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            # Use last known price (simplified)
            total_value += quantity * 100.0  # Assume $100 per share
        
        return {
            'cash': self.cash,
            'total_value': total_value,
            'positions': self.positions,
            'buying_power': self.cash
        }
    
    async def _process_order(self, order: LiveOrder) -> None:
        """Process order simulation."""
        try:
            # Simulate processing delay
            await asyncio.sleep(self.fill_delay)
            
            if order.status != OrderStatus.SUBMITTED:
                return
            
            order.status = OrderStatus.ACKNOWLEDGED
            
            # Simulate fill (simplified)
            fill_price = order.price if order.price else 100.0  # Default price
            commission = order.quantity * fill_price * self.commission_rate
            
            # Check if we have enough cash/shares
            if order.side == 'BUY':
                required_cash = order.quantity * fill_price + commission
                if self.cash >= required_cash:
                    self.cash -= required_cash
                    self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
                    order.update_fill(order.quantity, fill_price, commission)
                else:
                    order.status = OrderStatus.REJECTED
                    return
            else:  # SELL
                current_position = self.positions.get(order.symbol, 0)
                if current_position >= order.quantity:
                    self.cash += order.quantity * fill_price - commission
                    self.positions[order.symbol] = current_position - order.quantity
                    order.update_fill(order.quantity, fill_price, commission)
                else:
                    order.status = OrderStatus.REJECTED
                    return
            
            self.logger.info(f"Order filled: {order.order_id} at {fill_price}")
            
        except Exception as e:
            self.logger.error(f"Error processing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED


class OrderManager(BaseComponent):
    """Order management system."""
    
    def __init__(self, broker: LiveBroker, name: str = "OrderManager"):
        super().__init__(name)
        self.broker = broker
        self.orders: Dict[str, LiveOrder] = {}
        self.order_callbacks: List[Callable[[LiveOrder], None]] = []
        self.fill_callbacks: List[Callable[[FillEvent], None]] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._monitoring = False
        
    async def start(self) -> bool:
        """Start order manager."""
        if await self.broker.connect():
            self._monitoring = True
            asyncio.create_task(self._monitor_orders())
            self.logger.info("Order manager started")
            return True
        return False
    
    async def stop(self) -> None:
        """Stop order manager."""
        self._monitoring = False
        await self.broker.disconnect()
        self.logger.info("Order manager stopped")
    
    def add_order_callback(self, callback: Callable[[LiveOrder], None]) -> None:
        """Add order status callback."""
        self.order_callbacks.append(callback)
    
    def add_fill_callback(self, callback: Callable[[FillEvent], None]) -> None:
        """Add fill callback."""
        self.fill_callbacks.append(callback)
    
    async def submit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Submit market order."""
        return await self.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            metadata=metadata
        )
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Submit new order."""
        try:
            order = LiveOrder(
                order_id=f"order_{uuid4().hex[:8]}",
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                metadata=metadata or {}
            )
            
            if await self.broker.submit_order(order):
                self.orders[order.order_id] = order
                self._notify_order_callbacks(order)
                self.logger.info(f"Order submitted: {order.order_id}")
                return order.order_id
            else:
                self.logger.error(f"Failed to submit order: {order.order_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        if order_id in self.orders:
            if await self.broker.cancel_order(order_id):
                order = self.orders[order_id]
                order.status = OrderStatus.CANCELLED
                self._notify_order_callbacks(order)
                self.logger.info(f"Order cancelled: {order_id}")
                return True
        return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders, optionally filtered by symbol."""
        cancelled_count = 0
        for order in self.orders.values():
            if order.is_active and (symbol is None or order.symbol == symbol):
                if await self.cancel_order(order.order_id):
                    cancelled_count += 1
        return cancelled_count
    
    def get_order(self, order_id: str) -> Optional[LiveOrder]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None
    ) -> List[LiveOrder]:
        """Get orders with optional filters."""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return orders
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[LiveOrder]:
        """Get active orders."""
        orders = [o for o in self.orders.values() if o.is_active]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions from broker."""
        return await self.broker.get_positions()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from broker."""
        return await self.broker.get_account_info()
    
    async def _monitor_orders(self) -> None:
        """Monitor order status updates."""
        while self._monitoring:
            try:
                for order in list(self.orders.values()):
                    if order.is_active:
                        # Check for status updates
                        current_status = await self.broker.get_order_status(order.order_id)
                        if current_status and current_status != order.status:
                            old_status = order.status
                            order.status = current_status
                            
                            self.logger.info(
                                f"Order {order.order_id} status changed: {old_status} -> {current_status}"
                            )
                            
                            self._notify_order_callbacks(order)
                            
                            # Generate fill event if order was filled
                            if current_status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                                self._generate_fill_event(order)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(5)
    
    def _notify_order_callbacks(self, order: LiveOrder) -> None:
        """Notify order status callbacks."""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
    
    def _generate_fill_event(self, order: LiveOrder) -> None:
        """Generate fill event for filled order."""
        if order.filled_quantity > 0:
            fill_event = FillEvent(
                timestamp=datetime.now(),
                symbol=order.symbol,
                exchange="LIVE",
                quantity=order.filled_quantity,
                direction=order.side,
                fill_cost=order.avg_fill_price,
                commission=order.commission
            )
            
            for callback in self.fill_callbacks:
                try:
                    callback(fill_event)
                except Exception as e:
                    self.logger.error(f"Error in fill callback: {e}")
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        rejected_orders = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])
        active_orders = len([o for o in self.orders.values() if o.is_active])
        
        total_commission = sum(o.commission for o in self.orders.values())
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'rejected_orders': rejected_orders,
            'active_orders': active_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'total_commission': total_commission
        }