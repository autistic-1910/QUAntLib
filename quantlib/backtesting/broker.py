"""Simulated broker for backtesting.

Handles order execution and generates fill events with realistic market simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import uuid

from .events import OrderEvent, FillEvent
from .data_handler import DataHandler
from quantlib.core.utils import Logger


class ExecutionHandler(ABC):
    """Abstract base class for execution handlers."""
    
    @abstractmethod
    def execute_order(self, event: OrderEvent) -> Optional[FillEvent]:
        """Execute an order and return fill event."""
        pass


class SimulatedBroker(ExecutionHandler):
    """Simulated broker with realistic execution modeling."""
    
    def __init__(self, data_handler: DataHandler, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005, market_impact: float = 0.0001):
        """
        Initialize simulated broker.
        
        Args:
            data_handler: Data handler for market data
            commission_rate: Commission rate as fraction of trade value
            slippage_rate: Slippage rate as fraction of price
            market_impact: Market impact rate as fraction of price
        """
        self.data_handler = data_handler
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_impact = market_impact
        self.logger = Logger.get_logger("broker")
        
        # Order tracking
        self.pending_orders = {}
        self.executed_orders = {}
        
    def execute_order(self, event: OrderEvent) -> Optional[FillEvent]:
        """Execute an order and return fill event."""
        symbol = event.symbol
        
        # Get current market data
        latest_bar = self.data_handler.get_latest_bar(symbol)
        if not latest_bar:
            self.logger.warning(f"No market data available for {symbol}")
            return None
            
        # Determine execution price based on order type
        fill_price = self._calculate_fill_price(event, latest_bar)
        if fill_price is None:
            return None
            
        # Calculate commission
        trade_value = event.quantity * fill_price
        commission = trade_value * self.commission_rate
        
        # Generate unique order ID if not provided
        order_id = event.order_id or str(uuid.uuid4())
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=event.timestamp,
            symbol=symbol,
            exchange="SIM",  # Simulated exchange
            quantity=event.quantity,
            direction=event.direction,
            fill_price=fill_price,
            commission=commission,
            order_id=order_id
        )
        
        # Track executed order
        self.executed_orders[order_id] = {
            'order': event,
            'fill': fill_event,
            'execution_time': event.timestamp
        }
        
        self.logger.info(f"Order executed: {event} -> Fill: {fill_event}")
        return fill_event
        
    def _calculate_fill_price(self, order: OrderEvent, market_data: Dict) -> Optional[float]:
        """Calculate realistic fill price including slippage and market impact."""
        symbol = order.symbol
        direction = order.direction
        order_type = order.order_type
        
        # Base prices from market data
        open_price = market_data.get('open', 0)
        high_price = market_data.get('high', 0)
        low_price = market_data.get('low', 0)
        close_price = market_data.get('close', 0)
        volume = market_data.get('volume', 0)
        
        if close_price <= 0:
            self.logger.warning(f"Invalid price data for {symbol}")
            return None
            
        # Determine base execution price
        if order_type == 'MKT':  # Market order
            # Use close price as base (assuming order executed at close)
            base_price = close_price
        elif order_type == 'LMT':  # Limit order
            limit_price = order.price
            if direction == 'BUY':
                # Buy limit: can only execute at or below limit price
                if low_price <= limit_price:
                    base_price = min(limit_price, close_price)
                else:
                    # Order not filled
                    return None
            else:  # SELL
                # Sell limit: can only execute at or above limit price
                if high_price >= limit_price:
                    base_price = max(limit_price, close_price)
                else:
                    # Order not filled
                    return None
        elif order_type == 'STP':  # Stop order
            stop_price = order.price
            if direction == 'BUY':
                # Buy stop: triggered when price goes above stop
                if high_price >= stop_price:
                    base_price = max(stop_price, close_price)
                else:
                    return None
            else:  # SELL
                # Sell stop: triggered when price goes below stop
                if low_price <= stop_price:
                    base_price = min(stop_price, close_price)
                else:
                    return None
        else:
            self.logger.warning(f"Unknown order type: {order_type}")
            return None
            
        # Apply slippage and market impact
        fill_price = self._apply_execution_costs(base_price, order, volume)
        
        return fill_price
        
    def _apply_execution_costs(self, base_price: float, order: OrderEvent, volume: float) -> float:
        """Apply slippage and market impact to base price."""
        direction = order.direction
        quantity = order.quantity
        
        # Calculate slippage (random component)
        slippage_factor = np.random.normal(0, self.slippage_rate)
        
        # Calculate market impact (based on order size relative to volume)
        if volume > 0:
            volume_ratio = quantity / volume
            impact_factor = self.market_impact * np.sqrt(volume_ratio)
        else:
            impact_factor = self.market_impact
            
        # Apply costs based on direction
        if direction == 'BUY':
            # Buying: slippage and impact increase price
            total_cost = abs(slippage_factor) + impact_factor
            fill_price = base_price * (1 + total_cost)
        else:  # SELL
            # Selling: slippage and impact decrease price
            total_cost = abs(slippage_factor) + impact_factor
            fill_price = base_price * (1 - total_cost)
            
        return max(fill_price, 0.01)  # Ensure positive price
        
    def get_execution_statistics(self) -> Dict:
        """Get execution statistics."""
        if not self.executed_orders:
            return {}
            
        orders = list(self.executed_orders.values())
        
        # Calculate statistics
        total_orders = len(orders)
        total_volume = sum(order['fill'].quantity for order in orders)
        total_commission = sum(order['fill'].commission for order in orders)
        
        # Calculate average slippage
        slippages = []
        for order_data in orders:
            order = order_data['order']
            fill = order_data['fill']
            
            # Get market price at execution time
            market_data = self.data_handler.get_latest_bar(order.symbol)
            if market_data:
                market_price = market_data['close']
                if order.direction == 'BUY':
                    slippage = (fill.fill_price - market_price) / market_price
                else:
                    slippage = (market_price - fill.fill_price) / market_price
                slippages.append(slippage)
                
        avg_slippage = np.mean(slippages) if slippages else 0.0
        
        return {
            'total_orders': total_orders,
            'total_volume': total_volume,
            'total_commission': total_commission,
            'avg_commission_per_trade': total_commission / total_orders if total_orders > 0 else 0,
            'avg_slippage': avg_slippage,
            'success_rate': 1.0  # All orders executed in simulation
        }
        
    def get_order_history(self) -> pd.DataFrame:
        """Get order execution history as DataFrame."""
        if not self.executed_orders:
            return pd.DataFrame()
            
        records = []
        for order_id, order_data in self.executed_orders.items():
            order = order_data['order']
            fill = order_data['fill']
            
            record = {
                'order_id': order_id,
                'timestamp': order.timestamp,
                'symbol': order.symbol,
                'direction': order.direction,
                'order_type': order.order_type,
                'quantity': order.quantity,
                'order_price': order.price,
                'fill_price': fill.fill_price,
                'commission': fill.commission,
                'execution_time': order_data['execution_time']
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            
        return df
        
    def reset(self):
        """Reset broker state."""
        self.pending_orders.clear()
        self.executed_orders.clear()


class AdvancedBroker(SimulatedBroker):
    """Advanced broker with more sophisticated execution modeling."""
    
    def __init__(self, data_handler: DataHandler, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005, market_impact: float = 0.0001,
                 latency_ms: float = 10.0, partial_fill_prob: float = 0.1):
        """
        Initialize advanced broker.
        
        Args:
            data_handler: Data handler for market data
            commission_rate: Commission rate as fraction of trade value
            slippage_rate: Slippage rate as fraction of price
            market_impact: Market impact rate as fraction of price
            latency_ms: Execution latency in milliseconds
            partial_fill_prob: Probability of partial fills
        """
        super().__init__(data_handler, commission_rate, slippage_rate, market_impact)
        self.latency_ms = latency_ms
        self.partial_fill_prob = partial_fill_prob
        
    def execute_order(self, event: OrderEvent) -> Optional[FillEvent]:
        """Execute order with advanced features like partial fills."""
        # Check for partial fill
        if np.random.random() < self.partial_fill_prob and event.quantity > 100:
            # Create partial fill (50-90% of original quantity)
            fill_ratio = np.random.uniform(0.5, 0.9)
            partial_quantity = int(event.quantity * fill_ratio)
            
            # Create partial order event
            partial_order = OrderEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                order_type=event.order_type,
                quantity=partial_quantity,
                direction=event.direction,
                price=event.price,
                order_id=event.order_id
            )
            
            # Execute partial order
            fill_event = super().execute_order(partial_order)
            
            if fill_event:
                self.logger.info(f"Partial fill: {partial_quantity}/{event.quantity} shares")
                
            return fill_event
        else:
            # Normal execution
            return super().execute_order(event)