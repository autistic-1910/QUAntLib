"""Unit tests for broker execution semantics including commission and slippage."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock
from quantlib.backtesting.broker import SimulatedBroker, AdvancedBroker
from quantlib.backtesting.events import OrderEvent, FillEvent
from quantlib.backtesting.data_handler import DataHandler
from datetime import datetime


class TestSimulatedBroker:
    """Test basic broker execution with deterministic calculations."""
    
    @pytest.fixture
    def mock_data_handler(self):
        """Mock data handler with predictable market data."""
        data_handler = Mock(spec=DataHandler)
        
        # Standard market data
        market_data = {
            'open': 100.0,
            'high': 102.0,
            'low': 98.0,
            'close': 101.0,
            'volume': 10000
        }
        
        data_handler.get_latest_bar.return_value = market_data
        return data_handler
    
    @pytest.fixture
    def broker(self, mock_data_handler):
        """Simulated broker with known parameters."""
        return SimulatedBroker(
            data_handler=mock_data_handler,
            commission_rate=0.001,  # 0.1%
            slippage_rate=0.0005,   # 0.05%
            market_impact=0.0001    # 0.01%
        )
    
    @pytest.fixture
    def buy_market_order(self):
        """Standard buy market order."""
        return OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='MKT',
            quantity=100,
            direction='BUY',
            price=None,
            order_id='test_order_1'
        )
    
    @pytest.fixture
    def sell_market_order(self):
        """Standard sell market order."""
        return OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='MKT',
            quantity=100,
            direction='SELL',
            price=None,
            order_id='test_order_2'
        )
    
    def test_commission_calculation(self, broker, buy_market_order):
        """Test commission calculation is correct."""
        # Mock random to make slippage deterministic
        np.random.seed(42)
        
        fill_event = broker.execute_order(buy_market_order)
        
        assert fill_event is not None
        
        # Commission = trade_value * commission_rate
        trade_value = fill_event.quantity * fill_event.fill_price
        expected_commission = trade_value * broker.commission_rate
        
        assert abs(fill_event.commission - expected_commission) < 1e-10
    
    def test_buy_order_slippage_adverse(self, broker, buy_market_order):
        """Test that buy orders have adverse slippage (higher price)."""
        np.random.seed(42)  # Fixed seed for reproducible slippage
        
        fill_event = broker.execute_order(buy_market_order)
        
        assert fill_event is not None
        
        # For buy orders, fill price should be >= market price due to slippage
        market_price = 101.0  # Close price from mock data
        assert fill_event.fill_price >= market_price
    
    def test_sell_order_slippage_adverse(self, broker, sell_market_order):
        """Test that sell orders have adverse slippage (lower price)."""
        np.random.seed(42)  # Fixed seed for reproducible slippage
        
        fill_event = broker.execute_order(sell_market_order)
        
        assert fill_event is not None
        
        # For sell orders, fill price should be <= market price due to slippage
        market_price = 101.0  # Close price from mock data
        assert fill_event.fill_price <= market_price
    
    def test_market_impact_scaling(self, mock_data_handler):
        """Test that market impact scales with order size."""
        broker = SimulatedBroker(
            data_handler=mock_data_handler,
            commission_rate=0.0,  # No commission for cleaner test
            slippage_rate=0.0,    # No random slippage
            market_impact=0.001   # 0.1% market impact
        )
        
        # Small order
        small_order = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='MKT',
            quantity=100,
            direction='BUY'
        )
        
        # Large order (10x size)
        large_order = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='MKT',
            quantity=1000,
            direction='BUY'
        )
        
        np.random.seed(42)
        small_fill = broker.execute_order(small_order)
        
        np.random.seed(42)
        large_fill = broker.execute_order(large_order)
        
        # Large order should have higher impact (higher price for buy)
        assert large_fill.fill_price > small_fill.fill_price
    
    def test_limit_order_execution(self, broker):
        """Test limit order execution logic."""
        # Buy limit order below market price (should execute)
        buy_limit = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='LMT',
            quantity=100,
            direction='BUY',
            price=99.0  # Below low price of 98.0
        )
        
        fill_event = broker.execute_order(buy_limit)
        assert fill_event is not None
        assert fill_event.fill_price <= 99.0  # Should not exceed limit
    
    def test_limit_order_no_execution(self, broker):
        """Test limit order that should not execute."""
        # Buy limit order above market price (should not execute)
        buy_limit = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='LMT',
            quantity=100,
            direction='BUY',
            price=97.0  # Above high price of 102.0 (impossible for buy limit)
        )
        
        fill_event = broker.execute_order(buy_limit)
        assert fill_event is None  # Should not execute
    
    def test_stop_order_execution(self, broker):
        """Test stop order execution logic."""
        # Buy stop order (triggered when price goes above stop)
        buy_stop = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='STP',
            quantity=100,
            direction='BUY',
            price=100.0  # Below high price of 102.0 (should trigger)
        )
        
        fill_event = broker.execute_order(buy_stop)
        assert fill_event is not None
        assert fill_event.fill_price >= 100.0  # Should be at or above stop
    
    def test_no_market_data(self, buy_market_order):
        """Test order execution when no market data is available."""
        mock_data_handler = Mock(spec=DataHandler)
        mock_data_handler.get_latest_bar.return_value = None
        
        broker = SimulatedBroker(data_handler=mock_data_handler)
        
        fill_event = broker.execute_order(buy_market_order)
        assert fill_event is None
    
    def test_invalid_price_data(self, buy_market_order):
        """Test order execution with invalid price data."""
        mock_data_handler = Mock(spec=DataHandler)
        mock_data_handler.get_latest_bar.return_value = {
            'open': 0,
            'high': 0,
            'low': 0,
            'close': 0,  # Invalid price
            'volume': 1000
        }
        
        broker = SimulatedBroker(data_handler=mock_data_handler)
        
        fill_event = broker.execute_order(buy_market_order)
        assert fill_event is None
    
    def test_order_tracking(self, broker, buy_market_order):
        """Test that executed orders are properly tracked."""
        fill_event = broker.execute_order(buy_market_order)
        
        assert fill_event is not None
        assert buy_market_order.order_id in broker.executed_orders
        
        order_data = broker.executed_orders[buy_market_order.order_id]
        assert order_data['order'] == buy_market_order
        assert order_data['fill'] == fill_event
    
    def test_execution_statistics(self, broker, buy_market_order, sell_market_order):
        """Test execution statistics calculation."""
        np.random.seed(42)
        
        # Execute multiple orders
        fill1 = broker.execute_order(buy_market_order)
        fill2 = broker.execute_order(sell_market_order)
        
        stats = broker.get_execution_statistics()
        
        assert stats['total_orders'] == 2
        assert stats['total_volume'] == 200  # 100 + 100
        assert stats['total_commission'] == fill1.commission + fill2.commission
        assert stats['success_rate'] == 1.0
        assert 'avg_slippage' in stats


class TestAdvancedBroker:
    """Test advanced broker features like partial fills."""
    
    @pytest.fixture
    def mock_data_handler(self):
        """Mock data handler for advanced broker."""
        data_handler = Mock(spec=DataHandler)
        data_handler.get_latest_bar.return_value = {
            'open': 100.0,
            'high': 102.0,
            'low': 98.0,
            'close': 101.0,
            'volume': 10000
        }
        return data_handler
    
    @pytest.fixture
    def advanced_broker(self, mock_data_handler):
        """Advanced broker with partial fill capability."""
        return AdvancedBroker(
            data_handler=mock_data_handler,
            commission_rate=0.001,
            slippage_rate=0.0005,
            market_impact=0.0001,
            partial_fill_prob=1.0  # Always partial fill for testing
        )
    
    def test_partial_fill_execution(self, advanced_broker):
        """Test partial fill execution and tracking."""
        large_order = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='MKT',
            quantity=1000,  # Large order eligible for partial fill
            direction='BUY',
            order_id='partial_test'
        )
        
        np.random.seed(42)  # Fixed seed for reproducible partial fill
        
        fill_event = advanced_broker.execute_order(large_order)
        
        assert fill_event is not None
        assert fill_event.quantity < large_order.quantity  # Partial fill
        
        # Check that remaining quantity is tracked
        assert large_order.order_id in advanced_broker.pending_orders
        pending_data = advanced_broker.pending_orders[large_order.order_id]
        assert pending_data['remaining_quantity'] > 0
        assert pending_data['remaining_quantity'] == large_order.quantity - fill_event.quantity
    
    def test_no_partial_fill_small_order(self, advanced_broker):
        """Test that small orders don't get partial fills."""
        small_order = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='MKT',
            quantity=50,  # Small order, should execute fully
            direction='BUY'
        )
        
        fill_event = advanced_broker.execute_order(small_order)
        
        assert fill_event is not None
        assert fill_event.quantity == small_order.quantity  # Full fill
    
    def test_partial_fill_probability(self, mock_data_handler):
        """Test partial fill probability mechanism."""
        # Broker with 0% partial fill probability
        no_partial_broker = AdvancedBroker(
            data_handler=mock_data_handler,
            partial_fill_prob=0.0
        )
        
        large_order = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol='AAPL',
            order_type='MKT',
            quantity=1000,
            direction='BUY'
        )
        
        fill_event = no_partial_broker.execute_order(large_order)
        
        assert fill_event is not None
        assert fill_event.quantity == large_order.quantity  # Full fill
        assert large_order.order_id not in no_partial_broker.pending_orders