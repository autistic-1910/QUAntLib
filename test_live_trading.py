"""Comprehensive test suite for live trading module.

This module tests all components of the live trading system:
- Data feeds
- Order management
- Risk monitoring
- Trading engine
- Configuration
- Logging and alerts
- Dashboard
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import live trading components
from quantlib.live.data_feeds import (
    DataFeedStatus, MarketData, LiveDataFeed, WebSocketDataFeed,
    SimulatedDataFeed, DataFeedManager
)
from quantlib.live.order_manager import (
    OrderStatus, OrderType, TimeInForce, LiveOrder, LiveBroker,
    SimulatedBroker, OrderManager
)
from quantlib.live.engine import (
    EngineState, EngineConfig, EngineMetrics, LiveTradingEngine
)
from quantlib.live.risk_monitor import (
    RiskLevel, AlertType, RiskAlert, RiskLimits, RiskMetrics, RiskMonitor
)
from quantlib.live.config import (
    Environment, BrokerType, DataFeedType, BrokerConfig, DataFeedConfig,
    LoggingConfig, MonitoringConfig, LiveTradingConfig, ConfigManager
)
from quantlib.live.engine import EngineConfig
from quantlib.live.logging_alerts import (
    LogLevel, AlertChannel, LogEntry, TradeLogEntry, OrderLogEntry,
    StructuredLogger, TradeLogger, OrderLogger, AlertManager, LoggingSystem
)

# Import backtesting components for testing
from quantlib.backtesting.events import MarketEvent, OrderEvent, FillEvent
from quantlib.backtesting.portfolio import BacktestPortfolio
from quantlib.backtesting.data_handler import DataHandler
from quantlib.core.base import BaseStrategy


class TestDataFeeds:
    """Test data feed components."""
    
    def test_market_data_creation(self):
        """Test MarketData dataclass."""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.0,
            ask=150.1,
            last=150.05,
            volume=1000
        )
        
        assert data.symbol == "AAPL"
        assert data.bid == 150.0
        assert data.ask == 150.1
        assert data.last == 150.05
        assert data.volume == 1000
        assert abs(data.spread - 0.1) < 1e-10
    
    @pytest.mark.asyncio
    async def test_simulated_data_feed(self):
        """Test simulated data feed."""
        symbols = ["AAPL", "GOOGL"]
        feed = SimulatedDataFeed(symbols)
        
        # Test start
        assert await feed.start()
        assert feed.status == DataFeedStatus.CONNECTED
        
        # Test data generation
        await asyncio.sleep(0.2)
        data = feed.get_latest_data("AAPL")
        assert data is not None
        assert data.symbol == "AAPL"
        assert data.bid > 0
        assert data.ask > data.bid
        
        # Test stop
        await feed.stop()
        assert feed.status == DataFeedStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_data_feed_manager(self):
        """Test data feed manager."""
        # Create simulated feeds
        feed1 = SimulatedDataFeed(["AAPL"])
        feed2 = SimulatedDataFeed(["GOOGL"])
        
        manager = DataFeedManager()
        manager.add_feed("feed1", feed1)
        manager.add_feed("feed2", feed2)
        
        # Test start
        assert await manager.start()
        assert manager.is_connected()
        
        # Test data retrieval
        await asyncio.sleep(0.2)
        aapl_data = manager.get_latest_data("AAPL")
        googl_data = manager.get_latest_data("GOOGL")
        
        assert aapl_data is not None
        assert googl_data is not None
        assert aapl_data.symbol == "AAPL"
        assert googl_data.symbol == "GOOGL"
        
        # Test stop
        await manager.stop()
        assert not manager.is_connected()


class TestOrderManagement:
    """Test order management components."""
    
    def test_live_order_creation(self):
        """Test LiveOrder creation."""
        order = LiveOrder(
            order_id="test_001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        assert order.order_id == "test_001"
        assert order.symbol == "AAPL"
        assert order.side == "BUY"
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
    
    @pytest.mark.asyncio
    async def test_simulated_broker(self):
        """Test simulated broker."""
        broker = SimulatedBroker()
        
        # Test start
        assert await broker.start()
        
        # Create and submit order
        order = LiveOrder(
            order_id="test_001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        # Submit order
        success = await broker.submit_order(order)
        assert success
        
        # Check order status
        order_status = await broker.get_order_status("test_001")
        assert order_status is not None       
        assert order_status == OrderStatus.SUBMITTED
        
        # Wait for fill (simulated broker has a delay)
        await asyncio.sleep(0.2)
        filled_status = await broker.get_order_status("test_001")
        assert filled_status == OrderStatus.FILLED
        
        # Check positions
        positions = await broker.get_positions()
        assert positions.get("AAPL", 0) == 100
        
        # Test stop
        await broker.stop()
    
    @pytest.mark.asyncio
    async def test_order_manager(self):
        """Test order manager."""
        broker = SimulatedBroker()
        manager = OrderManager(broker)
        
        # Test start
        assert await manager.start()
        
        # Submit order
        order_id = await manager.submit_market_order("AAPL", "BUY", 100)
        assert order_id is not None
        
        # Check order exists
        order = manager.get_order(order_id)
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.side == "BUY"
        assert order.quantity == 100
        
        # Wait for fill
        await asyncio.sleep(0.2)
        
        # Check order is filled
        filled_order = manager.get_order(order_id)
        assert filled_order.status == OrderStatus.FILLED
        
        # Test stop
        await manager.stop()


class TestRiskMonitoring:
    """Test risk monitoring components."""
    
    def test_risk_alert_creation(self):
        """Test RiskAlert creation."""
        alert = RiskAlert(
            alert_id="risk_001",
            alert_type=AlertType.POSITION_LIMIT,
            level=RiskLevel.HIGH,
            message="Position limit exceeded",
            symbol="AAPL",
            current_value=1000000,
            limit_value=500000
        )
        
        assert alert.alert_id == "risk_001"
        assert alert.alert_type == AlertType.POSITION_LIMIT
        assert alert.level == RiskLevel.HIGH
        assert alert.symbol == "AAPL"
        assert alert.current_value == 1000000
        assert alert.limit_value == 500000
    
    def test_risk_limits_creation(self):
        """Test RiskLimits creation."""
        limits = RiskLimits(
            max_position_size=1000000,
            max_daily_loss=50000,
            max_symbol_exposure=0.1,
            max_sector_exposure=0.3,
            max_var_95=100000
        )
        
        assert limits.max_position_size == 1000000
        assert limits.max_daily_loss == 50000
        assert limits.max_symbol_exposure == 0.1
        assert limits.max_sector_exposure == 0.3
        assert limits.max_var_95 == 100000
    
    @pytest.mark.asyncio
    async def test_risk_monitor(self):
        """Test risk monitor."""
        limits = RiskLimits(
            max_position_size=100000,
            max_daily_loss=10000,
            max_symbol_exposure=0.1
        )
        
        monitor = RiskMonitor(limits, initial_capital=100000.0)
        
        # Test start
        assert await monitor.start()
        
        # Test risk checks
        alerts = await monitor.check_limits()
        assert isinstance(alerts, list)
        
        # Test portfolio update
        monitor.update_portfolio(total_value=100000.0, positions={'AAPL': 50000.0}, cash=50000.0)
        
        # Test metrics
        assert monitor.current_metrics is not None
        assert monitor.current_metrics.total_value == 100000.0
        
        # Test stop
        await monitor.stop()


class TestConfiguration:
    """Test configuration management."""
    
    def test_broker_config_creation(self):
        """Test BrokerConfig creation."""
        config = BrokerConfig(
            broker_type=BrokerType.SIMULATED,
            api_key="test_key",
            secret_key="test_secret",
            paper_trading=True
        )
        
        assert config.broker_type == BrokerType.SIMULATED
        assert config.api_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.paper_trading is True
    
    def test_live_trading_config_creation(self):
        """Test LiveTradingConfig creation."""
        broker_config = BrokerConfig(
            broker_type=BrokerType.SIMULATED,
            api_key="test_key",
            secret_key="test_secret"
        )
        
        data_feed_config = DataFeedConfig(
            feed_type=DataFeedType.SIMULATED,
            symbols=["AAPL", "GOOGL"]
        )
        
        engine_config = EngineConfig(
            initial_capital=100000
        )
        
        config = LiveTradingConfig(
            environment=Environment.DEVELOPMENT,
            broker=broker_config,
            data_feed=data_feed_config,
            engine=engine_config
        )
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.broker.broker_type == BrokerType.SIMULATED
        assert config.data_feed.feed_type == DataFeedType.SIMULATED
        assert config.engine.initial_capital == 100000
    
    def test_config_manager(self):
        """Test configuration manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            
            manager = ConfigManager(str(config_file))
            
            # Create default config
            config = manager.create_default_config()
            assert config.environment == Environment.DEVELOPMENT
            assert config.engine.initial_capital == 100000
            
            # Save config
            manager.save_config(config, "test_config")
            assert config_file.exists()
            
            # Load config
            loaded_config = manager.load_config()
            assert loaded_config.environment == config.environment
            assert loaded_config.initial_capital == config.initial_capital


class TestLoggingAndAlerts:
    """Test logging and alerting components."""
    
    def test_log_entry_creation(self):
        """Test LogEntry creation."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            component="test",
            message="Test message",
            data={"key": "value"}
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.component == "test"
        assert entry.message == "Test message"
        assert entry.data["key"] == "value"
        
        # Test serialization
        json_str = entry.to_json()
        assert "Test message" in json_str
        assert "test" in json_str
    
    def test_trade_log_entry(self):
        """Test TradeLogEntry creation."""
        entry = TradeLogEntry(
            timestamp=datetime.now(),
            trade_id="trade_001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=1.0,
            pnl=500.0
        )
        
        assert entry.trade_id == "trade_001"
        assert entry.symbol == "AAPL"
        assert entry.side == "BUY"
        assert entry.quantity == 100
        assert entry.price == 150.0
        assert entry.pnl == 500.0
        
        # Test CSV conversion
        csv_row = entry.to_csv_row()
        assert "trade_001" in csv_row
        assert "AAPL" in csv_row
        assert "BUY" in csv_row
    
    def test_structured_logger(self):
        """Test structured logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger("test", temp_dir)
            
            # Test logging
            logger.info("Test message", data={"key": "value"})
            logger.error("Error message", data={"error": "details"})
            
            # Check log file exists
            log_files = list(Path(temp_dir).glob("*.jsonl"))
            assert len(log_files) > 0
    
    @pytest.mark.asyncio
    async def test_alert_manager(self):
        """Test alert manager."""
        manager = AlertManager([AlertChannel.LOG])
        
        # Test start
        assert await manager.start()
        
        # Test alert sending
        success = await manager.send_alert(
            message="Test alert",
            level=RiskLevel.MEDIUM,
            alert_type="test",
            data={"key": "value"}
        )
        assert success
        
        # Test alert history
        history = manager.get_alert_history()
        assert len(history) > 0
        assert history[0]['message'] == "Test alert"
        
        # Test statistics
        stats = manager.get_alert_statistics()
        assert stats['total_alerts'] > 0
        
        # Test stop
        await manager.stop()


class TestTradingEngine:
    """Test live trading engine."""
    
    @pytest.mark.asyncio
    async def test_engine_creation(self):
        """Test engine creation and configuration."""
        # Create mock components
        data_handler = Mock(spec=DataHandler)
        portfolio = Mock(spec=BacktestPortfolio)
        strategy = Mock(spec=BaseStrategy)
        
        # Create config
        config = EngineConfig(
            symbols=["AAPL", "GOOGL"],
            initial_capital=100000,
            update_interval=1.0
        )
        
        # Create engine
        engine = LiveTradingEngine(
            config=config,
            data_handler=data_handler,
            portfolio=portfolio,
            strategy=strategy
        )
        
        assert engine.config.initial_capital == 100000
        assert engine.config.symbols == ["AAPL", "GOOGL"]
        assert engine.state == EngineState.STOPPED
    
    @pytest.mark.asyncio
    async def test_engine_lifecycle(self):
        """Test engine start/stop lifecycle."""
        # Create mock components
        data_handler = Mock(spec=DataHandler)
        data_handler.start = AsyncMock(return_value=True)
        data_handler.stop = AsyncMock()
        
        portfolio = Mock(spec=BacktestPortfolio)
        strategy = Mock(spec=BaseStrategy)
        strategy.start = AsyncMock(return_value=True)
        strategy.stop = AsyncMock()
        
        config = EngineConfig(
            symbols=["AAPL"],
            initial_capital=100000,
            update_interval=0.1
        )
        
        engine = LiveTradingEngine(
            config=config,
            data_handler=data_handler,
            portfolio=portfolio,
            strategy=strategy
        )
        
        # Test start
        assert await engine.start()
        assert engine.state == EngineState.RUNNING
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Test stop
        await engine.stop()
        assert engine.state == EngineState.STOPPED
        
        # Verify components were called
        data_handler.start.assert_called_once()
        strategy.start.assert_called_once()
        data_handler.stop.assert_called_once()
        strategy.stop.assert_called_once()


class TestIntegration:
    """Integration tests for live trading system."""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration with all components."""
        # Create configuration
        config_manager = ConfigManager()
        config = config_manager.create_default_config()
        config.data_feed.symbols = ["AAPL"]
        config.data_feed.update_interval = 0.1
        
        # Create data feed
        data_feed = SimulatedDataFeed(
            symbols=config.data_feed.symbols,
            update_interval=config.data_feed.update_interval
        )
        data_feed_manager = DataFeedManager()
        data_feed_manager.add_feed("main", data_feed)
        
        # Create broker and order manager
        broker = SimulatedBroker()
        order_manager = OrderManager(broker)
        
        # Create mock portfolio and strategy
        portfolio = Mock(spec=BacktestPortfolio)
        portfolio.current_holdings = {'total': 100000, 'cash': 100000}
        portfolio.current_positions = {}
        portfolio.initial_capital = 100000
        
        strategy = Mock(spec=BaseStrategy)
        strategy.start = AsyncMock(return_value=True)
        strategy.stop = AsyncMock()
        strategy.calculate_signals = Mock(return_value=[])
        
        # Create risk monitor
        risk_limits = RiskLimits(
            max_position_size=50000,
            max_daily_loss=10000
        )
        risk_monitor = RiskMonitor(risk_limits, portfolio)
        
        # Create alert manager
        alert_manager = AlertManager([AlertChannel.LOG])
        
        # Create logging system
        with tempfile.TemporaryDirectory() as temp_dir:
            logging_system = LoggingSystem(temp_dir)
            
            # Start all components
            assert await data_feed_manager.start()
            assert await order_manager.start()
            assert await risk_monitor.start()
            assert await alert_manager.start()
            assert await logging_system.start()
            
            # Let system run briefly
            await asyncio.sleep(0.3)
            
            # Test data flow
            market_data = await data_feed_manager.get_latest_data("AAPL")
            assert market_data is not None
            assert market_data.symbol == "AAPL"
            
            # Test order submission
            order_id = await order_manager.submit_market_order("AAPL", "BUY", 10)
            assert order_id is not None
            
            # Wait for order processing
            await asyncio.sleep(0.2)
            
            # Check order status
            order = order_manager.get_order(order_id)
            assert order is not None
            assert order.status == OrderStatus.FILLED
            
            # Test risk monitoring
            risk_metrics = risk_monitor.get_current_metrics()
            assert risk_metrics is not None
            
            # Test alert sending
            alert_success = await alert_manager.send_alert(
                "Integration test alert",
                RiskLevel.LOW,
                "test"
            )
            assert alert_success
            
            # Stop all components
            await logging_system.stop()
            await alert_manager.stop()
            await risk_monitor.stop()
            await order_manager.stop()
            await data_feed_manager.stop()
            
            # Verify clean shutdown
            assert not data_feed_manager.is_connected()
    
    def test_configuration_validation(self):
        """Test configuration validation across components."""
        config_manager = ConfigManager()
        
        # Test default config
        config = config_manager.create_default_config()
        assert config_manager.validate_config(config)
        
        # Test invalid config
        config.initial_capital = -1000  # Invalid negative capital
        assert not config_manager.validate_config(config)
        
        # Test production config
        prod_config = config_manager.create_production_config()
        assert config_manager.validate_config(prod_config)
        assert prod_config.environment == Environment.PRODUCTION


if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", __file__])