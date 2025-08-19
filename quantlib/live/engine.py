"""Live trading engine.

This module provides the main live trading engine that orchestrates:
- Real-time data feeds
- Strategy execution
- Order management
- Risk monitoring
- Portfolio tracking
"""

import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..backtesting.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from ..core.base import BaseStrategy
from ..backtesting.portfolio import BacktestPortfolio
from ..core.base import BaseComponent
from .data_feeds import LiveDataFeed, MarketData
from .order_manager import OrderManager, LiveOrder, OrderStatus
from .risk_monitor import RiskMonitor


class EngineState(Enum):
    """Live trading engine state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineConfig:
    """Live trading engine configuration."""
    # Trading parameters
    symbols: List[str] = field(default_factory=list)
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.05  # 5% daily loss limit
    
    # Timing parameters
    heartbeat_interval: float = 1.0  # seconds
    update_interval: float = 1.0  # seconds
    data_timeout: float = 30.0  # seconds
    order_timeout: float = 300.0  # seconds
    
    # Risk parameters
    enable_risk_checks: bool = True
    max_orders_per_minute: int = 60
    max_order_value: float = 50000.0
    
    # Logging
    log_level: str = "INFO"
    log_trades: bool = True
    log_orders: bool = True
    
    # Recovery
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay: float = 60.0  # seconds


@dataclass
class EngineMetrics:
    """Live trading engine metrics."""
    start_time: datetime = field(default_factory=datetime.now)
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    total_trades: int = 0
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    data_points_received: int = 0
    last_data_time: Optional[datetime] = None
    errors: int = 0
    restarts: int = 0
    
    def update_uptime(self) -> None:
        """Update uptime."""
        self.uptime = datetime.now() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': self.uptime.total_seconds(),
            'total_trades': self.total_trades,
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'order_success_rate': self.successful_orders / max(self.total_orders, 1),
            'data_points_received': self.data_points_received,
            'last_data_time': self.last_data_time.isoformat() if self.last_data_time else None,
            'errors': self.errors,
            'restarts': self.restarts
        }


class LiveTradingEngine(BaseComponent):
    """Live trading engine."""
    
    def __init__(
        self,
        config: EngineConfig,
        data_feed: LiveDataFeed,
        order_manager: OrderManager,
        strategy: BaseStrategy,
        portfolio: BacktestPortfolio,
        risk_monitor: Optional[RiskMonitor] = None
    ):
        super().__init__()
        self.config = config
        self.data_feed = data_feed
        self.order_manager = order_manager
        self.strategy = strategy
        self.portfolio = portfolio
        self.risk_monitor = risk_monitor
        
        self.state = EngineState.STOPPED
        self.metrics = EngineMetrics()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Event queues
        self.event_queue = asyncio.Queue()
        self.market_data_queue = asyncio.Queue()
        
        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self.state_callbacks: List[Callable[[EngineState], None]] = []
        self.trade_callbacks: List[Callable[[FillEvent], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Register callbacks
        self._setup_callbacks()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Signal handlers can only be set in main thread
            pass
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.stop())
    
    def _setup_callbacks(self) -> None:
        """Setup internal callbacks."""
        # Data feed callbacks
        self.data_feed.add_data_callback(self._on_market_data)
        self.data_feed.add_error_callback(self._on_data_error)
        
        # Order manager callbacks
        self.order_manager.add_order_callback(self._on_order_update)
        self.order_manager.add_fill_callback(self._on_fill)
    
    def add_state_callback(self, callback: Callable[[EngineState], None]) -> None:
        """Add state change callback."""
        self.state_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[FillEvent], None]) -> None:
        """Add trade callback."""
        self.trade_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add error callback."""
        self.error_callbacks.append(callback)
    
    async def start(self) -> bool:
        """Start the live trading engine."""
        try:
            self.logger.info("Starting live trading engine")
            self._set_state(EngineState.STARTING)
            
            # Start components
            if not await self._start_components():
                self._set_state(EngineState.ERROR)
                return False
            
            # Start main loop
            self._running = True
            self._set_state(EngineState.RUNNING)
            
            # Start background tasks
            asyncio.create_task(self._main_loop())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._metrics_loop())
            
            self.logger.info("Live trading engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start engine: {e}")
            self._set_state(EngineState.ERROR)
            self._notify_error_callbacks(e)
            return False
    
    async def stop(self) -> None:
        """Stop the live trading engine."""
        try:
            self.logger.info("Stopping live trading engine")
            self._set_state(EngineState.STOPPING)
            
            # Stop main loop
            self._running = False
            
            # Cancel all pending orders
            await self.order_manager.cancel_all_orders()
            
            # Stop components
            await self._stop_components()
            
            # Set shutdown event
            self._shutdown_event.set()
            
            self._set_state(EngineState.STOPPED)
            self.logger.info("Live trading engine stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping engine: {e}")
            self._notify_error_callbacks(e)
    
    async def pause(self) -> None:
        """Pause trading (stop new orders but keep monitoring)."""
        if self.state == EngineState.RUNNING:
            self._set_state(EngineState.PAUSED)
            self.logger.info("Trading paused")
    
    async def resume(self) -> None:
        """Resume trading."""
        if self.state == EngineState.PAUSED:
            self._set_state(EngineState.RUNNING)
            self.logger.info("Trading resumed")
    
    async def _start_components(self) -> bool:
        """Start all components."""
        try:
            # Start data feed
            if not await self.data_feed.start():
                self.logger.error("Failed to start data feed")
                return False
            
            # Start order manager
            if not await self.order_manager.start():
                self.logger.error("Failed to start order manager")
                return False
            
            # Start risk monitor
            if self.risk_monitor:
                if not await self.risk_monitor.start():
                    self.logger.error("Failed to start risk monitor")
                    return False
            
            # Subscribe to symbols
            for symbol in self.config.symbols:
                await self.data_feed.subscribe(symbol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting components: {e}")
            return False
    
    async def _stop_components(self) -> None:
        """Stop all components."""
        try:
            # Stop data feed
            await self.data_feed.stop()
            
            # Stop order manager
            await self.order_manager.stop()
            
            # Stop risk monitor
            if self.risk_monitor:
                await self.risk_monitor.stop()
                
        except Exception as e:
            self.logger.error(f"Error stopping components: {e}")
    
    async def _main_loop(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                # Process events with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=self.config.heartbeat_interval
                    )
                    await self._process_event(event)
                except asyncio.TimeoutError:
                    # No events to process, continue
                    pass
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.metrics.errors += 1
                self._notify_error_callbacks(e)
                
                # Sleep before continuing
                await asyncio.sleep(1)
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop for periodic tasks."""
        while self._running:
            try:
                await self._heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                self.metrics.errors += 1
                await asyncio.sleep(5)
    
    async def _metrics_loop(self) -> None:
        """Metrics update loop."""
        while self._running:
            try:
                self.metrics.update_uptime()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat(self) -> None:
        """Perform heartbeat tasks."""
        # Check data feed health
        if self.metrics.last_data_time:
            time_since_data = datetime.now() - self.metrics.last_data_time
            if time_since_data.total_seconds() > self.config.data_timeout:
                self.logger.warning(f"No data received for {time_since_data.total_seconds():.1f} seconds")
        
        # Update portfolio
        await self._update_portfolio()
        
        # Check risk limits
        if self.risk_monitor and self.config.enable_risk_checks:
            await self.risk_monitor.check_limits()
    
    async def _update_portfolio(self) -> None:
        """Update portfolio with current positions."""
        try:
            positions = await self.order_manager.get_positions()
            account_info = await self.order_manager.get_account_info()
            
            # Update portfolio (simplified)
            # In a real implementation, you'd update the portfolio with current market values
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    async def _process_event(self, event) -> None:
        """Process a single event."""
        try:
            if isinstance(event, MarketEvent):
                await self._process_market_event(event)
            elif isinstance(event, SignalEvent):
                await self._process_signal_event(event)
            elif isinstance(event, OrderEvent):
                await self._process_order_event(event)
            elif isinstance(event, FillEvent):
                await self._process_fill_event(event)
            else:
                self.logger.warning(f"Unknown event type: {type(event)}")
                
        except Exception as e:
            self.logger.error(f"Error processing event {type(event)}: {e}")
            self.metrics.errors += 1
    
    async def _process_market_event(self, event: MarketEvent) -> None:
        """Process market event."""
        if self.state != EngineState.RUNNING:
            return
        
        # Update strategy with market data
        signals = self.strategy.calculate_signals(event)
        
        # Queue signal events
        for signal in signals:
            await self.event_queue.put(signal)
    
    async def _process_signal_event(self, event: SignalEvent) -> None:
        """Process signal event."""
        if self.state != EngineState.RUNNING:
            return
        
        # Generate order from signal
        order_event = self.portfolio.update_signal(event)
        
        if order_event:
            await self.event_queue.put(order_event)
    
    async def _process_order_event(self, event: OrderEvent) -> None:
        """Process order event."""
        if self.state != EngineState.RUNNING:
            return
        
        # Risk checks
        if self.risk_monitor and self.config.enable_risk_checks:
            if not await self.risk_monitor.check_order(event):
                self.logger.warning(f"Order rejected by risk monitor: {event}")
                return
        
        # Submit order
        order_id = await self.order_manager.submit_order(
            symbol=event.symbol,
            side=event.direction,
            quantity=event.quantity,
            order_type=event.order_type,
            price=getattr(event, 'price', None)
        )
        
        if order_id:
            self.metrics.total_orders += 1
            self.metrics.successful_orders += 1
            
            if self.config.log_orders:
                self.logger.info(f"Order submitted: {order_id} - {event}")
        else:
            self.metrics.failed_orders += 1
            self.logger.error(f"Failed to submit order: {event}")
    
    async def _process_fill_event(self, event: FillEvent) -> None:
        """Process fill event."""
        # Update portfolio
        self.portfolio.update_fill(event)
        
        # Update metrics
        self.metrics.total_trades += 1
        
        # Log trade
        if self.config.log_trades:
            self.logger.info(f"Trade executed: {event}")
        
        # Notify callbacks
        self._notify_trade_callbacks(event)
    
    async def _on_market_data(self, data: MarketData) -> None:
        """Handle market data callback."""
        try:
            # Update metrics
            self.metrics.data_points_received += 1
            self.metrics.last_data_time = datetime.now()
            
            # Create market event
            market_event = MarketEvent(
                timestamp=data.timestamp,
                symbol=data.symbol,
                data={
                    'price': data.price,
                    'volume': data.volume,
                    'bid': data.bid,
                    'ask': data.ask
                }
            )
            
            # Queue event
            await self.event_queue.put(market_event)
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            self.metrics.errors += 1
    
    async def _on_data_error(self, error: Exception) -> None:
        """Handle data feed error."""
        self.logger.error(f"Data feed error: {error}")
        self.metrics.errors += 1
        self._notify_error_callbacks(error)
    
    async def _on_order_update(self, order: LiveOrder) -> None:
        """Handle order status update."""
        self.logger.debug(f"Order update: {order.order_id} - {order.status}")
    
    async def _on_fill(self, fill_event: FillEvent) -> None:
        """Handle fill event."""
        await self.event_queue.put(fill_event)
    
    def _set_state(self, new_state: EngineState) -> None:
        """Set engine state and notify callbacks."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.logger.info(f"Engine state changed: {old_state} -> {new_state}")
            
            for callback in self.state_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    self.logger.error(f"Error in state callback: {e}")
    
    def _notify_trade_callbacks(self, fill_event: FillEvent) -> None:
        """Notify trade callbacks."""
        for callback in self.trade_callbacks:
            try:
                callback(fill_event)
            except Exception as e:
                self.logger.error(f"Error in trade callback: {e}")
    
    def _notify_error_callbacks(self, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'state': self.state.value,
            'config': {
                'symbols': self.config.symbols,
                'initial_capital': self.config.initial_capital,
                'max_position_size': self.config.max_position_size,
                'enable_risk_checks': self.config.enable_risk_checks
            },
            'metrics': self.metrics.to_dict(),
            'data_feed_status': self.data_feed.get_status(),
            'order_stats': self.order_manager.get_order_statistics()
        }
    
    async def wait_for_shutdown(self) -> None:
        """Wait for engine shutdown."""
        await self._shutdown_event.wait()