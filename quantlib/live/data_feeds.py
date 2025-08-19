"""Real-time market data feeds for live trading.

This module provides real-time market data capabilities including:
- WebSocket connections to market data providers
- Data normalization and validation
- Subscription management
- Reconnection handling
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

import websockets
import pandas as pd
from ..backtesting.events import MarketEvent
from ..core.base import BaseComponent


class DataFeedStatus(Enum):
    """Data feed connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class MarketData:
    """Real-time market data structure."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: int = 0
    ask_size: int = 0
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'volume': self.volume,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'mid_price': self.mid_price,
            'spread': self.spread
        }


class LiveDataFeed(BaseComponent, ABC):
    """Abstract base class for live data feeds."""
    
    def __init__(self, symbols: List[str], name: str = "LiveDataFeed"):
        super().__init__(name=name)
        self.symbols = symbols
        self.status = DataFeedStatus.DISCONNECTED
        self.subscribers: List[Callable[[MarketData], None]] = []
        self.last_data: Dict[str, MarketData] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data feed."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data feed."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbols."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        pass
    
    def add_subscriber(self, callback: Callable[[MarketData], None]) -> None:
        """Add data subscriber."""
        self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable[[MarketData], None]) -> None:
        """Remove data subscriber."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, data: MarketData) -> None:
        """Notify all subscribers of new data."""
        self.last_data[data.symbol] = data
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error notifying subscriber: {e}")
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data for symbol."""
        return self.last_data.get(symbol)
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.status == DataFeedStatus.CONNECTED


class WebSocketDataFeed(LiveDataFeed):
    """WebSocket-based market data feed."""
    
    def __init__(self, symbols: List[str], ws_url: str, api_key: Optional[str] = None):
        super().__init__(symbols)
        self.ws_url = ws_url
        self.api_key = api_key
        self.websocket = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        self._running = False
        
    async def connect(self) -> bool:
        """Connect to WebSocket feed."""
        try:
            self.status = DataFeedStatus.CONNECTING
            self.logger.info(f"Connecting to WebSocket: {self.ws_url}")
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.status = DataFeedStatus.CONNECTED
            self.reconnect_attempts = 0
            self.logger.info("WebSocket connected successfully")
            
            # Start listening for messages
            asyncio.create_task(self._listen())
            
            # Subscribe to symbols
            await self.subscribe(self.symbols)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            self.status = DataFeedStatus.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.status = DataFeedStatus.DISCONNECTED
        self.logger.info("WebSocket disconnected")
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbols."""
        if not self.is_connected():
            return False
        
        try:
            # Generic subscription message format
            subscribe_msg = {
                "action": "subscribe",
                "symbols": symbols,
                "channels": ["quotes", "trades"]
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
            self.logger.info(f"Subscribed to symbols: {symbols}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to symbols: {e}")
            return False
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        if not self.is_connected():
            return False
        
        try:
            unsubscribe_msg = {
                "action": "unsubscribe",
                "symbols": symbols
            }
            
            await self.websocket.send(json.dumps(unsubscribe_msg))
            self.logger.info(f"Unsubscribed from symbols: {symbols}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from symbols: {e}")
            return False
    
    async def _listen(self) -> None:
        """Listen for WebSocket messages."""
        self._running = True
        
        try:
            while self._running and self.websocket:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=60
                    )
                    await self._process_message(message)
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.websocket:
                        await self.websocket.ping()
                    
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed")
                    await self._handle_reconnection()
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in WebSocket listener: {e}")
            await self._handle_reconnection()
    
    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if data.get('type') == 'quote':
                market_data = self._parse_quote_data(data)
                if market_data:
                    self._notify_subscribers(market_data)
                    
            elif data.get('type') == 'trade':
                market_data = self._parse_trade_data(data)
                if market_data:
                    self._notify_subscribers(market_data)
                    
            elif data.get('type') == 'error':
                self.logger.error(f"WebSocket error: {data.get('message')}")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    def _parse_quote_data(self, data: Dict) -> Optional[MarketData]:
        """Parse quote data from WebSocket message."""
        try:
            return MarketData(
                symbol=data['symbol'],
                timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                bid=float(data['bid']),
                ask=float(data['ask']),
                last=float(data.get('last', (data['bid'] + data['ask']) / 2)),
                volume=int(data.get('volume', 0)),
                bid_size=int(data.get('bid_size', 0)),
                ask_size=int(data.get('ask_size', 0))
            )
        except (KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse quote data: {e}")
            return None
    
    def _parse_trade_data(self, data: Dict) -> Optional[MarketData]:
        """Parse trade data from WebSocket message."""
        try:
            # For trade data, we might not have bid/ask, so use last price
            last_price = float(data['price'])
            return MarketData(
                symbol=data['symbol'],
                timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                bid=last_price,  # Approximate
                ask=last_price,  # Approximate
                last=last_price,
                volume=int(data['size'])
            )
        except (KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse trade data: {e}")
            return None
    
    async def _handle_reconnection(self) -> None:
        """Handle WebSocket reconnection."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            self.status = DataFeedStatus.ERROR
            return
        
        self.status = DataFeedStatus.RECONNECTING
        self.reconnect_attempts += 1
        
        self.logger.info(
            f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}"
        )
        
        await asyncio.sleep(self.reconnect_delay)
        
        if await self.connect():
            self.logger.info("Reconnection successful")
        else:
            await self._handle_reconnection()


class SimulatedDataFeed(LiveDataFeed):
    """Simulated data feed for testing."""
    
    def __init__(self, symbols: List[str], data_file: Optional[str] = None, update_interval: float = 1.0):
        super().__init__(symbols, name="SimulatedDataFeed")
        self.data_file = data_file
        self.update_interval = update_interval
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_index = 0
        self._running = False
    
    async def start(self) -> bool:
        """Start the simulated data feed."""
        return await self.connect()
    
    async def stop(self) -> None:
        """Stop the simulated data feed."""
        await self.disconnect()
        
    async def connect(self) -> bool:
        """Connect to simulated feed."""
        try:
            self.status = DataFeedStatus.CONNECTING
            
            if self.data_file:
                self._load_historical_data()
            else:
                self._generate_sample_data()
            
            self.status = DataFeedStatus.CONNECTED
            self.logger.info("Simulated data feed connected")
            
            # Start data simulation
            asyncio.create_task(self._simulate_data())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect simulated feed: {e}")
            self.status = DataFeedStatus.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect simulated feed."""
        self._running = False
        self.status = DataFeedStatus.DISCONNECTED
        self.logger.info("Simulated data feed disconnected")
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbols."""
        self.symbols.extend([s for s in symbols if s not in self.symbols])
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
        return True
    
    def _load_historical_data(self) -> None:
        """Load historical data from file."""
        # Implementation would load actual historical data
        # For now, generate sample data
        self._generate_sample_data()
    
    def _generate_sample_data(self) -> None:
        """Generate sample market data."""
        import numpy as np
        
        for symbol in self.symbols:
            # Generate sample price data
            dates = pd.date_range(
                start=datetime.now(),
                periods=1000,
                freq='1min'
            )
            
            base_price = 100.0
            prices = base_price + np.cumsum(np.random.randn(1000) * 0.1)
            
            self.historical_data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'price': prices,
                'volume': np.random.randint(1000, 10000, 1000)
            })
    
    async def _simulate_data(self) -> None:
        """Simulate real-time data."""
        self._running = True
        
        while self._running:
            try:
                for symbol in self.symbols:
                    if symbol in self.historical_data:
                        df = self.historical_data[symbol]
                        
                        if self.current_index < len(df):
                            row = df.iloc[self.current_index]
                            
                            # Create market data with simulated bid/ask
                            price = row['price']
                            spread = price * 0.001  # 0.1% spread
                            
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                bid=price - spread/2,
                                ask=price + spread/2,
                                last=price,
                                volume=int(row['volume'])
                            )
                            
                            self._notify_subscribers(market_data)
                
                self.current_index += 1
                await asyncio.sleep(1)  # 1 second intervals
                
            except Exception as e:
                self.logger.error(f"Error in data simulation: {e}")
                await asyncio.sleep(1)


class DataFeedManager:
    """Manager for multiple data feeds."""
    
    def __init__(self):
        self.feeds: Dict[str, LiveDataFeed] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_feed(self, name: str, feed: LiveDataFeed) -> None:
        """Add a data feed."""
        self.feeds[name] = feed
        self.logger.info(f"Added data feed: {name}")
    
    async def start(self) -> bool:
        """Start all data feeds."""
        results = await self.connect_all()
        return all(results.values())
    
    async def stop(self) -> None:
        """Stop all data feeds."""
        await self.disconnect_all()
    
    def is_connected(self) -> bool:
        """Check if all feeds are connected."""
        return all(feed.status == DataFeedStatus.CONNECTED for feed in self.feeds.values())
    
    def remove_feed(self, name: str) -> None:
        """Remove a data feed."""
        if name in self.feeds:
            del self.feeds[name]
            self.logger.info(f"Removed data feed: {name}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all data feeds."""
        results = {}
        for name, feed in self.feeds.items():
            results[name] = await feed.connect()
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all data feeds."""
        for feed in self.feeds.values():
            await feed.disconnect()
    
    def get_feed(self, name: str) -> Optional[LiveDataFeed]:
        """Get data feed by name."""
        return self.feeds.get(name)
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data for symbol from any feed."""
        for feed in self.feeds.values():
            data = feed.get_latest_data(symbol)
            if data:
                return data
        return None
    
    def add_global_subscriber(self, callback: Callable[[MarketData], None]) -> None:
        """Add subscriber to all feeds."""
        for feed in self.feeds.values():
            feed.add_subscriber(callback)