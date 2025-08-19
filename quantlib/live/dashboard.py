"""Live trading monitoring dashboard.

This module provides a web-based dashboard for monitoring live trading operations:
- Real-time portfolio metrics
- Position monitoring
- Risk metrics visualization
- Order and trade history
- System health monitoring
- Performance analytics
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None

from ..core.base import BaseComponent
from ..backtesting.portfolio import Portfolio
from .engine import LiveTradingEngine, EngineMetrics
from .risk_monitor import RiskMonitor, RiskMetrics
from .order_manager import OrderManager, LiveOrder
from .data_feeds import DataFeedManager
from .logging_alerts import AlertManager


@dataclass
class DashboardMetrics:
    """Dashboard metrics snapshot."""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    daily_pnl: float
    positions_count: int
    active_orders: int
    filled_orders: int
    risk_score: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_pnl: float
    system_uptime: float
    data_feed_status: str
    broker_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PositionSnapshot:
    """Position snapshot for dashboard."""
    symbol: str
    quantity: int
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    avg_cost: float
    current_price: float
    weight: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class OrderSnapshot:
    """Order snapshot for dashboard."""
    order_id: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    price: Optional[float]
    order_type: str
    status: str
    timestamp: datetime
    time_in_force: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class DashboardDataCollector:
    """Collects data for dashboard display."""
    
    def __init__(
        self,
        engine: LiveTradingEngine,
        risk_monitor: RiskMonitor,
        order_manager: OrderManager,
        data_feed_manager: DataFeedManager,
        alert_manager: AlertManager
    ):
        self.engine = engine
        self.risk_monitor = risk_monitor
        self.order_manager = order_manager
        self.data_feed_manager = data_feed_manager
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Historical data
        self.metrics_history: List[DashboardMetrics] = []
        self.max_history_length = 1000
        
        # Start time for uptime calculation
        self.start_time = datetime.now()
    
    def collect_metrics(self) -> DashboardMetrics:
        """Collect current metrics snapshot."""
        try:
            # Get portfolio metrics
            portfolio = self.engine.portfolio
            portfolio_value = portfolio.current_holdings.get('total', 0.0)
            
            # Calculate PnL
            total_pnl = portfolio_value - portfolio.initial_capital
            
            # Daily PnL (simplified - would need proper daily tracking)
            daily_pnl = 0.0
            if len(self.metrics_history) > 0:
                yesterday_value = self.metrics_history[-1].portfolio_value if self.metrics_history else portfolio.initial_capital
                daily_pnl = portfolio_value - yesterday_value
            
            # Position metrics
            positions = self._get_positions()
            positions_count = len([p for p in positions if p.quantity != 0])
            
            # Order metrics
            orders = self.order_manager.get_all_orders()
            active_orders = len([o for o in orders if o.status.value in ['pending', 'partial']])
            filled_orders = len([o for o in orders if o.status.value == 'filled'])
            
            # Risk metrics
            risk_metrics = self.risk_monitor.get_current_metrics()
            risk_score = self._calculate_risk_score(risk_metrics)
            
            # Performance metrics (simplified)
            max_drawdown = risk_metrics.max_drawdown if risk_metrics else 0.0
            sharpe_ratio = 0.0  # Would need returns history
            win_rate = self._calculate_win_rate()
            avg_trade_pnl = self._calculate_avg_trade_pnl()
            
            # System metrics
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
            data_feed_status = "connected" if self.data_feed_manager.is_connected() else "disconnected"
            broker_status = "connected"  # Would check actual broker connection
            
            metrics = DashboardMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                positions_count=positions_count,
                active_orders=active_orders,
                filled_orders=filled_orders,
                risk_score=risk_score,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                avg_trade_pnl=avg_trade_pnl,
                system_uptime=uptime,
                data_feed_status=data_feed_status,
                broker_status=broker_status
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_length:
                self.metrics_history = self.metrics_history[-self.max_history_length:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            # Return default metrics
            return DashboardMetrics(
                timestamp=datetime.now(),
                portfolio_value=0.0,
                total_pnl=0.0,
                daily_pnl=0.0,
                positions_count=0,
                active_orders=0,
                filled_orders=0,
                risk_score=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                avg_trade_pnl=0.0,
                system_uptime=0.0,
                data_feed_status="unknown",
                broker_status="unknown"
            )
    
    def _get_positions(self) -> List[PositionSnapshot]:
        """Get current positions."""
        positions = []
        portfolio = self.engine.portfolio
        
        for symbol, quantity in portfolio.current_positions.items():
            if quantity == 0:
                continue
                
            # Get current price (simplified)
            current_price = portfolio.current_holdings.get(symbol, 0.0) / quantity if quantity != 0 else 0.0
            market_value = current_price * quantity
            
            # Calculate PnL (simplified)
            avg_cost = 0.0  # Would need cost basis tracking
            unrealized_pnl = 0.0
            realized_pnl = 0.0
            
            # Portfolio weight
            total_value = portfolio.current_holdings.get('total', 1.0)
            weight = abs(market_value) / total_value if total_value > 0 else 0.0
            
            position = PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                avg_cost=avg_cost,
                current_price=current_price,
                weight=weight,
                last_updated=datetime.now()
            )
            positions.append(position)
        
        return positions
    
    def _get_recent_orders(self, limit: int = 50) -> List[OrderSnapshot]:
        """Get recent orders."""
        orders = self.order_manager.get_all_orders()
        
        # Sort by timestamp (newest first)
        orders.sort(key=lambda o: o.timestamp, reverse=True)
        
        order_snapshots = []
        for order in orders[:limit]:
            snapshot = OrderSnapshot(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.filled_quantity,
                price=order.price,
                order_type=order.order_type.value,
                status=order.status.value,
                timestamp=order.timestamp,
                time_in_force=order.time_in_force.value
            )
            order_snapshots.append(snapshot)
        
        return order_snapshots
    
    def _calculate_risk_score(self, risk_metrics: Optional[RiskMetrics]) -> float:
        """Calculate overall risk score (0-100)."""
        if not risk_metrics:
            return 0.0
        
        # Simple risk score calculation
        score = 0.0
        
        # Factor in various risk metrics
        if risk_metrics.var_95 > 0:
            score += min(risk_metrics.var_95 * 100, 30)  # VaR contribution
        
        if risk_metrics.max_drawdown > 0:
            score += min(risk_metrics.max_drawdown * 100, 25)  # Drawdown contribution
        
        if risk_metrics.portfolio_beta > 1.5:
            score += 20  # High beta penalty
        
        if risk_metrics.concentration_risk > 0.3:
            score += 25  # Concentration penalty
        
        return min(score, 100.0)
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        # Simplified - would need actual trade history
        return 0.0
    
    def _calculate_avg_trade_pnl(self) -> float:
        """Calculate average trade PnL."""
        # Simplified - would need actual trade history
        return 0.0
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        return [m.to_dict() for m in filtered_metrics]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        return self.alert_manager.get_alert_statistics()


class LiveTradingDashboard(BaseComponent):
    """Web-based dashboard for live trading monitoring."""
    
    def __init__(
        self,
        engine: LiveTradingEngine,
        risk_monitor: RiskMonitor,
        order_manager: OrderManager,
        data_feed_manager: DataFeedManager,
        alert_manager: AlertManager,
        host: str = "localhost",
        port: int = 5000,
        debug: bool = False
    ):
        super().__init__()
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and Flask-SocketIO are required for dashboard functionality")
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize data collector
        self.data_collector = DashboardDataCollector(
            engine, risk_monitor, order_manager, data_feed_manager, alert_manager
        )
        
        # Initialize Flask app
        self.app = Flask(__name__, template_folder=self._get_template_dir())
        self.app.config['SECRET_KEY'] = 'quantlib_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        # Update task
        self.update_task = None
        self.update_interval = 1.0  # seconds
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_template_dir(self) -> str:
        """Get template directory path."""
        return str(Path(__file__).parent / "templates")
    
    def _setup_routes(self) -> None:
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current metrics."""
            metrics = self.data_collector.collect_metrics()
            return jsonify(metrics.to_dict())
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions."""
            positions = self.data_collector._get_positions()
            return jsonify([p.to_dict() for p in positions])
        
        @self.app.route('/api/orders')
        def get_orders():
            """Get recent orders."""
            limit = request.args.get('limit', 50, type=int)
            orders = self.data_collector._get_recent_orders(limit)
            return jsonify([o.to_dict() for o in orders])
        
        @self.app.route('/api/history')
        def get_history():
            """Get metrics history."""
            hours = request.args.get('hours', 24, type=int)
            history = self.data_collector.get_metrics_history(hours)
            return jsonify(history)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get alert summary."""
            alerts = self.data_collector.get_alert_summary()
            return jsonify(alerts)
        
        @self.app.route('/api/system')
        def get_system_info():
            """Get system information."""
            return jsonify({
                'uptime': (datetime.now() - self.data_collector.start_time).total_seconds(),
                'version': '1.0.0',
                'environment': 'live',
                'components': {
                    'engine': 'running',
                    'risk_monitor': 'running',
                    'order_manager': 'running',
                    'data_feeds': 'running',
                    'alerts': 'running'
                }
            })
    
    def _setup_socketio_events(self) -> None:
        """Setup SocketIO events for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.logger.info("Client connected to dashboard")
            # Send initial data
            metrics = self.data_collector.collect_metrics()
            emit('metrics_update', metrics.to_dict())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.logger.info("Client disconnected from dashboard")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle manual update request."""
            metrics = self.data_collector.collect_metrics()
            emit('metrics_update', metrics.to_dict())
    
    async def start(self) -> bool:
        """Start dashboard server."""
        try:
            # Create templates directory if it doesn't exist
            template_dir = Path(self._get_template_dir())
            template_dir.mkdir(exist_ok=True)
            
            # Create basic dashboard template if it doesn't exist
            self._create_dashboard_template()
            
            # Start update task
            self.update_task = asyncio.create_task(self._update_loop())
            
            # Start Flask app in a separate thread
            import threading
            self.server_thread = threading.Thread(
                target=lambda: self.socketio.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    debug=self.debug,
                    use_reloader=False
                )
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.logger.info(f"Dashboard started at http://{self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop dashboard server."""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Dashboard stopped")
    
    async def _update_loop(self) -> None:
        """Real-time update loop."""
        while True:
            try:
                # Collect metrics
                metrics = self.data_collector.collect_metrics()
                
                # Broadcast to all connected clients
                self.socketio.emit('metrics_update', metrics.to_dict())
                
                # Also send positions and orders periodically
                if int(time.time()) % 5 == 0:  # Every 5 seconds
                    positions = self.data_collector._get_positions()
                    orders = self.data_collector._get_recent_orders(10)
                    
                    self.socketio.emit('positions_update', [p.to_dict() for p in positions])
                    self.socketio.emit('orders_update', [o.to_dict() for o in orders])
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _create_dashboard_template(self) -> None:
        """Create basic dashboard HTML template."""
        template_path = Path(self._get_template_dir()) / "dashboard.html"
        
        if template_path.exists():
            return
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantLib Live Trading Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .metric-value {
            font-weight: bold;
            color: #007bff;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .status-connected { color: #28a745; }
        .status-disconnected { color: #dc3545; }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #333;
            margin-bottom: 5px;
        }
        .header .subtitle {
            color: #666;
            font-size: 14px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>QuantLib Live Trading Dashboard</h1>
        <div class="subtitle">Real-time monitoring and analytics</div>
    </div>
    
    <div class="dashboard">
        <!-- Portfolio Overview -->
        <div class="card">
            <h3>Portfolio Overview</h3>
            <div class="metric">
                <span>Portfolio Value:</span>
                <span class="metric-value" id="portfolio-value">$0.00</span>
            </div>
            <div class="metric">
                <span>Total P&L:</span>
                <span class="metric-value" id="total-pnl">$0.00</span>
            </div>
            <div class="metric">
                <span>Daily P&L:</span>
                <span class="metric-value" id="daily-pnl">$0.00</span>
            </div>
            <div class="metric">
                <span>Positions:</span>
                <span class="metric-value" id="positions-count">0</span>
            </div>
        </div>
        
        <!-- Risk Metrics -->
        <div class="card">
            <h3>Risk Metrics</h3>
            <div class="metric">
                <span>Risk Score:</span>
                <span class="metric-value" id="risk-score">0</span>
            </div>
            <div class="metric">
                <span>Max Drawdown:</span>
                <span class="metric-value" id="max-drawdown">0.00%</span>
            </div>
            <div class="metric">
                <span>Sharpe Ratio:</span>
                <span class="metric-value" id="sharpe-ratio">0.00</span>
            </div>
            <div class="metric">
                <span>Win Rate:</span>
                <span class="metric-value" id="win-rate">0.00%</span>
            </div>
        </div>
        
        <!-- System Status -->
        <div class="card">
            <h3>System Status</h3>
            <div class="metric">
                <span>Uptime:</span>
                <span class="metric-value" id="uptime">0.0h</span>
            </div>
            <div class="metric">
                <span>Data Feed:</span>
                <span class="metric-value" id="data-feed-status">Unknown</span>
            </div>
            <div class="metric">
                <span>Broker:</span>
                <span class="metric-value" id="broker-status">Unknown</span>
            </div>
            <div class="metric">
                <span>Active Orders:</span>
                <span class="metric-value" id="active-orders">0</span>
            </div>
        </div>
        
        <!-- Portfolio Chart -->
        <div class="card" style="grid-column: span 2;">
            <h3>Portfolio Value</h3>
            <div class="chart-container">
                <canvas id="portfolio-chart"></canvas>
            </div>
        </div>
        
        <!-- Recent Orders -->
        <div class="card" style="grid-column: span 2;">
            <h3>Recent Orders</h3>
            <table id="orders-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // Chart setup
        const ctx = document.getElementById('portfolio-chart').getContext('2d');
        const portfolioChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        // Update functions
        function updateMetrics(data) {
            document.getElementById('portfolio-value').textContent = '$' + data.portfolio_value.toFixed(2);
            
            const totalPnl = document.getElementById('total-pnl');
            totalPnl.textContent = '$' + data.total_pnl.toFixed(2);
            totalPnl.className = 'metric-value ' + (data.total_pnl >= 0 ? 'positive' : 'negative');
            
            const dailyPnl = document.getElementById('daily-pnl');
            dailyPnl.textContent = '$' + data.daily_pnl.toFixed(2);
            dailyPnl.className = 'metric-value ' + (data.daily_pnl >= 0 ? 'positive' : 'negative');
            
            document.getElementById('positions-count').textContent = data.positions_count;
            document.getElementById('risk-score').textContent = data.risk_score.toFixed(1);
            document.getElementById('max-drawdown').textContent = (data.max_drawdown * 100).toFixed(2) + '%';
            document.getElementById('sharpe-ratio').textContent = data.sharpe_ratio.toFixed(2);
            document.getElementById('win-rate').textContent = (data.win_rate * 100).toFixed(1) + '%';
            document.getElementById('uptime').textContent = data.system_uptime.toFixed(1) + 'h';
            
            const dataFeedStatus = document.getElementById('data-feed-status');
            dataFeedStatus.textContent = data.data_feed_status;
            dataFeedStatus.className = 'metric-value status-' + data.data_feed_status;
            
            const brokerStatus = document.getElementById('broker-status');
            brokerStatus.textContent = data.broker_status;
            brokerStatus.className = 'metric-value status-' + data.broker_status;
            
            document.getElementById('active-orders').textContent = data.active_orders;
            
            // Update chart
            const time = new Date(data.timestamp).toLocaleTimeString();
            portfolioChart.data.labels.push(time);
            portfolioChart.data.datasets[0].data.push(data.portfolio_value);
            
            // Keep only last 50 points
            if (portfolioChart.data.labels.length > 50) {
                portfolioChart.data.labels.shift();
                portfolioChart.data.datasets[0].data.shift();
            }
            
            portfolioChart.update('none');
        }
        
        function updateOrders(orders) {
            const tbody = document.querySelector('#orders-table tbody');
            tbody.innerHTML = '';
            
            orders.slice(0, 10).forEach(order => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${new Date(order.timestamp).toLocaleTimeString()}</td>
                    <td>${order.symbol}</td>
                    <td>${order.side}</td>
                    <td>${order.quantity}</td>
                    <td>${order.price ? '$' + order.price.toFixed(2) : 'Market'}</td>
                    <td>${order.status}</td>
                `;
            });
        }
        
        // Socket event handlers
        socket.on('metrics_update', updateMetrics);
        socket.on('orders_update', updateOrders);
        
        socket.on('connect', function() {
            console.log('Connected to dashboard');
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from dashboard');
        });
        
        // Request initial data
        socket.emit('request_update');
    </script>
</body>
</html>
        """
        
        with open(template_path, 'w') as f:
            f.write(html_content)
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.host}:{self.port}"