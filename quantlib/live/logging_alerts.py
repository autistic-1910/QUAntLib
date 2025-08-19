"""Logging and alerting system for live trading.

This module provides comprehensive logging and alerting capabilities:
- Structured logging
- Trade and order logging
- Real-time alerts
- Email notifications
- Slack integration
- Performance monitoring
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

from ..backtesting.events import FillEvent, OrderEvent
from ..core.base import BaseComponent
from .risk_monitor import RiskAlert, RiskLevel
from .order_manager import LiveOrder


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertChannel(Enum):
    """Alert channel enumeration."""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'component': self.component,
            'message': self.message,
            'data': self.data,
            'correlation_id': self.correlation_id
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class TradeLogEntry:
    """Trade log entry."""
    timestamp: datetime
    trade_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    pnl: float = 0.0
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'pnl': self.pnl,
            'strategy': self.strategy,
            'metadata': self.metadata
        }
    
    def to_csv_row(self) -> str:
        """Convert to CSV row."""
        return f"{self.timestamp.isoformat()},{self.trade_id},{self.symbol},{self.side},{self.quantity},{self.price},{self.commission},{self.pnl},{self.strategy or ''}"


@dataclass
class OrderLogEntry:
    """Order log entry."""
    timestamp: datetime
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float]
    status: str
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'price': self.price,
            'status': self.status,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'metadata': self.metadata
        }


class StructuredLogger:
    """Structured logger with JSON output."""
    
    def __init__(self, name: str, log_dir: Optional[str] = None):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"structured_{name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # JSON file handler
        json_file = self.log_dir / f"{name}_structured.jsonl"
        json_handler = RotatingFileHandler(
            json_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(json_handler)
        self.json_handler = json_handler
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        # Track handlers for explicit closing
        self._handlers = [self.json_handler, console_handler]
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    def close(self):
        """Close all handlers and cleanup resources."""
        for handler in self._handlers:
            handler.close()
            self.logger.removeHandler(handler)
        self._handlers.clear()
    
    def log(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Log structured entry."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=self.name,
            message=message,
            data=data or {},
            correlation_id=correlation_id
        )
        
        # Log JSON to file
        log_level = getattr(logging, level.value)
        self.logger.log(log_level, entry.to_json())
        # Ensure file handler flushes to release file handles for tests cleanup
        for handler in self.logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close logger handlers to release file locks (Windows-friendly)."""
        for handler in getattr(self, '_handlers', []):
            try:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def close(self) -> None:
        """Close handlers to release file locks (important on Windows)."""
        for handler in getattr(self, '_handlers', []):
            try:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass


class TradeLogger:
    """Specialized logger for trades."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup trade logger
        self.logger = logging.getLogger("trade_logger")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # JSON file handler
        json_file = self.log_dir / "trades.jsonl"
        json_handler = TimedRotatingFileHandler(
            json_file,
            when='midnight',
            interval=1,
            backupCount=365
        )
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(json_handler)
        
        # CSV file handler
        csv_file = self.log_dir / "trades.csv"
        csv_handler = TimedRotatingFileHandler(
            csv_file,
            when='midnight',
            interval=1,
            backupCount=365
        )
        csv_handler.setFormatter(logging.Formatter('%(message)s'))
        self.csv_handler = csv_handler
        
        # Write CSV header if file doesn't exist
        if not csv_file.exists():
            with open(csv_file, 'w') as f:
                f.write("timestamp,trade_id,symbol,side,quantity,price,commission,pnl,strategy\n")

    def close(self) -> None:
        """Close underlying file handlers."""
        try:
            self.csv_handler.flush()
            self.csv_handler.close()
            self.logger.removeHandler(self.csv_handler)
        except Exception:
            pass
        for handler in self.logger.handlers[:]:
            try:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass
    
    def log_trade(self, fill_event: FillEvent, strategy: Optional[str] = None, pnl: float = 0.0) -> None:
        """Log trade execution."""
        entry = TradeLogEntry(
            timestamp=fill_event.timestamp,
            trade_id=f"trade_{int(time.time() * 1000)}",
            symbol=fill_event.symbol,
            side=fill_event.direction,
            quantity=fill_event.quantity,
            price=fill_event.fill_cost,
            commission=fill_event.commission,
            pnl=pnl,
            strategy=strategy
        )
        
        # Log JSON
        self.logger.info(json.dumps(entry.to_dict(), default=str))
        
        # Log CSV
        self.csv_handler.emit(logging.LogRecord(
            name="trade_csv",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=entry.to_csv_row(),
            args=(),
            exc_info=None
        ))


class OrderLogger:
    """Specialized logger for orders."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup order logger
        self.logger = logging.getLogger("order_logger")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # JSON file handler
        json_file = self.log_dir / "orders.jsonl"
        json_handler = TimedRotatingFileHandler(
            json_file,
            when='midnight',
            interval=1,
            backupCount=365
        )
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(json_handler)

    def close(self) -> None:
        for handler in self.logger.handlers[:]:
            try:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass
    
    def log_order(self, order: LiveOrder) -> None:
        """Log order status."""
        entry = OrderLogEntry(
            timestamp=datetime.now(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type.value,
            price=order.price,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            avg_fill_price=order.avg_fill_price,
            commission=order.commission,
            metadata=order.metadata
        )
        
        self.logger.info(json.dumps(entry.to_dict(), default=str))


class EmailAlerter:
    """Email alert sender."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def send_alert(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        alert_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send email alert."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username or "noreply@quantlib.com"
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"[QuantLib Alert] {subject}"
            
            # Create HTML body
            html_body = self._create_html_body(message, alert_data)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email, msg, recipients
            )
            
            self.logger.info(f"Email alert sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_email(self, msg: MIMEMultipart, recipients: List[str]) -> None:
        """Send email synchronously."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.send_message(msg, to_addrs=recipients)
    
    def _create_html_body(self, message: str, alert_data: Optional[Dict[str, Any]]) -> str:
        """Create HTML email body."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .data {{ background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h2>QuantLib Trading Alert</h2>
            <div class="alert">
                <p><strong>Message:</strong> {message}</p>
                <p class="timestamp"><strong>Time:</strong> {datetime.now().isoformat()}</p>
            </div>
        """
        
        if alert_data:
            html += "<div class='data'><h3>Alert Data:</h3><pre>"
            html += json.dumps(alert_data, indent=2, default=str)
            html += "</pre></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html


class SlackAlerter:
    """Slack alert sender."""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.channel = channel
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def send_alert(
        self,
        message: str,
        alert_data: Optional[Dict[str, Any]] = None,
        level: RiskLevel = RiskLevel.MEDIUM
    ) -> bool:
        """Send Slack alert."""
        try:
            # Create Slack message
            color = {
                RiskLevel.LOW: "good",
                RiskLevel.MEDIUM: "warning",
                RiskLevel.HIGH: "danger",
                RiskLevel.CRITICAL: "danger"
            }.get(level, "warning")
            
            payload = {
                "text": "QuantLib Trading Alert",
                "attachments": [
                    {
                        "color": color,
                        "title": f"Alert Level: {level.value.upper()}",
                        "text": message,
                        "timestamp": int(time.time()),
                        "fields": []
                    }
                ]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            # Add alert data as fields
            if alert_data:
                for key, value in alert_data.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Slack alert sent successfully")
                        return True
                    else:
                        self.logger.error(f"Slack alert failed: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertManager(BaseComponent):
    """Centralized alert management system."""
    
    def __init__(
        self,
        channels: List[AlertChannel],
        email_config: Optional[Dict[str, Any]] = None,
        slack_config: Optional[Dict[str, Any]] = None,
        name: str = "AlertManager"
    ):
        super().__init__(name)
        self.channels = channels
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize alerters
        self.email_alerter = None
        self.slack_alerter = None
        
        if AlertChannel.EMAIL in channels and email_config:
            self.email_alerter = EmailAlerter(**email_config)
        
        if AlertChannel.SLACK in channels and slack_config:
            self.slack_alerter = SlackAlerter(**slack_config)
        
        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Rate limiting
        self.alert_counts: Dict[str, List[datetime]] = {}
        self.rate_limit_window = timedelta(minutes=5)
        self.max_alerts_per_window = 10
    
    async def start(self) -> bool:
        """Start alert manager."""
        self.logger.info(f"Alert manager started with channels: {[c.value for c in self.channels]}")
        return True
    
    async def stop(self) -> None:
        """Stop alert manager."""
        self.logger.info("Alert manager stopped")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    async def send_alert(
        self,
        message: str,
        level: RiskLevel = RiskLevel.MEDIUM,
        alert_type: str = "general",
        data: Optional[Dict[str, Any]] = None,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Send alert through configured channels."""
        try:
            # Check rate limiting
            if not self._check_rate_limit(alert_type):
                self.logger.warning(f"Rate limit exceeded for alert type: {alert_type}")
                return False
            
            # Create alert record
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'level': level.value,
                'type': alert_type,
                'data': data or {},
                'channels_sent': []
            }
            
            success = True
            
            # Send through each channel
            for channel in self.channels:
                try:
                    if channel == AlertChannel.LOG:
                        self._send_log_alert(message, level, data)
                        alert_record['channels_sent'].append('log')
                    
                    elif channel == AlertChannel.EMAIL and self.email_alerter and recipients:
                        if await self.email_alerter.send_alert(recipients, f"{alert_type.title()} Alert", message, data):
                            alert_record['channels_sent'].append('email')
                        else:
                            success = False
                    
                    elif channel == AlertChannel.SLACK and self.slack_alerter:
                        if await self.slack_alerter.send_alert(message, data, level):
                            alert_record['channels_sent'].append('slack')
                        else:
                            success = False
                            
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel.value}: {e}")
                    success = False
            
            # Store alert history
            self.alert_history.append(alert_record)
            
            # Keep only recent alerts (last 1000)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_record)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False
    
    async def send_risk_alert(self, risk_alert: RiskAlert, recipients: Optional[List[str]] = None) -> bool:
        """Send risk-specific alert."""
        return await self.send_alert(
            message=risk_alert.message,
            level=risk_alert.level,
            alert_type=risk_alert.alert_type.value,
            data={
                'alert_id': risk_alert.alert_id,
                'symbol': risk_alert.symbol,
                'current_value': risk_alert.current_value,
                'limit_value': risk_alert.limit_value,
                'metadata': risk_alert.metadata
            },
            recipients=recipients
        )
    
    def _send_log_alert(self, message: str, level: RiskLevel, data: Optional[Dict[str, Any]]) -> None:
        """Send alert to log."""
        log_level = {
            RiskLevel.LOW: logging.INFO,
            RiskLevel.MEDIUM: logging.WARNING,
            RiskLevel.HIGH: logging.ERROR,
            RiskLevel.CRITICAL: logging.CRITICAL
        }.get(level, logging.WARNING)
        
        log_message = f"ALERT [{level.value.upper()}]: {message}"
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        
        self.logger.log(log_level, log_message)
    
    def _check_rate_limit(self, alert_type: str) -> bool:
        """Check if alert type is within rate limits."""
        now = datetime.now()
        
        # Initialize if not exists
        if alert_type not in self.alert_counts:
            self.alert_counts[alert_type] = []
        
        # Remove old alerts outside window
        cutoff_time = now - self.rate_limit_window
        self.alert_counts[alert_type] = [
            timestamp for timestamp in self.alert_counts[alert_type]
            if timestamp > cutoff_time
        ]
        
        # Check if under limit
        if len(self.alert_counts[alert_type]) >= self.max_alerts_per_window:
            return False
        
        # Add current alert
        self.alert_counts[alert_type].append(now)
        return True
    
    def get_alert_history(
        self,
        alert_type: Optional[str] = None,
        level: Optional[RiskLevel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get alert history with optional filters."""
        alerts = self.alert_history.copy()
        
        if alert_type:
            alerts = [a for a in alerts if a['type'] == alert_type]
        
        if level:
            alerts = [a for a in alerts if a['level'] == level.value]
        
        # Sort by timestamp (newest first) and limit
        alerts.sort(key=lambda a: a['timestamp'], reverse=True)
        return alerts[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.alert_history:
            return {'total_alerts': 0}
        
        # Count by level
        level_counts = {}
        type_counts = {}
        
        for alert in self.alert_history:
            level = alert['level']
            alert_type = alert['type']
            
            level_counts[level] = level_counts.get(level, 0) + 1
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        # Recent alerts (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            a for a in self.alert_history
            if datetime.fromisoformat(a['timestamp']) > recent_cutoff
        ]
        
        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts_1h': len(recent_alerts),
            'level_counts': level_counts,
            'type_counts': type_counts,
            'channels_configured': [c.value for c in self.channels]
        }


class LoggingSystem(BaseComponent):
    """Comprehensive logging system for live trading."""
    
    def __init__(self, log_dir: Optional[str] = None):
        super().__init__()
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.structured_logger = StructuredLogger("live_trading", str(self.log_dir))
        self.trade_logger = TradeLogger(str(self.log_dir))
        self.order_logger = OrderLogger(str(self.log_dir))
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def start(self) -> bool:
        """Start logging system."""
        self.logger.info(f"Logging system started, log directory: {self.log_dir}")
        return True
    
    async def stop(self) -> None:
        """Stop logging system."""
        try:
            self.structured_logger.close()
            self.trade_logger.close()
            self.order_logger.close()
        except Exception:
            pass
        self.logger.info("Logging system stopped")
    
    def log_trade(self, fill_event: FillEvent, strategy: Optional[str] = None, pnl: float = 0.0) -> None:
        """Log trade execution."""
        self.trade_logger.log_trade(fill_event, strategy, pnl)
        
        # Also log to structured logger
        self.structured_logger.info(
            f"Trade executed: {fill_event.symbol} {fill_event.direction} {fill_event.quantity} @ {fill_event.fill_cost}",
            data={
                'symbol': fill_event.symbol,
                'side': fill_event.direction,
                'quantity': fill_event.quantity,
                'price': fill_event.fill_cost,
                'commission': fill_event.commission,
                'pnl': pnl,
                'strategy': strategy
            }
        )
    
    def log_order(self, order: LiveOrder) -> None:
        """Log order status."""
        self.order_logger.log_order(order)
        
        # Also log to structured logger
        self.structured_logger.info(
            f"Order {order.status.value}: {order.symbol} {order.side} {order.quantity}",
            data=order.to_dict()
        )
    
    def log_system_event(
        self,
        component: str,
        event: str,
        level: LogLevel = LogLevel.INFO,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log system event."""
        logger = StructuredLogger(component, str(self.log_dir))
        logger.log(level, event, data)
    
    def get_log_files(self) -> Dict[str, List[str]]:
        """Get list of log files by category."""
        log_files = {
            'structured': [],
            'trades': [],
            'orders': [],
            'other': []
        }
        
        for log_file in self.log_dir.glob("*.log*"):
            filename = log_file.name
            if 'structured' in filename:
                log_files['structured'].append(filename)
            elif 'trade' in filename:
                log_files['trades'].append(filename)
            elif 'order' in filename:
                log_files['orders'].append(filename)
            else:
                log_files['other'].append(filename)
        
        # Sort files
        for category in log_files:
            log_files[category].sort(reverse=True)
        
        return log_files