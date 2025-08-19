"""Configuration management for live trading.

This module provides configuration management for:
- Trading parameters
- Risk limits
- Broker settings
- Data feed configuration
- Logging and monitoring
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from .risk_monitor import RiskLimits
from .engine import EngineConfig


class Environment(Enum):
    """Trading environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class BrokerType(Enum):
    """Broker type enumeration."""
    SIMULATED = "simulated"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    TD_AMERITRADE = "td_ameritrade"
    BINANCE = "binance"
    CUSTOM = "custom"


class DataFeedType(Enum):
    """Data feed type enumeration."""
    SIMULATED = "simulated"
    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    FILE = "file"
    CUSTOM = "custom"


@dataclass
class BrokerConfig:
    """Broker configuration."""
    broker_type: BrokerType = BrokerType.SIMULATED
    name: str = "default"
    
    # Connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    
    # Trading settings
    account_id: Optional[str] = None
    paper_trading: bool = True
    commission_rate: float = 0.001  # 0.1%
    min_commission: float = 1.0
    
    # Connection settings
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 5.0
    
    # Additional settings
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum values serialized as strings."""
        data = asdict(self)
        if isinstance(data.get('broker_type'), BrokerType):
            data['broker_type'] = data['broker_type'].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrokerConfig':
        """Create from dictionary."""
        # Convert broker_type string to enum
        if 'broker_type' in data and isinstance(data['broker_type'], str):
            raw = data['broker_type']
            # Accept values like 'SIMULATED', 'simulated', or 'BrokerType.SIMULATED'
            if raw.startswith('BrokerType.'):
                raw = raw.split('.', 1)[1]
            # Normalize to enum value strings
            raw_norm = raw.lower()
            for bt in BrokerType:
                if bt.value == raw_norm or bt.name.lower() == raw_norm:
                    data['broker_type'] = bt
                    break
            else:
                data['broker_type'] = BrokerType.SIMULATED
        return cls(**data)


@dataclass
class DataFeedConfig:
    """Data feed configuration."""
    feed_type: DataFeedType = DataFeedType.SIMULATED
    name: str = "default"
    
    # Connection settings
    url: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    
    # Data settings
    symbols: List[str] = field(default_factory=list)
    update_frequency: float = 1.0  # seconds
    buffer_size: int = 1000
    
    # Quality settings
    enable_validation: bool = True
    max_staleness: float = 30.0  # seconds
    
    # Additional settings
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum values serialized as strings."""
        data = asdict(self)
        if isinstance(data.get('feed_type'), DataFeedType):
            data['feed_type'] = data['feed_type'].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataFeedConfig':
        """Create from dictionary."""
        # Convert feed_type string to enum
        if 'feed_type' in data and isinstance(data['feed_type'], str):
            raw = data['feed_type']
            if raw.startswith('DataFeedType.'):
                raw = raw.split('.', 1)[1]
            raw_norm = raw.lower()
            for ft in DataFeedType:
                if ft.value == raw_norm or ft.name.lower() == raw_norm:
                    data['feed_type'] = ft
                    break
            else:
                data['feed_type'] = DataFeedType.SIMULATED
        return cls(**data)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    log_to_file: bool = True
    log_file: str = "quantlib_live.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Console logging
    log_to_console: bool = True
    console_level: str = "INFO"
    
    # Component-specific levels
    component_levels: Dict[str, str] = field(default_factory=dict)
    
    # Trade logging
    log_trades: bool = True
    trade_log_file: str = "trades.log"
    
    # Order logging
    log_orders: bool = True
    order_log_file: str = "orders.log"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: float = 60.0  # seconds
    
    # Metrics collection
    enable_metrics: bool = True
    metrics_interval: float = 30.0  # seconds
    
    # Alerting
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log"])
    
    # Email alerts
    email_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_recipients: List[str] = field(default_factory=list)
    
    # Slack alerts
    slack_enabled: bool = False
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    
    # Dashboard
    enable_dashboard: bool = False
    dashboard_port: int = 8080
    dashboard_host: str = "localhost"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LiveTradingConfig:
    """Complete live trading configuration."""
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    config_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Core components
    engine: EngineConfig = field(default_factory=EngineConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    data_feed: DataFeedConfig = field(default_factory=DataFeedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Strategy settings
    strategy_name: str = "default"
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience properties expected by some tests
    @property
    def initial_capital(self) -> float:
        return self.engine.initial_capital

    @initial_capital.setter
    def initial_capital(self, value: float) -> None:
        self.engine.initial_capital = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        
        # Convert datetime to string
        if isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        
        # Convert enums to strings
        if isinstance(data['environment'], Environment):
            data['environment'] = data['environment'].value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiveTradingConfig':
        """Create from dictionary."""
        # Convert string to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Convert environment string to enum
        if 'environment' in data and isinstance(data['environment'], str):
            data['environment'] = Environment(data['environment'])
        
        # Convert nested configs
        if 'engine' in data and isinstance(data['engine'], dict):
            data['engine'] = EngineConfig(**data['engine'])
        
        if 'risk_limits' in data and isinstance(data['risk_limits'], dict):
            data['risk_limits'] = RiskLimits(**data['risk_limits'])
        
        if 'broker' in data and isinstance(data['broker'], dict):
            data['broker'] = BrokerConfig.from_dict(data['broker'])
        
        if 'data_feed' in data and isinstance(data['data_feed'], dict):
            data['data_feed'] = DataFeedConfig.from_dict(data['data_feed'])
        
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig.from_dict(data['logging'])
        
        if 'monitoring' in data and isinstance(data['monitoring'], dict):
            data['monitoring'] = MonitoringConfig.from_dict(data['monitoring'])
        
        return cls(**data)


class ConfigManager:
    """Configuration manager for live trading."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        # If a file path is provided, remember it; otherwise ensure directory exists
        self.config_file: Optional[Path] = None
        if self.config_dir.suffix.lower() == '.json':
            self.config_file = self.config_dir
            self.config_dir = self.config_dir.parent
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._configs: Dict[str, LiveTradingConfig] = {}
        
        # Environment variable overrides
        self._env_overrides = self._load_env_overrides()
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        # Common environment variables
        env_mappings = {
            'QUANTLIB_ENVIRONMENT': 'environment',
            'QUANTLIB_LOG_LEVEL': 'logging.level',
            'QUANTLIB_BROKER_TYPE': 'broker.broker_type',
            'QUANTLIB_BROKER_API_KEY': 'broker.api_key',
            'QUANTLIB_BROKER_SECRET_KEY': 'broker.secret_key',
            'QUANTLIB_DATA_FEED_URL': 'data_feed.url',
            'QUANTLIB_DATA_FEED_API_KEY': 'data_feed.api_key',
            'QUANTLIB_INITIAL_CAPITAL': 'engine.initial_capital',
            'QUANTLIB_MAX_POSITION_SIZE': 'risk_limits.max_position_size',
            'QUANTLIB_MAX_DAILY_LOSS': 'risk_limits.max_daily_loss'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                overrides[config_path] = value
        
        return overrides
    
    def create_default_config(
        self,
        name: str = "default",
        environment: Environment = Environment.DEVELOPMENT
    ) -> LiveTradingConfig:
        """Create default configuration."""
        config = LiveTradingConfig(
            environment=environment,
            engine=EngineConfig(
                symbols=["AAPL", "GOOGL", "MSFT"],
                initial_capital=100000.0,
                max_position_size=0.1,
                max_daily_loss=0.05,
                heartbeat_interval=1.0,
                enable_risk_checks=True,
                log_level="INFO"
            ),
            risk_limits=RiskLimits(
                max_position_size=0.1,
                max_daily_loss=0.05,
                max_weekly_loss=0.10,
                max_drawdown=0.20,
                max_var_95=0.03,
                max_order_size=50000.0
            ),
            broker=BrokerConfig(
                broker_type=BrokerType.SIMULATED,
                paper_trading=True,
                commission_rate=0.001
            ),
            data_feed=DataFeedConfig(
                feed_type=DataFeedType.SIMULATED,
                symbols=["AAPL", "GOOGL", "MSFT"],
                update_frequency=1.0
            ),
            logging=LoggingConfig(
                level="INFO",
                log_to_file=True,
                log_to_console=True
            ),
            monitoring=MonitoringConfig(
                enable_health_checks=True,
                enable_metrics=True,
                enable_alerts=True
            )
        )
        
        # Apply environment overrides
        config = self._apply_overrides(config)
        
        self._configs[name] = config
        return config
    
    def load_config(self, name: Optional[str] = None) -> Optional[LiveTradingConfig]:
        """Load configuration from file."""
        try:
            # If config_dir is a file path, use it directly
            if name is None:
                # If a concrete file path provided at init, load it; else default
                if self.config_file is not None:
                    config_file = self.config_file
                else:
                    config_file = self.config_dir / "default.json"
            else:
                config_file = self.config_dir / f"{name}.json"
            
            if not config_file.exists():
                self.logger.warning(f"Config file not found: {config_file}")
                return None
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            config = LiveTradingConfig.from_dict(data)
            
            # Apply environment overrides
            config = self._apply_overrides(config)
            
            self._configs[name] = config
            self.logger.info(f"Loaded config: {name}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load config {name}: {e}")
            return None
    
    def save_config(self, config: LiveTradingConfig, name: str) -> bool:
        """Save configuration to file."""
        try:
            # If a concrete file path provided at init, write to it; else name-based file
            if self.config_file is not None:
                config_file = self.config_file
            else:
                config_file = self.config_dir / f"{name}.json"
            
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
            
            self._configs[name] = config
            self.logger.info(f"Saved config: {name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config {name}: {e}")
            return False
    
    def get_config(self, name: str) -> Optional[LiveTradingConfig]:
        """Get configuration by name."""
        if name in self._configs:
            return self._configs[name]
        
        # Try to load from file
        return self.load_config(name)
    
    def list_configs(self) -> List[str]:
        """List available configurations."""
        configs = set(self._configs.keys())
        
        # Add configs from files
        for config_file in self.config_dir.glob("*.json"):
            configs.add(config_file.stem)
        
        return sorted(list(configs))
    
    def delete_config(self, name: str) -> bool:
        """Delete configuration."""
        try:
            # Remove from memory
            if name in self._configs:
                del self._configs[name]
            
            # Remove file
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
            
            self.logger.info(f"Deleted config: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete config {name}: {e}")
            return False
    
    def validate_config(self, config: LiveTradingConfig) -> bool:
        """Validate configuration and return True if valid, False otherwise."""
        issues = self.list_issues(config)
        return len(issues) == 0
    
    def list_issues(self, config: LiveTradingConfig) -> List[str]:
        """List all validation issues with the configuration."""
        issues = []
        
        try:
            # Validate engine config
            if not config.engine.symbols:
                issues.append("No symbols specified in engine config")
            
            if config.engine.initial_capital <= 0:
                issues.append("Initial capital must be positive")
            
            if config.engine.max_position_size <= 0 or config.engine.max_position_size > 1:
                issues.append("Max position size must be between 0 and 1")
            
            # Validate risk limits
            if config.risk_limits.max_daily_loss <= 0 or config.risk_limits.max_daily_loss > 1:
                issues.append("Max daily loss must be between 0 and 1")
            
            if config.risk_limits.max_order_size <= 0:
                issues.append("Max order size must be positive")
            
            # Validate broker config
            if config.broker.broker_type == BrokerType.CUSTOM and not config.broker.host:
                issues.append("Custom broker requires host configuration")
            
            # Validate data feed config
            if config.data_feed.feed_type == DataFeedType.WEBSOCKET and not config.data_feed.url:
                issues.append("WebSocket data feed requires URL configuration")
            
            # Validate consistency
            if set(config.engine.symbols) != set(config.data_feed.symbols):
                issues.append("Engine symbols and data feed symbols must match")
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return issues

    # Convenience boolean variant expected by tests
    def validate(self, config: LiveTradingConfig) -> bool:
        return len(self.validate_config(config)) == 0
    
    def _apply_overrides(self, config: LiveTradingConfig) -> LiveTradingConfig:
        """Apply environment variable overrides to configuration."""
        try:
            for config_path, value in self._env_overrides.items():
                self._set_nested_value(config, config_path, value)
        except Exception as e:
            self.logger.error(f"Error applying overrides: {e}")
        
        return config
    
    def _set_nested_value(self, obj: Any, path: str, value: str) -> None:
        """Set nested value using dot notation path."""
        parts = path.split('.')
        current = obj
        
        # Navigate to parent object
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Set final value with type conversion
        attr_name = parts[-1]
        current_value = getattr(current, attr_name)
        
        # Convert value to appropriate type
        if isinstance(current_value, bool):
            converted_value = value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(current_value, int):
            converted_value = int(value)
        elif isinstance(current_value, float):
            converted_value = float(value)
        elif isinstance(current_value, Enum):
            # Handle enum conversion
            enum_class = type(current_value)
            converted_value = enum_class(value)
        else:
            converted_value = value
        
        setattr(current, attr_name, converted_value)
    
    def create_production_config(
        self,
        name: str = "production",
        broker_type: BrokerType = BrokerType.INTERACTIVE_BROKERS,
        symbols: List[str] = None,
        initial_capital: float = 100000.0
    ) -> LiveTradingConfig:
        """Create production-ready configuration."""
        if symbols is None:
            symbols = ["SPY", "QQQ", "IWM"]
        
        config = LiveTradingConfig(
            environment=Environment.PRODUCTION,
            engine=EngineConfig(
                symbols=symbols,
                initial_capital=initial_capital,
                max_position_size=0.05,  # More conservative
                max_daily_loss=0.02,     # More conservative
                heartbeat_interval=0.5,   # Faster updates
                enable_risk_checks=True,
                auto_restart=True,
                log_level="WARNING"       # Less verbose
            ),
            risk_limits=RiskLimits(
                max_position_size=0.05,
                max_daily_loss=0.02,
                max_weekly_loss=0.05,
                max_drawdown=0.10,
                max_var_95=0.02,
                max_order_size=25000.0,   # Smaller orders
                max_orders_per_minute=5   # Rate limiting
            ),
            broker=BrokerConfig(
                broker_type=broker_type,
                paper_trading=False,      # Live trading
                commission_rate=0.0005,   # Lower commission
                timeout=10.0,             # Faster timeout
                retry_attempts=5          # More retries
            ),
            data_feed=DataFeedConfig(
                feed_type=DataFeedType.WEBSOCKET,
                symbols=symbols,
                update_frequency=0.1,     # High frequency
                enable_validation=True,
                max_staleness=5.0         # Strict staleness
            ),
            logging=LoggingConfig(
                level="WARNING",
                log_to_file=True,
                log_to_console=False,     # No console in production
                max_file_size=50 * 1024 * 1024,  # 50MB
                backup_count=10
            ),
            monitoring=MonitoringConfig(
                enable_health_checks=True,
                health_check_interval=30.0,
                enable_metrics=True,
                metrics_interval=10.0,
                enable_alerts=True,
                alert_channels=["email", "slack"],
                enable_dashboard=True
            )
        )
        
        # Apply environment overrides
        config = self._apply_overrides(config)
        
        self._configs[name] = config
        return config
    
    def get_config_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration summary."""
        config = self.get_config(name)
        if not config:
            return None
        
        return {
            'name': name,
            'environment': config.environment.value,
            'version': config.config_version,
            'created_at': config.created_at.isoformat(),
            'symbols': config.engine.symbols,
            'initial_capital': config.engine.initial_capital,
            'broker_type': config.broker.broker_type.value,
            'data_feed_type': config.data_feed.feed_type.value,
            'paper_trading': config.broker.paper_trading,
            'risk_checks_enabled': config.engine.enable_risk_checks,
            'validation_issues': len(self.validate_config(config))
        }