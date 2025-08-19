# QuantLib - Quantitative Analysis Library

A comprehensive Python library for quantitative finance analysis, providing tools for risk management, performance attribution, statistical analysis, portfolio optimization, and live trading capabilities.

## Features

### Core Infrastructure ✅
- **Base Components**: Extensible base classes for strategies, indicators, and portfolios
- **Data Management**: Market data handling with validation and preprocessing
- **Utilities**: Logging, configuration management, and data validation
- **Testing Framework**: Comprehensive test suite with pytest integration

### Analytics & Risk Management ✅
- **Risk Metrics**: VaR (Historical, Parametric, Monte Carlo), Expected Shortfall, Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown, drawdown periods, stress testing
- **Performance Analytics**: Comprehensive performance metrics and attribution analysis
- **Statistical Analysis**: Distribution fitting, hypothesis testing, Monte Carlo simulation, time series analysis
- **ARCH/GARCH Models**: Volatility clustering detection and modeling

### Portfolio Management ✅
- **Optimization**: Mean-variance optimization, Black-Litterman model, Hierarchical Risk Parity
- **Allocation**: Dynamic asset allocation and rebalancing strategies
- **Attribution**: Performance attribution analysis
- **Risk Monitoring**: Real-time portfolio risk assessment

### Technical Analysis ✅
- **Indicators**: Moving averages, RSI, MACD, Bollinger Bands, and more
- **Oscillators**: Momentum and mean-reversion indicators
- **Pattern Recognition**: Chart pattern detection
- **Signal Generation**: Technical trading signals

### Strategy Framework ✅
- **Base Strategy**: Extensible strategy framework
- **Execution**: Order execution and position management
- **Machine Learning**: ML-based strategy development
- **Risk Management**: Strategy-level risk controls

### Live Trading (Optional) ✅
- **Data Feeds**: Real-time and simulated market data feeds
- **Order Management**: Order routing and execution
- **Risk Monitoring**: Real-time risk assessment and alerts
- **Configuration**: Environment-specific configuration management
- **Logging & Alerts**: Structured logging and alert system
- **Dashboard**: Web-based monitoring interface

## Installation

### Basic Installation
```bash
# Clone the repository
git clone <repository-url>
cd quantlib

# Install in development mode
pip install -e .
```

### Optional Dependencies
```bash
# For portfolio optimization
pip install cvxpy

# For visualization
pip install matplotlib seaborn

# For live trading
pip install fastapi uvicorn websockets psycopg2-binary influxdb-client redis
```

## Quick Start

### Risk Analysis
```python
import numpy as np
from quantlib.analytics import VaRCalculator, PerformanceAnalyzer

# Generate sample returns
returns = np.random.normal(0.001, 0.02, 252)

# Calculate VaR using different methods
var_calc = VaRCalculator()
historical_var = var_calc.historical_var(returns, confidence_level=0.05)
parametric_var = var_calc.parametric_var(returns, confidence_level=0.05)
monte_carlo_var = var_calc.monte_carlo_var(returns, confidence_level=0.05)

# Performance metrics
perf_analyzer = PerformanceAnalyzer()
sharpe = perf_analyzer.sharpe_ratio(returns, risk_free_rate=0.02/252)
sortino = perf_analyzer.sortino_ratio(returns, risk_free_rate=0.02/252)
max_dd = perf_analyzer.max_drawdown(returns)
```

### Portfolio Optimization
```python
import numpy as np
from quantlib.portfolio import MeanVarianceOptimizer, HierarchicalRiskParity

# Sample data
returns = np.random.normal(0.001, 0.02, (252, 5))  # 5 assets, 252 days

# Mean-variance optimization
mvo = MeanVarianceOptimizer()
weights = mvo.optimize(returns, target_return=0.1)

# Hierarchical Risk Parity
hrp = HierarchicalRiskParity()
hrp_weights = hrp.optimize(returns)
```

### Technical Analysis
```python
import numpy as np
from quantlib.technical import SimpleMovingAverage, RelativeStrengthIndex, BollingerBands

# Sample price data
prices = np.random.normal(100, 2, 100).cumsum()

# Technical indicators
sma = SimpleMovingAverage(window=20)
sma_values = sma.calculate(prices)

rsi = RelativeStrengthIndex(window=14)
rsi_values = rsi.calculate(prices)

bb = BollingerBands(window=20, std_dev=2)
bb_upper, bb_middle, bb_lower = bb.calculate(prices)
```

## Project Structure

```
quantlib/
├── analytics/          # Risk and performance analytics
│   ├── performance.py  # Performance metrics and attribution
│   ├── risk.py        # Risk calculations (VaR, ES, etc.)
│   └── statistics.py  # Statistical analysis and tests
├── backtesting/       # Backtesting framework
│   ├── broker.py      # Simulated broker
│   ├── data_handler.py # Historical data management
│   ├── engine.py      # Backtesting engine
│   ├── events.py      # Event-driven architecture
│   ├── performance.py # Backtesting performance analysis
│   └── portfolio.py   # Portfolio management
├── core/              # Core infrastructure
│   ├── base.py        # Base classes
│   ├── data.py        # Data structures and validation
│   └── utils.py       # Utilities and helpers
├── live/              # Live trading components
│   ├── config.py      # Configuration management
│   ├── dashboard.py   # Web dashboard
│   ├── data_feeds.py  # Real-time data feeds
│   ├── engine.py      # Live trading engine
│   ├── logging_alerts.py # Logging and alerting
│   ├── order_manager.py # Order management
│   └── risk_monitor.py # Real-time risk monitoring
├── portfolio/         # Portfolio management
│   ├── allocation.py  # Asset allocation
│   ├── attribution.py # Performance attribution
│   ├── optimization.py # Portfolio optimization
│   └── rebalancing.py # Rebalancing strategies
├── strategy/          # Strategy framework
│   ├── base.py        # Base strategy classes
│   ├── execution.py   # Execution algorithms
│   ├── ml.py          # Machine learning strategies
│   ├── risk.py        # Strategy risk management
│   └── signals.py     # Signal generation
└── technical/         # Technical analysis
    ├── indicators.py  # Technical indicators
    ├── oscillators.py # Oscillators and momentum
    ├── patterns.py    # Pattern recognition
    └── signals.py     # Technical signals
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_analytics/
pytest tests/test_core/

# Run with coverage
pytest --cov=quantlib --cov-report=html
```

### Code Quality
```bash
# Install development dependencies
pip install -e .[dev]

# Run linting (if configured)
flake8 quantlib/

# Format code (if configured)
black quantlib/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License