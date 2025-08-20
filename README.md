# QuantLib - Quantitative Analysis Library

A Python library for quantitative finance analysis, providing foundational tools for risk management, performance attribution, statistical analysis, and portfolio optimization. This is an educational and research-oriented project with significant limitations for production use.

## Features

### Core Infrastructure ✅
- **Base Components**: Extensible base classes for strategies, indicators, and portfolios
- **Data Management**: Market data handling with validation and preprocessing
- **Utilities**: Logging, configuration management, and data validation
- **Testing Framework**: Basic test structure with pytest integration

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
- **Machine Learning**: LSTM price prediction, RL portfolio optimization, ensemble risk models
- **Risk Management**: Strategy-level risk controls

### Live Trading (Experimental) ⚠️
- **Data Feeds**: Basic structure for real-time data (not production-ready)
- **Order Management**: Prototype order routing (lacks broker integration)
- **Risk Monitoring**: Basic risk assessment framework
- **Configuration**: Simple configuration management
- **Logging & Alerts**: Basic logging structure
- **Dashboard**: Minimal web interface (not functional)

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

# For machine learning models
pip install -r requirements_ml.txt

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

### Machine Learning Models
```python
import numpy as np
import pandas as pd
from examples.ml_training_models import (
    LSTMPricePredictionModel,
    PortfolioOptimizationRL,
    EnsembleRiskModel
)

# Generate sample data
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
prices = pd.DataFrame({
    'AAPL': 150 * np.cumprod(1 + np.random.normal(0.0008, 0.02, len(dates))),
    'GOOGL': 2500 * np.cumprod(1 + np.random.normal(0.0006, 0.025, len(dates))),
    'MSFT': 300 * np.cumprod(1 + np.random.normal(0.0007, 0.022, len(dates)))
}, index=dates)

# LSTM Price Prediction
lstm_model = LSTMPricePredictionModel(sequence_length=60)
train_data = prices[['AAPL']].rename(columns={'AAPL': 'close'})
train_data['volume'] = np.random.lognormal(15, 0.5, len(train_data))
train_data['high'] = train_data['close'] * 1.02
train_data['low'] = train_data['close'] * 0.98

training_results = lstm_model.train(train_data.iloc[:-100])
predictions = lstm_model.predict(train_data.iloc[-160:-100], steps_ahead=5)

# Reinforcement Learning Portfolio Optimization
rl_optimizer = PortfolioOptimizationRL(['AAPL', 'GOOGL', 'MSFT'])
rl_results = rl_optimizer.train(prices, episodes=500)
optimal_allocation = rl_optimizer.get_allocation(prices.tail(50))

# Ensemble Risk Prediction
risk_model = EnsembleRiskModel()
risk_results = risk_model.train(prices.iloc[:-100])
risk_prediction = risk_model.predict_risk(prices.tail(50))

print(f"Risk Level: {risk_prediction['risk_level']}")
print(f"Risk Probability: {risk_prediction['risk_probability']:.1%}")
print(f"Model Uncertainty: {risk_prediction['uncertainty']:.3f}")
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
├── examples/          # Example implementations and demonstrations
│   ├── ml_training_models.py      # ML model implementations
│   └── ml_models_demonstration.ipynb  # Interactive ML demo notebook
└── technical/         # Technical analysis
    ├── indicators.py  # Technical indicators
    ├── oscillators.py # Oscillators and momentum
    ├── patterns.py    # Pattern recognition
    └── signals.py     # Technical signals
```

## Status

### Currently Implemented and Tested ✅
- **Core Analytics**: VaR calculations (Historical, Parametric with normal/t-distribution, Monte Carlo)
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, Maximum drawdown
- **Backtesting Framework**: Event-driven architecture with simulated broker
- **Order Execution**: Commission and slippage modeling with partial fill support
- **Data Structures**: Market data handling and validation
- **Unit Tests**: Deterministic tests for VaR, ratios, drawdown, and broker execution

### Partially Implemented ⚠️
- **Portfolio Optimization**: Framework exists but needs comprehensive testing
- **Technical Analysis**: Basic indicators implemented but limited testing
- **Live Trading**: Infrastructure present but requires integration testing
- **Statistical Analysis**: Basic functionality with room for expansion

### Recently Added ✅
- **Machine Learning Models**: LSTM price prediction, RL portfolio optimization, ensemble risk models
- **ML Examples**: Demonstration examples with synthetic data
- **Interactive Demos**: Jupyter notebook with ML model examples
- **Visualization Tools**: Graph generation for model results

### Not Yet Implemented ❌
- **ARCH/GARCH Models**: Planned but not implemented
- **Pattern Recognition**: Placeholder implementation only
- **Real-time Dashboard**: Non-functional prototype
- **Model Deployment**: No production deployment capabilities
- **Broker Integration**: No real broker API connections
- **Data Validation**: Limited input validation and error handling
- **Performance Optimization**: No optimization for large datasets
- **Production Monitoring**: No health checks or system monitoring

## Roadmap

### Phase 1: Core Stability (Current)
- ✅ Fix VaR calculation bugs and mathematical correctness
- ✅ Add comprehensive unit tests for analytics and backtesting
- ✅ Improve repository hygiene (CI, dependencies, documentation)
- 🔄 Add type hints throughout codebase
- 🔄 Standardize period handling across all ratio calculations

### Phase 2: Portfolio & Optimization
- 📋 Complete portfolio optimization testing and validation
- 📋 Implement robust rebalancing strategies
- 📋 Add performance attribution analysis
- 📋 Enhance risk monitoring capabilities

### Phase 3: Advanced Analytics
- 📋 Implement ARCH/GARCH volatility models
- 📋 Add advanced statistical tests and distribution fitting
- 📋 Develop machine learning strategy framework
- 📋 Create pattern recognition algorithms

### Phase 4: Production Ready
- 📋 Complete live trading integration testing
- 📋 Build real-time monitoring dashboard
- 📋 Add comprehensive logging and alerting
- 📋 Performance optimization and scalability

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_analytics/
pytest tests/test_backtesting/

# Run with coverage
pytest --cov=quantlib --cov-report=html
```

### Code Quality
```bash
# Install development dependencies
pip install -r requirements-lock.txt

# Run linting
flake8 quantlib/

# Format code
black quantlib/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Known Limitations and Issues

### Technical Constraints
- **Data Scale**: Not optimized for large datasets (>100k records may cause performance issues)
- **Memory Usage**: No memory management for long-running processes
- **Concurrency**: Not thread-safe, no support for parallel processing
- **Error Handling**: Limited exception handling and recovery mechanisms
- **Input Validation**: Minimal data validation, prone to errors with malformed inputs

### Mathematical and Statistical Limitations
- **VaR Models**: Limited to basic implementations, no advanced copula models
- **Backtesting**: No walk-forward analysis or out-of-sample testing framework
- **Risk Models**: Assumes normal distributions in many calculations
- **Portfolio Optimization**: No transaction cost modeling in optimization
- **Statistical Tests**: Limited statistical robustness testing

### Machine Learning Constraints
- **Model Validation**: No proper cross-validation or hyperparameter tuning
- **Data Leakage**: Potential look-ahead bias in feature engineering
- **Overfitting**: No regularization or model selection frameworks
- **Real-time Inference**: Models not optimized for low-latency prediction
- **Model Monitoring**: No drift detection or model performance tracking

### Live Trading Limitations
- **Broker Integration**: No actual broker API implementations
- **Order Management**: Simulated execution only, no real order routing
- **Risk Controls**: Basic risk checks, no sophisticated pre-trade risk management
- **Latency**: Not designed for high-frequency trading
- **Reliability**: No failover mechanisms or system redundancy
- **Compliance**: No regulatory compliance features

### Testing and Quality Assurance
- **Test Coverage**: Incomplete test coverage (~60-70%)
- **Integration Tests**: Limited integration testing between components
- **Performance Tests**: No performance benchmarking or stress testing
- **Edge Cases**: Many edge cases not covered in testing
- **Data Quality**: Tests use synthetic data, limited real-world validation

### Documentation and Usability
- **API Documentation**: Incomplete docstrings and API documentation
- **Examples**: Limited real-world examples and use cases
- **Error Messages**: Often unclear error messages and debugging information
- **Configuration**: Complex configuration with limited validation

### Security Considerations
- **Data Security**: No encryption for sensitive financial data
- **Authentication**: No user authentication or authorization
- **API Security**: No rate limiting or input sanitization
- **Logging**: Potential sensitive data exposure in logs

### Performance and Scalability
- **Single-threaded**: No multi-threading or async processing
- **Memory Leaks**: Potential memory leaks in long-running processes
- **Database**: No persistent storage or database integration
- **Caching**: No caching mechanisms for expensive calculations

### Maintenance and Support
- **Dependencies**: May have outdated or conflicting dependencies
- **Versioning**: No semantic versioning or backward compatibility guarantees
- **Support**: No official support or maintenance commitments
- **Updates**: Irregular updates and potential breaking changes

## Recommended Use Cases

### Suitable For:
- Educational purposes and learning quantitative finance concepts
- Research and prototyping of trading strategies
- Academic projects and coursework
- Small-scale backtesting with synthetic or clean historical data
- Proof-of-concept development

### Not Suitable For:
- Production trading systems
- Managing real money or client funds
- High-frequency trading applications
- Large-scale institutional use
- Regulatory compliance requirements
- Mission-critical financial applications

## Disclaimer

**Important**: This software is provided for educational and research purposes only. It has not been tested for production use and contains known limitations that make it unsuitable for real trading or financial decision-making. Users should not rely on this software for actual investment decisions or risk management. The authors assume no responsibility for any financial losses or damages resulting from the use of this software.

## License

MIT License