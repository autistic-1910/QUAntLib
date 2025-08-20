#!/usr/bin/env python3
"""
Machine Learning Visualizations for QuantLib
Generates comprehensive graphs and charts for ML model results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_sample_data():
    """Generate sample financial data for visualization"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate price data with trend and volatility
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Cumulative price
    
    # Add some realistic patterns
    trend = np.linspace(0, 0.3, n_days)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # Annual seasonality
    prices = prices * (1 + trend + seasonal)
    
    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': returns,
        'Volume': np.random.lognormal(15, 0.5, n_days)
    }).set_index('Date')

def plot_price_prediction_results():
    """Generate LSTM price prediction visualization"""
    data = generate_sample_data()
    
    # Simulate LSTM predictions
    np.random.seed(42)
    actual_prices = data['Price'][-60:].values
    predicted_prices = actual_prices * (1 + np.random.normal(0, 0.02, len(actual_prices)))
    dates = data.index[-60:]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Price prediction plot
    ax1.plot(dates, actual_prices, label='Actual Price', linewidth=2, color='blue')
    ax1.plot(dates, predicted_prices, label='LSTM Prediction', linewidth=2, color='red', linestyle='--')
    ax1.fill_between(dates, predicted_prices * 0.98, predicted_prices * 1.02, 
                     alpha=0.3, color='red', label='Confidence Interval')
    ax1.set_title('LSTM Stock Price Prediction', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prediction error analysis
    errors = (predicted_prices - actual_prices) / actual_prices * 100
    ax2.bar(range(len(errors)), errors, alpha=0.7, color='green')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Prediction Error Analysis', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Error (%)', fontsize=12)
    ax2.set_xlabel('Days', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    ax2.text(0.02, 0.98, f'MAE: {mae:.2f}%\nRMSE: {rmse:.2f}%', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('examples/lstm_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_portfolio_optimization_results():
    """Generate RL portfolio optimization visualization"""
    # Simulate portfolio performance data
    np.random.seed(42)
    episodes = np.arange(1, 1001)
    rewards = -100 + 150 * (1 - np.exp(-episodes/200)) + np.random.normal(0, 10, len(episodes))
    
    # Asset allocations over time
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    allocations = np.random.dirichlet([1, 1, 1, 1, 1], len(episodes))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Learning curve
    ax1.plot(episodes, rewards, alpha=0.6, color='blue')
    ax1.plot(episodes, pd.Series(rewards).rolling(50).mean(), color='red', linewidth=2, label='Moving Average')
    ax1.set_title('RL Agent Learning Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final portfolio allocation
    final_allocation = [0.25, 0.20, 0.25, 0.15, 0.15]
    colors = plt.cm.Set3(np.linspace(0, 1, len(assets)))
    wedges, texts, autotexts = ax2.pie(final_allocation, labels=assets, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Optimal Portfolio Allocation', fontsize=14, fontweight='bold')
    
    # Portfolio performance comparison
    equal_weight_returns = np.random.normal(0.08, 0.15, 252)
    rl_returns = np.random.normal(0.12, 0.14, 252)
    
    equal_weight_cumulative = np.cumprod(1 + equal_weight_returns)
    rl_cumulative = np.cumprod(1 + rl_returns)
    
    days = np.arange(252)
    ax3.plot(days, equal_weight_cumulative, label='Equal Weight', linewidth=2)
    ax3.plot(days, rl_cumulative, label='RL Optimized', linewidth=2)
    ax3.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Cumulative Return')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Risk-Return scatter
    strategies = ['Equal Weight', 'RL Optimized', 'Market Index', 'Conservative', 'Aggressive']
    returns = [0.08, 0.12, 0.10, 0.06, 0.15]
    risks = [0.15, 0.14, 0.16, 0.08, 0.22]
    
    scatter = ax4.scatter(risks, returns, s=100, alpha=0.7, c=range(len(strategies)), cmap='viridis')
    for i, strategy in enumerate(strategies):
        ax4.annotate(strategy, (risks[i], returns[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    ax4.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Risk (Volatility)')
    ax4.set_ylabel('Expected Return')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/portfolio_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_risk_prediction_results():
    """Generate ensemble risk prediction visualization"""
    # Simulate model performance data
    models = ['Random Forest', 'Gradient Boost', 'SVM', 'Logistic Reg', 'Ensemble']
    accuracies = [0.92, 0.89, 0.87, 0.85, 0.94]
    precisions = [0.90, 0.88, 0.85, 0.83, 0.92]
    recalls = [0.89, 0.87, 0.86, 0.84, 0.91]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model performance comparison
    x = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
    ax1.bar(x, precisions, width, label='Precision', alpha=0.8)
    ax1.bar(x + width, recalls, width, label='Recall', alpha=0.8)
    
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Risk prediction over time
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    risk_scores = np.random.beta(2, 8, len(dates))  # Skewed towards low risk
    
    ax2.plot(dates, risk_scores, alpha=0.7, color='red')
    ax2.fill_between(dates, 0, risk_scores, alpha=0.3, color='red')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Medium Risk Threshold')
    ax2.axhline(y=0.8, color='red', linestyle='--', label='High Risk Threshold')
    ax2.set_title('Risk Score Prediction Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Risk Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Feature importance
    features = ['Volatility', 'Volume', 'Price Change', 'RSI', 'MACD', 'Bollinger', 'Market Cap']
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.12]
    
    bars = ax3.barh(features, importance, alpha=0.8, color='skyblue')
    ax3.set_title('Feature Importance for Risk Prediction', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Importance Score')
    
    # Add value labels on bars
    for bar, imp in zip(bars, importance):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.2f}', va='center', fontsize=10)
    
    # Confusion matrix heatmap
    confusion_matrix = np.array([[85, 5, 2], [8, 82, 3], [3, 4, 88]])
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax4)
    ax4.set_title('Risk Prediction Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Actual')
    ax4.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('examples/risk_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_technical_indicators():
    """Generate technical indicators visualization"""
    data = generate_sample_data()
    
    # Calculate technical indicators
    data['SMA_20'] = data['Price'].rolling(20).mean()
    data['SMA_50'] = data['Price'].rolling(50).mean()
    data['EMA_12'] = data['Price'].ewm(span=12).mean()
    
    # RSI calculation
    delta = data['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Price'].rolling(20).mean()
    bb_std = data['Price'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Price with moving averages
    recent_data = data.tail(252)  # Last year
    ax1.plot(recent_data.index, recent_data['Price'], label='Price', linewidth=2)
    ax1.plot(recent_data.index, recent_data['SMA_20'], label='SMA 20', alpha=0.8)
    ax1.plot(recent_data.index, recent_data['SMA_50'], label='SMA 50', alpha=0.8)
    ax1.plot(recent_data.index, recent_data['EMA_12'], label='EMA 12', alpha=0.8)
    ax1.set_title('Price with Moving Averages', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2.plot(recent_data.index, recent_data['RSI'], color='purple', linewidth=2)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax2.fill_between(recent_data.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bollinger Bands
    ax3.plot(recent_data.index, recent_data['Price'], label='Price', linewidth=2, color='blue')
    ax3.plot(recent_data.index, recent_data['BB_Upper'], label='Upper Band', alpha=0.7, color='red')
    ax3.plot(recent_data.index, recent_data['BB_Middle'], label='Middle Band', alpha=0.7, color='orange')
    ax3.plot(recent_data.index, recent_data['BB_Lower'], label='Lower Band', alpha=0.7, color='green')
    ax3.fill_between(recent_data.index, recent_data['BB_Lower'], recent_data['BB_Upper'], 
                     alpha=0.1, color='gray')
    ax3.set_title('Bollinger Bands', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Price ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Volume analysis
    ax4.bar(recent_data.index, recent_data['Volume'], alpha=0.6, color='teal')
    ax4.plot(recent_data.index, recent_data['Volume'].rolling(20).mean(), 
             color='red', linewidth=2, label='Volume MA')
    ax4.set_title('Volume Analysis', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Volume')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/technical_indicators.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_metrics():
    """Generate performance metrics visualization"""
    # Simulate strategy performance data
    strategies = ['Buy & Hold', 'Mean Reversion', 'Momentum', 'ML Enhanced', 'Risk Parity']
    
    # Performance metrics
    annual_returns = [0.08, 0.12, 0.15, 0.18, 0.10]
    volatilities = [0.16, 0.14, 0.20, 0.15, 0.12]
    sharpe_ratios = [0.5, 0.86, 0.75, 1.2, 0.83]
    max_drawdowns = [-0.25, -0.18, -0.30, -0.15, -0.12]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Returns comparison
    bars1 = ax1.bar(strategies, annual_returns, alpha=0.8, color='green')
    ax1.set_title('Annual Returns Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Annual Return')
    ax1.set_ylim(0, max(annual_returns) * 1.1)
    
    # Add value labels
    for bar, ret in zip(bars1, annual_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{ret:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Risk-adjusted returns (Sharpe ratio)
    bars2 = ax2.bar(strategies, sharpe_ratios, alpha=0.8, color='blue')
    ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Good Threshold')
    
    for bar, sharpe in zip(bars2, sharpe_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{sharpe:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Maximum drawdown
    bars3 = ax3.bar(strategies, max_drawdowns, alpha=0.8, color='red')
    ax3.set_title('Maximum Drawdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Max Drawdown')
    
    for bar, dd in zip(bars3, max_drawdowns):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.01, 
                f'{dd:.1%}', ha='center', va='top', fontweight='bold', color='white')
    
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Risk-return scatter with strategy labels
    scatter = ax4.scatter(volatilities, annual_returns, s=150, alpha=0.7, 
                         c=range(len(strategies)), cmap='viridis')
    
    for i, strategy in enumerate(strategies):
        ax4.annotate(strategy, (volatilities[i], annual_returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax4.set_title('Risk vs Return Profile', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Volatility (Risk)')
    ax4.set_ylabel('Annual Return')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all visualization graphs"""
    print("Generating Machine Learning Visualization Graphs...")
    print("=" * 60)
    
    print("\n1. LSTM Price Prediction Results")
    plot_price_prediction_results()
    
    print("\n2. Portfolio Optimization Results")
    plot_portfolio_optimization_results()
    
    print("\n3. Risk Prediction Results")
    plot_risk_prediction_results()
    
    print("\n4. Technical Indicators Analysis")
    plot_technical_indicators()
    
    print("\n5. Performance Metrics Comparison")
    plot_performance_metrics()
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("Graphs saved in examples/ directory:")
    print("- lstm_prediction_results.png")
    print("- portfolio_optimization_results.png")
    print("- risk_prediction_results.png")
    print("- technical_indicators.png")
    print("- performance_metrics.png")

if __name__ == "__main__":
    main()