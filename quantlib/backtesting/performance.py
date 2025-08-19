"""Performance analysis for backtesting results.

Provides comprehensive analysis of backtest performance and statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from .portfolio import BacktestPortfolio
from .data_handler import DataHandler
from .broker import ExecutionHandler
from quantlib.analytics.performance import PerformanceAnalyzer
from quantlib.analytics.risk import RiskMetrics
from quantlib.core.utils import Logger


class BacktestResults:
    """Comprehensive backtest results and analysis."""
    
    def __init__(self, portfolio: BacktestPortfolio, data_handler: DataHandler,
                 execution_handler: ExecutionHandler, start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None):
        """
        Initialize backtest results.
        
        Args:
            portfolio: Backtest portfolio with performance data
            data_handler: Data handler used in backtest
            execution_handler: Execution handler used in backtest
            start_time: Backtest start time
            end_time: Backtest end time
        """
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.start_time = start_time
        self.end_time = end_time
        self.logger = Logger.get_logger("backtest_results")
        
        # Initialize analyzers
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_metrics = RiskMetrics()
        
        # Cache for computed metrics
        self._summary_stats = None
        self._trade_analysis = None
        
    def get_equity_curve(self) -> pd.DataFrame:
        """Get portfolio equity curve."""
        return self.portfolio.get_equity_curve_df()
        
    def get_returns(self) -> pd.Series:
        """Get portfolio returns series."""
        return self.portfolio.get_returns_series()
        
    def get_trades(self) -> pd.DataFrame:
        """Get trade history."""
        return self.portfolio.get_trades_df()
        
    def get_positions(self) -> Dict[str, Any]:
        """Get final positions."""
        positions = {}
        for symbol, position in self.portfolio.positions.items():
            if position.quantity != 0:
                positions[symbol] = {
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'realized_pnl': position.realized_pnl,
                    'unrealized_pnl': position.get_unrealized_pnl(
                        self.portfolio.current_prices.get(symbol, position.avg_price)
                    )
                }
        return positions
        
    def get_summary_stats(self) -> Dict[str, float]:
        """Get comprehensive summary statistics."""
        if self._summary_stats is not None:
            return self._summary_stats
            
        returns = self.get_returns()
        equity_curve = self.get_equity_curve()
        
        if returns.empty or equity_curve.empty:
            self.logger.warning("No returns data available for analysis")
            return {}
            
        # Basic portfolio metrics
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        # Performance metrics
        performance_metrics = self.performance_analyzer.calculate_performance_metrics(returns)
        
        # Trade analysis
        trade_stats = self._analyze_trades()
        
        # Execution statistics
        execution_stats = self.execution_handler.get_execution_statistics()
        
        # Combine all metrics
        summary = {
            **portfolio_summary,
            **performance_metrics,
            **trade_stats,
            **execution_stats
        }
        
        # Add backtest metadata
        if self.start_time and self.end_time:
            summary['backtest_duration'] = (self.end_time - self.start_time).total_seconds()
            
        data_summary = self.data_handler.get_data_summary()
        summary['data_start'] = data_summary.get('start_date')
        summary['data_end'] = data_summary.get('end_date')
        summary['total_periods'] = data_summary.get('total_timestamps', 0)
        
        self._summary_stats = summary
        return summary
        
    def _analyze_trades(self) -> Dict[str, float]:
        """Analyze individual trades."""
        trades_df = self.get_trades()
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
            
        # Group trades by symbol to calculate P&L
        trade_pnl = []
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
            symbol_trades = symbol_trades.sort_index()
            
            position = 0
            avg_price = 0.0
            
            for _, trade in symbol_trades.iterrows():
                if trade['direction'] == 'BUY':
                    if position <= 0:
                        # Opening long or closing short
                        if position < 0:
                            # Closing short position
                            pnl = -position * (avg_price - trade['price'])
                            trade_pnl.append(pnl)
                            
                        # Update position
                        new_position = position + trade['quantity']
                        if new_position > 0:
                            avg_price = trade['price']
                        position = new_position
                    else:
                        # Adding to long position
                        total_cost = position * avg_price + trade['quantity'] * trade['price']
                        position += trade['quantity']
                        avg_price = total_cost / position
                        
                else:  # SELL
                    if position >= 0:
                        # Closing long or opening short
                        if position > 0:
                            # Closing long position
                            pnl = min(position, trade['quantity']) * (trade['price'] - avg_price)
                            trade_pnl.append(pnl)
                            
                        # Update position
                        new_position = position - trade['quantity']
                        if new_position < 0:
                            avg_price = trade['price']
                        position = new_position
                    else:
                        # Adding to short position
                        total_cost = -position * avg_price + trade['quantity'] * trade['price']
                        position -= trade['quantity']
                        avg_price = total_cost / abs(position) if position != 0 else 0
                        
        # Calculate trade statistics
        if not trade_pnl:
            return {
                'total_trades': len(trades_df),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
            
        trade_pnl = np.array(trade_pnl)
        winning_trades = trade_pnl[trade_pnl > 0]
        losing_trades = trade_pnl[trade_pnl < 0]
        
        total_trades = len(trade_pnl)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        largest_win = winning_trades.max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
    def get_monthly_returns(self) -> pd.DataFrame:
        """Get monthly returns breakdown."""
        returns = self.get_returns()
        
        if returns.empty:
            return pd.DataFrame()
            
        # Resample to monthly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table by year and month
        monthly_df = monthly_returns.to_frame('returns')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot_table = monthly_df.pivot_table(
            values='returns', index='year', columns='month', fill_value=0
        )
        
        # Add month names as columns
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        # Add annual returns
        annual_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        pivot_table['Annual'] = annual_returns.values
        
        return pivot_table
        
    def get_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """Get rolling performance metrics."""
        returns = self.get_returns()
        
        if returns.empty or len(returns) < window:
            return pd.DataFrame()
            
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        rolling_metrics['rolling_sharpe'] = self.risk_metrics.rolling_sharpe_ratio(
            returns, window
        )
        
        # Rolling max drawdown
        equity_curve = self.get_equity_curve()
        if not equity_curve.empty:
            rolling_metrics['rolling_max_drawdown'] = equity_curve['portfolio_value'].rolling(
                window
            ).apply(lambda x: self.risk_metrics._calculate_max_drawdown_series(pd.Series(x))['max_drawdown'])
            
        return rolling_metrics.dropna()
        
    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        summary = self.get_summary_stats()
        
        report = []
        report.append("=" * 60)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Portfolio Summary
        report.append("\nPORTFOLIO SUMMARY:")
        report.append("-" * 20)
        report.append(f"Initial Capital: ${summary.get('initial_capital', 0):,.2f}")
        report.append(f"Final Value: ${summary.get('current_value', 0):,.2f}")
        report.append(f"Total Return: {summary.get('total_return', 0):.2%}")
        report.append(f"Total P&L: ${summary.get('total_pnl', 0):,.2f}")
        
        # Performance Metrics
        report.append("\nPERFORMANCE METRICS:")
        report.append("-" * 20)
        report.append(f"Annualized Return: {summary.get('annualized_return', 0):.2%}")
        report.append(f"Volatility: {summary.get('volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {summary.get('sortino_ratio', 0):.2f}")
        report.append(f"Calmar Ratio: {summary.get('calmar_ratio', 0):.2f}")
        
        # Risk Metrics
        report.append("\nRISK METRICS:")
        report.append("-" * 20)
        report.append(f"Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
        report.append(f"VaR (95%): {summary.get('var_95', 0):.2%}")
        report.append(f"Expected Shortfall (95%): {summary.get('expected_shortfall_95', 0):.2%}")
        report.append(f"Skewness: {summary.get('skewness', 0):.2f}")
        report.append(f"Kurtosis: {summary.get('kurtosis', 0):.2f}")
        
        # Trade Analysis
        report.append("\nTRADE ANALYSIS:")
        report.append("-" * 20)
        report.append(f"Total Trades: {summary.get('total_trades', 0)}")
        report.append(f"Winning Trades: {summary.get('winning_trades', 0)}")
        report.append(f"Losing Trades: {summary.get('losing_trades', 0)}")
        report.append(f"Win Rate: {summary.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
        report.append(f"Average Win: ${summary.get('avg_win', 0):.2f}")
        report.append(f"Average Loss: ${summary.get('avg_loss', 0):.2f}")
        
        # Execution Statistics
        report.append("\nEXECUTION STATISTICS:")
        report.append("-" * 20)
        report.append(f"Total Commission: ${summary.get('total_commission', 0):.2f}")
        report.append(f"Average Slippage: {summary.get('avg_slippage', 0):.4%}")
        
        # Backtest Metadata
        if summary.get('data_start') and summary.get('data_end'):
            report.append("\nBACKTEST PERIOD:")
            report.append("-" * 20)
            report.append(f"Start Date: {summary['data_start']}")
            report.append(f"End Date: {summary['data_end']}")
            report.append(f"Total Periods: {summary.get('total_periods', 0)}")
            
        if summary.get('backtest_duration'):
            duration = summary['backtest_duration']
            report.append(f"Execution Time: {duration:.2f} seconds")
            
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
        
    def save_results(self, filepath: str, format: str = 'csv'):
        """Save backtest results to file."""
        if format.lower() == 'csv':
            # Save equity curve
            equity_curve = self.get_equity_curve()
            if not equity_curve.empty:
                equity_curve.to_csv(f"{filepath}_equity_curve.csv")
                
            # Save trades
            trades = self.get_trades()
            if not trades.empty:
                trades.to_csv(f"{filepath}_trades.csv")
                
            # Save summary stats
            summary = self.get_summary_stats()
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(f"{filepath}_summary.csv", index=False)
            
        elif format.lower() == 'excel':
            with pd.ExcelWriter(f"{filepath}.xlsx") as writer:
                # Equity curve
                equity_curve = self.get_equity_curve()
                if not equity_curve.empty:
                    equity_curve.to_excel(writer, sheet_name='Equity Curve')
                    
                # Trades
                trades = self.get_trades()
                if not trades.empty:
                    trades.to_excel(writer, sheet_name='Trades')
                    
                # Summary
                summary = self.get_summary_stats()
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Monthly returns
                monthly_returns = self.get_monthly_returns()
                if not monthly_returns.empty:
                    monthly_returns.to_excel(writer, sheet_name='Monthly Returns')
                    
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        self.logger.info(f"Results saved to {filepath}")