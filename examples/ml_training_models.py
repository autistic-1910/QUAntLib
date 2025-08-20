#!/usr/bin/env python3
"""
Machine Learning Training Models Examples

Demonstrates various ML models for quantitative finance:
- LSTM/GRU for price prediction
- Reinforcement Learning for portfolio optimization
- Ensemble models for risk prediction
- Model evaluation and backtesting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core quantlib imports (optional - will use simplified versions if not available)
try:
    from quantlib.core.data import MarketData
    from quantlib.analytics.performance import PerformanceAnalyzer
    from quantlib.analytics.risk import RiskAnalyzer
    from quantlib.strategy.ml import FeatureEngineer, RandomForestModel, MLSignalGenerator
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("QuantLib not installed - using simplified feature engineering")
    
    class FeatureEngineer:
        """Simplified feature engineer for demonstration"""
        def create_technical_features(self, data):
            features = pd.DataFrame(index=data.index)
            for col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    # Simple technical features
                    features[f'{col}_sma_5'] = data[col].rolling(5).mean()
                    features[f'{col}_sma_20'] = data[col].rolling(20).mean()
                    features[f'{col}_rsi'] = self._calculate_rsi(data[col])
            return features.fillna(0)
        
        def _calculate_rsi(self, prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

# Additional ML dependencies (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LSTMPricePredictionModel:
    """
    LSTM model for stock price prediction
    
    Demonstrates:
    - Time series forecasting with LSTM
    - Feature engineering for sequential data
    - Model evaluation and backtesting
    """
    
    def __init__(self, sequence_length: int = 60, features: List[str] = None):
        self.sequence_length = sequence_length
        self.features = features or ['close', 'volume', 'high', 'low']
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for data preprocessing")
            
        from sklearn.preprocessing import MinMaxScaler
        
        # Select and scale features
        feature_data = data[self.features].values
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict close price
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM model")
            
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """Train the LSTM model"""
        X, y = self.prepare_data(data)
        
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=validation_split,
            verbose=0
        )
        
        self.is_trained = True
        
        # Return training metrics
        return {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_mae': history.history['mae'][-1],
            'final_val_mae': history.history['val_mae'][-1]
        }
    
    def predict(self, data: pd.DataFrame, steps_ahead: int = 1) -> np.ndarray:
        """Make price predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Prepare last sequence
        feature_data = data[self.features].tail(self.sequence_length).values
        scaled_data = self.scaler.transform(feature_data)
        
        predictions = []
        current_sequence = scaled_data.reshape(1, self.sequence_length, len(self.features))
        
        for _ in range(steps_ahead):
            pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            new_row = np.zeros((1, 1, len(self.features)))
            new_row[0, 0, 0] = pred  # Use prediction as next close price
            current_sequence = np.concatenate([
                current_sequence[:, 1:, :], new_row
            ], axis=1)
        
        # Inverse transform predictions
        pred_array = np.array(predictions).reshape(-1, 1)
        dummy_features = np.zeros((len(predictions), len(self.features)))
        dummy_features[:, 0] = pred_array.flatten()
        
        return self.scaler.inverse_transform(dummy_features)[:, 0]


class PortfolioOptimizationRL:
    """
    Reinforcement Learning for Portfolio Optimization
    
    Demonstrates:
    - Q-learning for asset allocation
    - Risk-adjusted reward functions
    - Dynamic rebalancing strategies
    """
    
    def __init__(self, assets: List[str], lookback_window: int = 20):
        self.assets = assets
        self.lookback_window = lookback_window
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
    def get_state(self, data: pd.DataFrame) -> str:
        """Convert market data to state representation"""
        if len(data) < self.lookback_window:
            return "insufficient_data"
            
        recent_data = data.tail(self.lookback_window)
        
        # Calculate state features
        features = []
        for asset in self.assets:
            if asset in recent_data.columns:
                returns = recent_data[asset].pct_change().dropna()
                volatility = returns.std()
                momentum = returns.mean()
                
                # Discretize features
                vol_state = "high" if volatility > 0.02 else "low"
                mom_state = "up" if momentum > 0 else "down"
                features.append(f"{asset}_{vol_state}_{mom_state}")
        
        return "_".join(features)
    
    def get_actions(self) -> List[Tuple[float, ...]]:
        """Define possible portfolio allocations"""
        # Simple allocation strategies
        n_assets = len(self.assets)
        actions = []
        
        # Equal weight
        actions.append(tuple([1.0/n_assets] * n_assets))
        
        # Concentrated positions
        for i in range(n_assets):
            allocation = [0.1] * n_assets
            allocation[i] = 0.7
            actions.append(tuple(allocation))
        
        # Conservative (cash heavy)
        conservative = [0.2] * n_assets
        actions.append(tuple(conservative))
        
        return actions
    
    def calculate_reward(self, returns: pd.Series, allocation: Tuple[float, ...]) -> float:
        """Calculate risk-adjusted reward"""
        portfolio_return = sum(ret * weight for ret, weight in zip(returns, allocation))
        portfolio_vol = returns.std() * np.sqrt(sum(w**2 for w in allocation))
        
        # Sharpe-like reward with penalty for high volatility
        if portfolio_vol > 0:
            sharpe = portfolio_return / portfolio_vol
            return sharpe - 0.1 * portfolio_vol  # Volatility penalty
        return portfolio_return
    
    def train(self, data: pd.DataFrame, episodes: int = 1000) -> Dict:
        """Train the RL agent"""
        actions = self.get_actions()
        rewards_history = []
        
        for episode in range(episodes):
            # Random starting point
            start_idx = np.random.randint(self.lookback_window, len(data) - 20)
            episode_data = data.iloc[start_idx:start_idx + 20]
            
            total_reward = 0
            
            for i in range(len(episode_data) - 1):
                current_data = episode_data.iloc[:i+1]
                state = self.get_state(current_data)
                
                # Choose action (epsilon-greedy)
                if np.random.random() < self.epsilon or state not in self.q_table:
                    action_idx = np.random.randint(len(actions))
                else:
                    action_idx = np.argmax(self.q_table[state])
                
                action = actions[action_idx]
                
                # Calculate reward
                next_returns = episode_data.iloc[i+1][self.assets].pct_change()
                reward = self.calculate_reward(next_returns, action)
                total_reward += reward
                
                # Update Q-table
                if state not in self.q_table:
                    self.q_table[state] = [0.0] * len(actions)
                
                next_state = self.get_state(episode_data.iloc[:i+2])
                next_q_value = 0
                if next_state in self.q_table:
                    next_q_value = max(self.q_table[next_state])
                
                # Q-learning update
                self.q_table[state][action_idx] += self.learning_rate * (
                    reward + self.discount_factor * next_q_value - self.q_table[state][action_idx]
                )
            
            rewards_history.append(total_reward)
            
            # Decay exploration
            if episode % 100 == 0:
                self.epsilon *= 0.95
        
        return {
            'final_reward': rewards_history[-1],
            'average_reward': np.mean(rewards_history[-100:]),
            'total_states': len(self.q_table)
        }
    
    def get_allocation(self, data: pd.DataFrame) -> Tuple[float, ...]:
        """Get optimal allocation for current state"""
        state = self.get_state(data)
        actions = self.get_actions()
        
        if state not in self.q_table:
            # Default to equal weight
            return tuple([1.0/len(self.assets)] * len(self.assets))
        
        best_action_idx = np.argmax(self.q_table[state])
        return actions[best_action_idx]


class EnsembleRiskModel:
    """
    Ensemble model for risk prediction
    
    Demonstrates:
    - Multiple model combination
    - VaR and drawdown prediction
    - Model uncertainty quantification
    """
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
    def build_ensemble(self) -> None:
        """Build ensemble of different models"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for ensemble models")
            
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        # Individual models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'
        )
    
    def prepare_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for risk prediction"""
        # Technical features
        tech_features = self.feature_engineer.create_technical_features(data)
        
        # Risk-specific features
        risk_features = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                returns = data[col].pct_change()
                
                # Volatility clustering
                risk_features[f'{col}_vol_cluster'] = returns.rolling(20).std()
                
                # Extreme movements
                risk_features[f'{col}_extreme_up'] = (returns > returns.quantile(0.95)).astype(int)
                risk_features[f'{col}_extreme_down'] = (returns < returns.quantile(0.05)).astype(int)
                
                # Drawdown features
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                risk_features[f'{col}_drawdown'] = drawdown
                
                # VaR approximation
                risk_features[f'{col}_var_95'] = returns.rolling(20).quantile(0.05)
        
        # Combine features
        all_features = pd.concat([tech_features, risk_features], axis=1)
        return all_features.fillna(0)
    
    def create_risk_labels(self, data: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Create risk labels (high risk = 1, low risk = 0)"""
        # Calculate portfolio volatility (assuming equal weights)
        returns = data.pct_change().dropna()
        portfolio_returns = returns.mean(axis=1)
        
        # Rolling volatility
        rolling_vol = portfolio_returns.rolling(20).std()
        
        # High risk periods
        risk_labels = (rolling_vol > threshold).astype(int)
        return risk_labels
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the ensemble model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for ensemble training")
            
        # Prepare features and labels
        features = self.prepare_risk_features(data)
        labels = self.create_risk_labels(data)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Remove rows with insufficient data
        valid_rows = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_rows]
        y = y[valid_rows]
        
        if len(X) < 50:
            raise ValueError("Insufficient data for training")
        
        # Build and train ensemble
        self.build_ensemble()
        
        # Train individual models
        results = {}
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                score = model.score(X, y)
                results[f'{name}_accuracy'] = score
            except Exception as e:
                print(f"Failed to train {name}: {e}")
        
        # Train ensemble
        self.ensemble.fit(X, y)
        results['ensemble_accuracy'] = self.ensemble.score(X, y)
        
        self.is_trained = True
        return results
    
    def predict_risk(self, data: pd.DataFrame) -> Dict:
        """Predict risk level and uncertainty"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        features = self.prepare_risk_features(data)
        latest_features = features.iloc[[-1]].fillna(0)
        
        # Individual model predictions
        individual_preds = {}
        for name, model in self.models.items():
            try:
                pred = model.predict_proba(latest_features)[0, 1]
                individual_preds[name] = pred
            except:
                individual_preds[name] = 0.5
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict_proba(latest_features)[0, 1]
        
        # Calculate uncertainty (disagreement between models)
        pred_values = list(individual_preds.values())
        uncertainty = np.std(pred_values)
        
        return {
            'risk_probability': ensemble_pred,
            'uncertainty': uncertainty,
            'individual_predictions': individual_preds,
            'risk_level': 'HIGH' if ensemble_pred > 0.6 else 'LOW'
        }


def demonstrate_models():
    """
    Demonstrate all ML models with sample data and results
    """
    print("=" * 60)
    print("QUANTLIB ML MODELS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Simulate correlated stock prices
    returns = np.random.multivariate_normal(
        [0.0005, 0.0003, 0.0004],  # Mean returns
        [[0.0004, 0.0001, 0.0002],  # Covariance matrix
         [0.0001, 0.0003, 0.0001],
         [0.0002, 0.0001, 0.0005]],
        n_days
    )
    
    prices = pd.DataFrame({
        'AAPL': 100 * np.cumprod(1 + returns[:, 0]),
        'GOOGL': 150 * np.cumprod(1 + returns[:, 1]),
        'MSFT': 120 * np.cumprod(1 + returns[:, 2])
    }, index=dates)
    
    # Add volume and OHLC data
    for col in prices.columns:
        prices[f'{col}_volume'] = np.random.lognormal(15, 0.5, n_days)
        prices[f'{col}_high'] = prices[col] * (1 + np.random.uniform(0, 0.02, n_days))
        prices[f'{col}_low'] = prices[col] * (1 - np.random.uniform(0, 0.02, n_days))
    
    # Rename for consistency
    data = pd.DataFrame({
        'close': prices['AAPL'],
        'volume': prices['AAPL_volume'],
        'high': prices['AAPL_high'],
        'low': prices['AAPL_low']
    }, index=dates)
    
    print(f"\nGenerated sample data: {len(data)} days of market data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 1. LSTM Price Prediction
    print("\n" + "="*50)
    print("1. LSTM PRICE PREDICTION MODEL")
    print("="*50)
    
    if TENSORFLOW_AVAILABLE:
        lstm_model = LSTMPricePredictionModel(sequence_length=30)
        
        try:
            # Train model
            train_data = data.iloc[:-100]  # Leave last 100 days for testing
            training_results = lstm_model.train(train_data)
            
            print(f"Training completed:")
            print(f"  Final Loss: {training_results['final_loss']:.6f}")
            print(f"  Validation Loss: {training_results['final_val_loss']:.6f}")
            print(f"  Final MAE: {training_results['final_mae']:.6f}")
            
            # Make predictions
            test_data = data.iloc[-130:-100]  # Use 30 days before test period
            predictions = lstm_model.predict(test_data, steps_ahead=5)
            actual_prices = data['close'].iloc[-100:-95].values
            
            print(f"\nPrediction Results (5-day forecast):")
            for i, (pred, actual) in enumerate(zip(predictions, actual_prices)):
                error = abs(pred - actual) / actual * 100
                print(f"  Day {i+1}: Predicted ${pred:.2f}, Actual ${actual:.2f}, Error: {error:.1f}%")
            
            avg_error = np.mean([abs(p-a)/a*100 for p, a in zip(predictions, actual_prices)])
            print(f"  Average Prediction Error: {avg_error:.1f}%")
            
        except Exception as e:
            print(f"LSTM training failed: {e}")
            print("This is normal if TensorFlow is not properly configured")
    else:
        print("TensorFlow not available - LSTM model skipped")
        print("Install with: pip install tensorflow")
    
    # 2. Portfolio Optimization RL
    print("\n" + "="*50)
    print("2. REINFORCEMENT LEARNING PORTFOLIO OPTIMIZATION")
    print("="*50)
    
    assets = ['AAPL', 'GOOGL', 'MSFT']
    rl_model = PortfolioOptimizationRL(assets)
    
    try:
        # Train RL agent
        rl_results = rl_model.train(prices[assets], episodes=500)
        
        print(f"RL Training completed:")
        print(f"  Final Episode Reward: {rl_results['final_reward']:.4f}")
        print(f"  Average Reward (last 100): {rl_results['average_reward']:.4f}")
        print(f"  States Learned: {rl_results['total_states']}")
        
        # Get optimal allocation
        recent_data = prices[assets].tail(50)
        optimal_allocation = rl_model.get_allocation(recent_data)
        
        print(f"\nOptimal Portfolio Allocation:")
        for asset, weight in zip(assets, optimal_allocation):
            print(f"  {asset}: {weight:.1%}")
        
        # Calculate performance
        portfolio_returns = (prices[assets].pct_change() * optimal_allocation).sum(axis=1)
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        print(f"\nPortfolio Performance:")
        print(f"  Annual Return: {annual_return:.1%}")
        print(f"  Annual Volatility: {annual_vol:.1%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        
    except Exception as e:
        print(f"RL training failed: {e}")
    
    # 3. Ensemble Risk Model
    print("\n" + "="*50)
    print("3. ENSEMBLE RISK PREDICTION MODEL")
    print("="*50)
    
    if SKLEARN_AVAILABLE:
        risk_model = EnsembleRiskModel()
        
        try:
            # Train ensemble
            train_data = prices[assets].iloc[:-100]
            risk_results = risk_model.train(train_data)
            
            print(f"Ensemble Training completed:")
            for model_name, accuracy in risk_results.items():
                print(f"  {model_name}: {accuracy:.3f}")
            
            # Make risk prediction
            recent_data = prices[assets].tail(50)
            risk_prediction = risk_model.predict_risk(recent_data)
            
            print(f"\nCurrent Risk Assessment:")
            print(f"  Risk Level: {risk_prediction['risk_level']}")
            print(f"  Risk Probability: {risk_prediction['risk_probability']:.1%}")
            print(f"  Model Uncertainty: {risk_prediction['uncertainty']:.3f}")
            
            print(f"\nIndividual Model Predictions:")
            for model, pred in risk_prediction['individual_predictions'].items():
                print(f"  {model}: {pred:.1%}")
            
        except Exception as e:
            print(f"Ensemble training failed: {e}")
    else:
        print("Scikit-learn not available - Ensemble model skipped")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("• LSTM models can predict price movements with reasonable accuracy")
    print("• RL agents learn optimal portfolio allocations through trial and error")
    print("• Ensemble models provide robust risk predictions with uncertainty estimates")
    print("• All models can be integrated into the quantlib backtesting framework")
    print("\nNext Steps:")
    print("• Experiment with different model architectures")
    print("• Add more sophisticated feature engineering")
    print("• Implement walk-forward validation")
    print("• Integrate with live trading systems")


if __name__ == "__main__":
    demonstrate_models()