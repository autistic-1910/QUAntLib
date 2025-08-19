#!/usr/bin/env python3
"""
Machine Learning Integration Module

Provides ML-based strategy components:
- Feature engineering for financial data
- Model training and prediction
- ML-based signal generation
- Model evaluation and validation
- Online learning capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from quantlib.core.utils import Logger
from quantlib.strategy.base import StrategySignal, SignalType
from quantlib.strategy.signals import SignalGenerator


@dataclass
class MLModelResult:
    """Result from ML model prediction"""
    prediction: Union[float, int, str]
    probability: Optional[float] = None
    confidence: float = 0.5
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    cross_val_scores: Optional[List[float]] = None


class FeatureEngineer:
    """Feature engineering for financial data"""
    
    def __init__(self):
        self.logger = Logger.get_logger("feature_engineer")
        self.scalers = {}
    
    def create_technical_features(self, data: pd.DataFrame, 
                                lookback_periods: List[int] = None) -> pd.DataFrame:
        """Create technical analysis features"""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 50]
        
        features = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                # Price-based features
                features[f'{col}_return_1d'] = data[col].pct_change(1)
                features[f'{col}_return_5d'] = data[col].pct_change(5)
                features[f'{col}_return_20d'] = data[col].pct_change(20)
                
                # Moving averages
                for period in lookback_periods:
                    if len(data) >= period:
                        features[f'{col}_sma_{period}'] = data[col].rolling(period).mean()
                        features[f'{col}_ema_{period}'] = data[col].ewm(span=period).mean()
                        features[f'{col}_std_{period}'] = data[col].rolling(period).std()
                        
                        # Relative position
                        sma = data[col].rolling(period).mean()
                        features[f'{col}_rel_sma_{period}'] = (data[col] - sma) / sma
                
                # Momentum features
                features[f'{col}_momentum_5'] = data[col] / data[col].shift(5) - 1
                features[f'{col}_momentum_20'] = data[col] / data[col].shift(20) - 1
                
                # Volatility features
                features[f'{col}_volatility_5'] = data[col].rolling(5).std()
                features[f'{col}_volatility_20'] = data[col].rolling(20).std()
                
                # High/Low features
                features[f'{col}_high_5'] = data[col].rolling(5).max()
                features[f'{col}_low_5'] = data[col].rolling(5).min()
                features[f'{col}_range_5'] = features[f'{col}_high_5'] - features[f'{col}_low_5']
        
        return features.fillna(method='ffill').fillna(0)
    
    def create_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset features"""
        features = pd.DataFrame(index=data.index)
        
        if len(data.columns) < 2:
            return features
        
        # Correlation features
        for i, col1 in enumerate(data.columns):
            for col2 in data.columns[i+1:]:
                if data[col1].dtype in ['float64', 'int64'] and data[col2].dtype in ['float64', 'int64']:
                    # Rolling correlation
                    features[f'corr_{col1}_{col2}_20'] = data[col1].rolling(20).corr(data[col2])
                    
                    # Relative performance
                    ret1 = data[col1].pct_change()
                    ret2 = data[col2].pct_change()
                    features[f'rel_perf_{col1}_{col2}'] = ret1 - ret2
        
        return features.fillna(0)
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame(index=data.index)
        
        if hasattr(data.index, 'dayofweek'):
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Cyclical encoding
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features.fillna(0)
    
    def scale_features(self, features: pd.DataFrame, 
                      scaler_type: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """Scale features using specified scaler"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available, returning unscaled features")
            return features
        
        if scaler_type not in self.scalers or fit:
            if scaler_type == 'standard':
                self.scalers[scaler_type] = StandardScaler()
            elif scaler_type == 'minmax':
                self.scalers[scaler_type] = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        scaler = self.scalers[scaler_type]
        
        if fit:
            scaled_data = scaler.fit_transform(features.fillna(0))
        else:
            scaled_data = scaler.transform(features.fillna(0))
        
        return pd.DataFrame(scaled_data, index=features.index, columns=features.columns)


class BaseMLModel(ABC):
    """Base class for ML models"""
    
    def __init__(self, name: str, model_type: str = 'classification'):
        self.name = name
        self.model_type = model_type  # 'classification' or 'regression'
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.logger = Logger.get_logger(f"ml_model_{name.lower()}")
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create the underlying ML model"""
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> ModelPerformance:
        """Train the model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for ML functionality")
        
        self.feature_names = list(X.columns)
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model(**kwargs)
        
        # Clean data
        X_clean = X.fillna(0)
        y_clean = y.fillna(0 if self.model_type == 'regression' else y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Train model
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
        # Evaluate performance
        performance = self._evaluate_model(X_clean, y_clean)
        
        self.logger.info(f"Model {self.name} trained successfully")
        return performance
    
    def predict(self, X: pd.DataFrame) -> MLModelResult:
        """Make prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_clean = X.fillna(0)
        
        # Ensure feature consistency
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X_clean.columns)
            if missing_features:
                for feature in missing_features:
                    X_clean[feature] = 0
            X_clean = X_clean[self.feature_names]
        
        prediction = self.model.predict(X_clean)[0]
        
        # Get probability for classification
        probability = None
        if self.model_type == 'classification' and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_clean)[0]
            probability = max(proba)
        
        # Get feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return MLModelResult(
            prediction=prediction,
            probability=probability,
            confidence=probability if probability else 0.5,
            feature_importance=feature_importance
        )
    
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> ModelPerformance:
        """Evaluate model performance"""
        predictions = self.model.predict(X)
        
        performance = ModelPerformance()
        
        if self.model_type == 'classification':
            performance.accuracy = accuracy_score(y, predictions)
            performance.precision = precision_score(y, predictions, average='weighted', zero_division=0)
            performance.recall = recall_score(y, predictions, average='weighted', zero_division=0)
            performance.f1_score = f1_score(y, predictions, average='weighted', zero_division=0)
        else:
            performance.mse = mean_squared_error(y, predictions)
            performance.mae = mean_absolute_error(y, predictions)
            performance.r2_score = r2_score(y, predictions)
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=3)
            performance.cross_val_scores = cv_scores.tolist()
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
        
        return performance


class RandomForestModel(BaseMLModel):
    """Random Forest model implementation"""
    
    def __init__(self, name: str, model_type: str = 'classification', **kwargs):
        super().__init__(name, model_type)
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        params = {**self.model_kwargs, **kwargs}
        
        if self.model_type == 'classification':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=params.get('random_state', 42)
            )
        else:
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=params.get('random_state', 42)
            )


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression model implementation"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, 'classification')
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        params = {**self.model_kwargs, **kwargs}
        return LogisticRegression(
            random_state=params.get('random_state', 42),
            max_iter=params.get('max_iter', 1000)
        )


class MLSignalGenerator(SignalGenerator):
    """ML-based signal generator"""
    
    def __init__(self, model: BaseMLModel, feature_engineer: FeatureEngineer,
                 lookback_window: int = 50, retrain_frequency: int = 100):
        super().__init__("MLSignals")
        self.model = model
        self.feature_engineer = feature_engineer
        self.lookback_window = lookback_window
        self.retrain_frequency = retrain_frequency
        self.prediction_count = 0
        self.last_training_data = None
    
    def generate_signals(self, data: pd.DataFrame, 
                        timestamp: datetime,
                        assets: List[str]) -> List[StrategySignal]:
        """Generate ML-based signals"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available, skipping ML signals")
            return []
        
        signals = []
        
        # Ensure we have enough data
        if len(data) < self.lookback_window:
            return signals
        
        # Get recent data
        recent_data = data.tail(self.lookback_window)
        
        # Create features
        try:
            features = self._create_features(recent_data)
            if features.empty:
                return signals
            
            # Train model if needed
            if not self.model.is_trained or self.prediction_count % self.retrain_frequency == 0:
                self._train_model(features, recent_data, assets)
            
            # Generate predictions for each asset
            for asset in assets:
                if asset in recent_data.columns:
                    signal = self._generate_asset_signal(asset, features, recent_data, timestamp)
                    if signal:
                        signals.append(signal)
            
            self.prediction_count += 1
            
        except Exception as e:
            self.logger.error(f"Error generating ML signals: {e}")
        
        return signals
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        # Technical features
        tech_features = self.feature_engineer.create_technical_features(data)
        
        # Cross-asset features
        cross_features = self.feature_engineer.create_cross_asset_features(data)
        
        # Time features
        time_features = self.feature_engineer.create_time_features(data)
        
        # Combine all features
        all_features = pd.concat([tech_features, cross_features, time_features], axis=1)
        
        # Scale features
        scaled_features = self.feature_engineer.scale_features(
            all_features, fit=not self.model.is_trained
        )
        
        return scaled_features.dropna()
    
    def _train_model(self, features: pd.DataFrame, data: pd.DataFrame, assets: List[str]):
        """Train the ML model"""
        if len(features) < 20:  # Need minimum data for training
            return
        
        # Create target variable (simplified: future return direction)
        target_data = []
        feature_data = []
        
        for asset in assets:
            if asset in data.columns:
                asset_returns = data[asset].pct_change().shift(-1)  # Next period return
                asset_target = (asset_returns > 0).astype(int)  # 1 for positive, 0 for negative
                
                # Align features and targets
                valid_indices = features.index.intersection(asset_target.index)
                if len(valid_indices) > 10:
                    target_data.extend(asset_target[valid_indices].values)
                    feature_data.extend(features.loc[valid_indices].values)
        
        if len(target_data) < 20:
            return
        
        # Prepare training data
        X_train = pd.DataFrame(feature_data, columns=features.columns)
        y_train = pd.Series(target_data)
        
        # Train model
        try:
            performance = self.model.train(X_train, y_train)
            self.logger.info(f"Model retrained. Accuracy: {performance.accuracy:.3f}")
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
    
    def _generate_asset_signal(self, asset: str, features: pd.DataFrame,
                             data: pd.DataFrame, timestamp: datetime) -> Optional[StrategySignal]:
        """Generate signal for a specific asset"""
        if features.empty or asset not in data.columns:
            return None
        
        try:
            # Get latest features
            latest_features = features.iloc[[-1]]
            
            # Make prediction
            result = self.model.predict(latest_features)
            
            # Convert prediction to signal
            if self.model.model_type == 'classification':
                if result.prediction == 1:  # Positive return predicted
                    signal_type = SignalType.BUY
                else:
                    signal_type = SignalType.SELL
            else:
                if result.prediction > 0:
                    signal_type = SignalType.BUY
                else:
                    signal_type = SignalType.SELL
            
            # Calculate signal strength based on confidence
            strength = result.confidence
            
            # Skip weak signals
            if strength < 0.6:
                return None
            
            return StrategySignal(
                timestamp=timestamp,
                asset=asset,
                signal_type=signal_type,
                strength=strength,
                confidence=result.confidence,
                target_position=result.prediction if self.model.model_type == 'regression' else 
                               (strength if signal_type == SignalType.BUY else -strength),
                metadata={
                    'model_name': self.model.name,
                    'prediction': result.prediction,
                    'feature_importance': result.feature_importance,
                    'analysis_type': 'ml'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {asset}: {e}")
            return None


class OnlineLearningModel(BaseMLModel):
    """Online learning model that updates incrementally"""
    
    def __init__(self, name: str, learning_rate: float = 0.01):
        super().__init__(name, 'regression')
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0.0
    
    def _create_model(self, **kwargs):
        # Simple online linear model
        return self
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit model (initialize weights)"""
        if self.weights is None:
            self.weights = np.random.normal(0, 0.01, X.shape[1])
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.weights is None:
            return np.zeros(len(X))
        return np.dot(X.values, self.weights) + self.bias
    
    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """Update model with new data"""
        if self.weights is None:
            self.fit(X, y)
            return
        
        # Simple gradient descent update
        for i in range(len(X)):
            x_i = X.iloc[i].values
            y_i = y.iloc[i]
            
            # Prediction
            pred = np.dot(x_i, self.weights) + self.bias
            
            # Error
            error = y_i - pred
            
            # Update weights
            self.weights += self.learning_rate * error * x_i
            self.bias += self.learning_rate * error