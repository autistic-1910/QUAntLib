#!/usr/bin/env python3
"""
Portfolio Optimization Module

Implements various portfolio optimization techniques including:
- Mean-Variance Optimization (Markowitz)
- Efficient Frontier calculation
- Risk budgeting
- Black-Litterman model
- Hierarchical Risk Parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.linalg import inv, sqrtm
import warnings

from quantlib.core.utils import Logger


class BaseOptimizer(ABC):
    """Base class for portfolio optimizers"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger.get_logger(f"optimizer_{name.lower()}")
        
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize portfolio weights"""
        pass
        
    def validate_inputs(self, returns: pd.DataFrame) -> bool:
        """Validate input data"""
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        if returns.isnull().any().any():
            warnings.warn("Returns data contains NaN values")
        return True


class MeanVarianceOptimizer(BaseOptimizer):
    """Mean-Variance Optimizer (Markowitz)"""
    
    def __init__(self, risk_aversion: float = 1.0):
        super().__init__("MeanVariance")
        self.risk_aversion = risk_aversion
        
    def optimize(self, returns: pd.DataFrame, 
                target_return: Optional[float] = None,
                target_volatility: Optional[float] = None,
                constraints: Optional[Dict] = None) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize portfolio using mean-variance optimization"""
        self.validate_inputs(returns)
        
        # Calculate expected returns and covariance matrix
        mu = returns.mean().values
        cov = returns.cov().values
        n_assets = len(mu)
        
        # Set up constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, mu) - target_return
            })
            
        if target_volatility is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x, np.dot(cov, x))) - target_volatility
            })
        
        # Add custom constraints
        if constraints:
            if 'min_weights' in constraints:
                bounds = [(constraints['min_weights'], 1.0) for _ in range(n_assets)]
            else:
                bounds = [(0.0, 1.0) for _ in range(n_assets)]
                
            if 'max_weights' in constraints:
                bounds = [(bounds[i][0], min(bounds[i][1], constraints['max_weights'])) 
                         for i in range(n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Objective function
        if target_return is not None or target_volatility is not None:
            # Minimize variance for given return or return for given volatility
            objective = lambda x: np.dot(x, np.dot(cov, x))
        else:
            # Maximize utility: return - (risk_aversion/2) * variance
            objective = lambda x: -(np.dot(x, mu) - 0.5 * self.risk_aversion * np.dot(x, np.dot(cov, x)))
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints_list)
        
        if not result.success:
            self.logger.warning(f"Optimization failed: {result.message}")
        
        weights = result.x
        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }


class EfficientFrontier(BaseOptimizer):
    """Efficient Frontier Calculator"""
    
    def __init__(self, n_points: int = 100):
        super().__init__("EfficientFrontier")
        self.n_points = n_points
        
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """Calculate efficient frontier"""
        self.validate_inputs(returns)
        
        mu = returns.mean().values
        cov = returns.cov().values
        n_assets = len(mu)
        
        # Calculate minimum variance portfolio
        min_var_weights = self._min_variance_portfolio(cov)
        min_return = np.dot(min_var_weights, mu)
        
        # Calculate maximum return portfolio
        max_return = np.max(mu)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, self.n_points)
        
        frontier_weights = []
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        
        optimizer = MeanVarianceOptimizer()
        
        for target_return in target_returns:
            try:
                result = optimizer.optimize(returns, target_return=target_return)
                if result['success']:
                    frontier_weights.append(result['weights'])
                    frontier_returns.append(result['expected_return'])
                    frontier_volatilities.append(result['volatility'])
                    frontier_sharpe_ratios.append(result['sharpe_ratio'])
            except:
                continue
        
        # Convert to arrays
        frontier_weights = np.array(frontier_weights)
        frontier_returns = np.array(frontier_returns)
        frontier_volatilities = np.array(frontier_volatilities)
        frontier_sharpe_ratios = np.array(frontier_sharpe_ratios)
        
        # Sort by volatility to ensure monotonic frontier
        if len(frontier_volatilities) > 0:
            sort_indices = np.argsort(frontier_volatilities)
            frontier_weights = frontier_weights[sort_indices]
            frontier_returns = frontier_returns[sort_indices]
            frontier_volatilities = frontier_volatilities[sort_indices]
            frontier_sharpe_ratios = frontier_sharpe_ratios[sort_indices]
        
        return {
            'weights': frontier_weights,
            'returns': frontier_returns,
            'volatilities': frontier_volatilities,
            'sharpe_ratios': frontier_sharpe_ratios,
            'assets': returns.columns.tolist()
        }
    
    def _min_variance_portfolio(self, cov: np.ndarray) -> np.ndarray:
        """Calculate minimum variance portfolio"""
        n = cov.shape[0]
        ones = np.ones((n, 1))
        
        try:
            cov_inv = inv(cov)
            weights = cov_inv @ ones / (ones.T @ cov_inv @ ones)
            return weights.flatten()
        except np.linalg.LinAlgError:
            # If covariance matrix is singular, use equal weights
            return np.ones(n) / n


class RiskBudgetOptimizer(BaseOptimizer):
    """Risk Budget Optimizer"""
    
    def __init__(self, risk_budgets: Optional[Dict[str, float]] = None):
        super().__init__("RiskBudget")
        self.risk_budgets = risk_budgets
        
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize portfolio using risk budgeting"""
        self.validate_inputs(returns)
        
        cov = returns.cov().values
        n_assets = len(returns.columns)
        
        # Default equal risk budgets
        if self.risk_budgets is None:
            risk_budgets = np.ones(n_assets) / n_assets
        else:
            risk_budgets = np.array([self.risk_budgets.get(asset, 1/n_assets) 
                                   for asset in returns.columns])
            risk_budgets = risk_budgets / np.sum(risk_budgets)  # Normalize
        
        # Objective function: minimize sum of squared deviations from target risk contributions
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            if portfolio_vol == 0:
                return 1e6
            
            marginal_contrib = np.dot(cov, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            return np.sum((risk_contrib - risk_budgets) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.001, 1.0) for _ in range(n_assets)]  # Small minimum to avoid division by zero
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        mu = returns.mean().values
        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'risk_budgets': risk_budgets,
            'success': result.success
        }


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman Optimizer"""
    
    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.025):
        super().__init__("BlackLitterman")
        self.risk_aversion = risk_aversion
        self.tau = tau
        
    def optimize(self, returns: pd.DataFrame, 
                market_caps: Optional[pd.Series] = None,
                views: Optional[Dict] = None,
                view_uncertainty: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize using Black-Litterman model"""
        self.validate_inputs(returns)
        
        cov = returns.cov().values
        n_assets = len(returns.columns)
        
        # Market capitalization weights (if not provided, use equal weights)
        if market_caps is None:
            w_market = np.ones(n_assets) / n_assets
        else:
            w_market = market_caps.values / market_caps.sum()
        
        # Implied equilibrium returns
        pi = self.risk_aversion * np.dot(cov, w_market)
        
        # If no views provided, return market portfolio
        if views is None:
            mu_bl = pi
            cov_bl = cov
        else:
            # Process views
            P, Q = self._process_views(views, returns.columns)
            
            # View uncertainty matrix
            if view_uncertainty is None:
                omega = np.diag(np.diag(P @ (self.tau * cov) @ P.T))
            else:
                omega = view_uncertainty
            
            # Black-Litterman formula
            tau_cov = self.tau * cov
            
            try:
                M1 = inv(tau_cov)
                M2 = P.T @ inv(omega) @ P
                M3 = inv(tau_cov) @ pi + P.T @ inv(omega) @ Q
                
                cov_bl = inv(M1 + M2)
                mu_bl = cov_bl @ M3
            except np.linalg.LinAlgError:
                self.logger.warning("Matrix inversion failed, using equilibrium returns")
                mu_bl = pi
                cov_bl = cov
        
        # Optimize portfolio
        optimizer = MeanVarianceOptimizer(self.risk_aversion)
        
        # Create temporary DataFrame for optimization
        temp_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
        temp_returns.iloc[0] = mu_bl  # Set first row to expected returns
        temp_returns = temp_returns.fillna(0)  # Fill rest with zeros
        
        # Override covariance calculation
        temp_returns._cov_override = cov_bl
        original_cov = temp_returns.cov
        temp_returns.cov = lambda: pd.DataFrame(cov_bl, 
                                               index=returns.columns, 
                                               columns=returns.columns)
        
        result = optimizer.optimize(temp_returns)
        
        # Restore original cov method
        temp_returns.cov = original_cov
        
        result['implied_returns'] = pi
        result['bl_returns'] = mu_bl
        result['bl_covariance'] = cov_bl
        
        return result
    
    def _process_views(self, views: Dict, assets: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
        """Process investor views into P and Q matrices"""
        n_assets = len(assets)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, (view_assets, expected_return) in enumerate(views.items()):
            if isinstance(view_assets, str):
                # Single asset view
                asset_idx = assets.get_loc(view_assets)
                P[i, asset_idx] = 1.0
            elif isinstance(view_assets, tuple) and len(view_assets) == 2:
                # Relative view (asset1 vs asset2)
                asset1_idx = assets.get_loc(view_assets[0])
                asset2_idx = assets.get_loc(view_assets[1])
                P[i, asset1_idx] = 1.0
                P[i, asset2_idx] = -1.0
            
            Q[i] = expected_return
        
        return P, Q


class HierarchicalRiskParity(BaseOptimizer):
    """Hierarchical Risk Parity Optimizer"""
    
    def __init__(self, linkage_method: str = 'ward'):
        super().__init__("HierarchicalRiskParity")
        self.linkage_method = linkage_method
        
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize using Hierarchical Risk Parity"""
        self.validate_inputs(returns)
        
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
        except ImportError:
            raise ImportError("scipy is required for Hierarchical Risk Parity")
        
        corr = returns.corr().values
        n_assets = len(returns.columns)
        
        # Convert correlation to distance matrix
        distance = np.sqrt(0.5 * (1 - corr))
        
        # Hierarchical clustering
        condensed_distance = squareform(distance, checks=False)
        linkage_matrix = linkage(condensed_distance, method=self.linkage_method)
        
        # Get clusters and calculate weights
        weights = self._get_hrp_weights(linkage_matrix, corr, n_assets)
        
        # Calculate portfolio metrics
        mu = returns.mean().values
        cov = returns.cov().values
        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'linkage_matrix': linkage_matrix,
            'success': True
        }
    
    def _get_hrp_weights(self, linkage_matrix: np.ndarray, 
                        corr: np.ndarray, n_assets: int) -> np.ndarray:
        """Calculate HRP weights from linkage matrix"""
        # Initialize weights
        weights = np.ones(n_assets)
        
        # Recursive bisection
        clusters = [list(range(n_assets))]
        
        for i in range(n_assets - 1):
            # Find clusters to split
            new_clusters = []
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Split cluster
                    left, right = self._split_cluster(cluster, linkage_matrix, i)
                    
                    # Calculate cluster variances
                    left_var = self._cluster_variance(left, corr)
                    right_var = self._cluster_variance(right, corr)
                    
                    # Allocate weights inversely proportional to variance
                    total_var = left_var + right_var
                    if total_var > 0:
                        left_weight = right_var / total_var
                        right_weight = left_var / total_var
                    else:
                        left_weight = right_weight = 0.5
                    
                    # Update weights
                    cluster_weight = np.sum(weights[cluster])
                    for idx in left:
                        weights[idx] *= left_weight
                    for idx in right:
                        weights[idx] *= right_weight
                    
                    new_clusters.extend([left, right])
                else:
                    new_clusters.append(cluster)
            
            clusters = new_clusters
        
        return weights / np.sum(weights)  # Normalize
    
    def _split_cluster(self, cluster: List[int], 
                      linkage_matrix: np.ndarray, step: int) -> Tuple[List[int], List[int]]:
        """Split cluster based on linkage matrix"""
        # Simple implementation - split in half
        mid = len(cluster) // 2
        return cluster[:mid], cluster[mid:]
    
    def _cluster_variance(self, cluster: List[int], corr: np.ndarray) -> float:
        """Calculate cluster variance"""
        if len(cluster) == 1:
            return 1.0
        
        cluster_corr = corr[np.ix_(cluster, cluster)]
        weights = np.ones(len(cluster)) / len(cluster)
        
        return np.dot(weights, np.dot(cluster_corr, weights))