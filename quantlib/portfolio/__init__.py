#!/usr/bin/env python3
"""
QuantLib Portfolio Management Module

This module provides portfolio optimization and management capabilities including:
- Modern Portfolio Theory (MPT) implementation
- Mean-variance optimization
- Efficient frontier calculation
- Risk budgeting and allocation
- Portfolio performance attribution
- Rebalancing strategies
"""

from .optimization import (
    MeanVarianceOptimizer,
    EfficientFrontier,
    RiskBudgetOptimizer,
    BlackLittermanOptimizer,
    HierarchicalRiskParity
)

from .allocation import (
    BaseAllocator,
    StrategicAssetAllocator,
    TacticalAssetAllocator,
    DynamicAssetAllocator,
    RiskBasedAllocator,
    AdaptiveAllocator,
    AllocationManager
)

from .rebalancing import (
    BaseRebalancer,
    CalendarRebalancer,
    ThresholdRebalancer,
    VolatilityRebalancer,
    CostOptimizedRebalancer,
    BandRebalancer,
    RebalancingManager,
    RebalanceTransaction,
    RebalanceResult
)

from .attribution import (
    BaseAttributor,
    BrinsonAttributor,
    SectorAttributor,
    FactorAttributor,
    MultiPeriodAttributor,
    CurrencyAttributor,
    AttributionManager,
    AttributionResult,
    SectorAttributionResult
)

__all__ = [
    # Optimization
    'MeanVarianceOptimizer',
    'EfficientFrontier',
    'RiskBudgetOptimizer',
    'BlackLittermanOptimizer',
    'HierarchicalRiskParity',
    
    # Allocation
    'BaseAllocator',
    'StrategicAssetAllocator',
    'TacticalAssetAllocator',
    'DynamicAssetAllocator',
    'RiskBasedAllocator',
    'AdaptiveAllocator',
    'AllocationManager',
    
    # Rebalancing
    'BaseRebalancer',
    'CalendarRebalancer',
    'ThresholdRebalancer',
    'VolatilityRebalancer',
    'CostOptimizedRebalancer',
    'BandRebalancer',
    'RebalancingManager',
    'RebalanceTransaction',
    'RebalanceResult',
    
    # Attribution
    'BaseAttributor',
    'BrinsonAttributor',
    'SectorAttributor',
    'FactorAttributor',
    'MultiPeriodAttributor',
    'CurrencyAttributor',
    'AttributionManager',
    'AttributionResult',
    'SectorAttributionResult'
]

__version__ = '1.0.0'
__author__ = 'QuantLib Team'