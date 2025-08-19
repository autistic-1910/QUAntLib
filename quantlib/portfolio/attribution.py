#!/usr/bin/env python3
"""
Portfolio Attribution Module

Implements various portfolio attribution analysis methods including:
- Brinson-Hood-Beebower attribution
- Sector/Factor attribution
- Security selection and allocation effects
- Multi-period attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from quantlib.core.utils import Logger


@dataclass
class AttributionResult:
    """Results of attribution analysis"""
    total_return: float
    benchmark_return: float
    excess_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    currency_effect: Optional[float] = None
    
    @property
    def total_active_return(self) -> float:
        """Total active return (should equal excess return)"""
        return self.allocation_effect + self.selection_effect + self.interaction_effect


@dataclass
class SectorAttributionResult:
    """Sector-level attribution results"""
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    
    @property
    def total_contribution(self) -> float:
        """Total contribution from this sector"""
        return self.allocation_effect + self.selection_effect + self.interaction_effect


class BaseAttributor:
    """Base class for attribution analysis"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger.get_logger(f"attributor_{name.lower()}")
        
    def validate_inputs(self, portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       portfolio_weights: pd.Series,
                       benchmark_weights: pd.Series) -> bool:
        """Validate input data for attribution analysis"""
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have same length")
        
        if len(portfolio_weights) != len(benchmark_weights):
            raise ValueError("Portfolio and benchmark weights must have same length")
        
        # Check if weights sum to approximately 1
        port_sum = portfolio_weights.sum()
        bench_sum = benchmark_weights.sum()
        
        if abs(port_sum - 1.0) > 0.01:
            self.logger.warning(f"Portfolio weights sum to {port_sum:.4f}, not 1.0")
        
        if abs(bench_sum - 1.0) > 0.01:
            self.logger.warning(f"Benchmark weights sum to {bench_sum:.4f}, not 1.0")
        
        return True


class BrinsonAttributor(BaseAttributor):
    """Brinson-Hood-Beebower Attribution Analysis"""
    
    def __init__(self):
        super().__init__("BrinsonAttribution")
        
    def analyze(self, portfolio_returns: pd.Series,
               benchmark_returns: pd.Series,
               portfolio_weights: pd.Series,
               benchmark_weights: pd.Series,
               sector_mapping: Optional[Dict[str, str]] = None) -> AttributionResult:
        """Perform Brinson attribution analysis"""
        
        self.validate_inputs(portfolio_returns, benchmark_returns,
                           portfolio_weights, benchmark_weights)
        
        # Calculate portfolio and benchmark returns
        portfolio_return = (portfolio_weights * portfolio_returns).sum()
        benchmark_return = (benchmark_weights * benchmark_returns).sum()
        excess_return = portfolio_return - benchmark_return
        
        # Calculate attribution effects
        allocation_effect = self._calculate_allocation_effect(
            portfolio_weights, benchmark_weights, benchmark_returns, benchmark_return
        )
        
        selection_effect = self._calculate_selection_effect(
            benchmark_weights, portfolio_returns, benchmark_returns
        )
        
        interaction_effect = self._calculate_interaction_effect(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )
        
        return AttributionResult(
            total_return=portfolio_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect
        )
    
    def _calculate_allocation_effect(self, portfolio_weights: pd.Series,
                                   benchmark_weights: pd.Series,
                                   benchmark_returns: pd.Series,
                                   benchmark_return: float) -> float:
        """Calculate allocation effect"""
        weight_diff = portfolio_weights - benchmark_weights
        return_diff = benchmark_returns - benchmark_return
        return (weight_diff * return_diff).sum()
    
    def _calculate_selection_effect(self, benchmark_weights: pd.Series,
                                  portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """Calculate selection effect"""
        return_diff = portfolio_returns - benchmark_returns
        return (benchmark_weights * return_diff).sum()
    
    def _calculate_interaction_effect(self, portfolio_weights: pd.Series,
                                    benchmark_weights: pd.Series,
                                    portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series) -> float:
        """Calculate interaction effect"""
        weight_diff = portfolio_weights - benchmark_weights
        return_diff = portfolio_returns - benchmark_returns
        return (weight_diff * return_diff).sum()


class SectorAttributor(BaseAttributor):
    """Sector-based attribution analysis"""
    
    def __init__(self):
        super().__init__("SectorAttribution")
        
    def analyze_by_sector(self, portfolio_returns: pd.Series,
                         benchmark_returns: pd.Series,
                         portfolio_weights: pd.Series,
                         benchmark_weights: pd.Series,
                         sector_mapping: Dict[str, str]) -> Dict[str, SectorAttributionResult]:
        """Perform sector-level attribution analysis"""
        
        self.validate_inputs(portfolio_returns, benchmark_returns,
                           portfolio_weights, benchmark_weights)
        
        # Group by sectors
        sectors = set(sector_mapping.values())
        results = {}
        
        for sector in sectors:
            # Get assets in this sector
            sector_assets = [asset for asset, sec in sector_mapping.items() if sec == sector]
            sector_assets = [asset for asset in sector_assets 
                           if asset in portfolio_returns.index]
            
            if not sector_assets:
                continue
            
            # Calculate sector weights and returns
            sector_port_weights = portfolio_weights[sector_assets]
            sector_bench_weights = benchmark_weights[sector_assets]
            sector_port_returns = portfolio_returns[sector_assets]
            sector_bench_returns = benchmark_returns[sector_assets]
            
            # Aggregate to sector level
            sector_port_weight = sector_port_weights.sum()
            sector_bench_weight = sector_bench_weights.sum()
            
            if sector_port_weight > 0:
                sector_port_return = (sector_port_weights * sector_port_returns).sum() / sector_port_weight
            else:
                sector_port_return = 0.0
            
            if sector_bench_weight > 0:
                sector_bench_return = (sector_bench_weights * sector_bench_returns).sum() / sector_bench_weight
            else:
                sector_bench_return = 0.0
            
            # Calculate benchmark return for allocation effect
            total_bench_return = (benchmark_weights * benchmark_returns).sum()
            
            # Calculate attribution effects
            allocation_effect = (sector_port_weight - sector_bench_weight) * \
                              (sector_bench_return - total_bench_return)
            
            selection_effect = sector_bench_weight * \
                             (sector_port_return - sector_bench_return)
            
            interaction_effect = (sector_port_weight - sector_bench_weight) * \
                               (sector_port_return - sector_bench_return)
            
            results[sector] = SectorAttributionResult(
                sector=sector,
                portfolio_weight=sector_port_weight,
                benchmark_weight=sector_bench_weight,
                portfolio_return=sector_port_return,
                benchmark_return=sector_bench_return,
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect
            )
        
        return results
    
    def summarize_sector_attribution(self, 
                                   sector_results: Dict[str, SectorAttributionResult]) -> AttributionResult:
        """Summarize sector attribution results"""
        
        total_allocation = sum(r.allocation_effect for r in sector_results.values())
        total_selection = sum(r.selection_effect for r in sector_results.values())
        total_interaction = sum(r.interaction_effect for r in sector_results.values())
        
        # Calculate total returns
        total_port_return = sum(r.portfolio_weight * r.portfolio_return 
                              for r in sector_results.values())
        total_bench_return = sum(r.benchmark_weight * r.benchmark_return 
                               for r in sector_results.values())
        
        return AttributionResult(
            total_return=total_port_return,
            benchmark_return=total_bench_return,
            excess_return=total_port_return - total_bench_return,
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction
        )


class FactorAttributor(BaseAttributor):
    """Factor-based attribution analysis"""
    
    def __init__(self):
        super().__init__("FactorAttribution")
        
    def analyze_factor_attribution(self, portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 portfolio_weights: pd.Series,
                                 benchmark_weights: pd.Series,
                                 factor_exposures: pd.DataFrame,
                                 factor_returns: pd.Series) -> Dict[str, float]:
        """Perform factor-based attribution analysis"""
        
        self.validate_inputs(portfolio_returns, benchmark_returns,
                           portfolio_weights, benchmark_weights)
        
        # Calculate factor exposures for portfolio and benchmark
        portfolio_factor_exposure = (portfolio_weights.values.reshape(-1, 1) * 
                                   factor_exposures.values).sum(axis=0)
        benchmark_factor_exposure = (benchmark_weights.values.reshape(-1, 1) * 
                                    factor_exposures.values).sum(axis=0)
        
        # Calculate factor attribution
        factor_attribution = {}
        
        for i, factor in enumerate(factor_exposures.columns):
            exposure_diff = portfolio_factor_exposure[i] - benchmark_factor_exposure[i]
            factor_return = factor_returns[factor] if factor in factor_returns.index else 0.0
            factor_attribution[factor] = exposure_diff * factor_return
        
        return factor_attribution


class MultiPeriodAttributor(BaseAttributor):
    """Multi-period attribution analysis"""
    
    def __init__(self, compounding_method: str = 'geometric'):
        super().__init__("MultiPeriodAttribution")
        self.compounding_method = compounding_method
        
    def analyze_multi_period(self, portfolio_returns: pd.DataFrame,
                           benchmark_returns: pd.DataFrame,
                           portfolio_weights: pd.DataFrame,
                           benchmark_weights: pd.DataFrame) -> pd.DataFrame:
        """Perform multi-period attribution analysis"""
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark data must have same length")
        
        results = []
        attributor = BrinsonAttributor()
        
        for date in portfolio_returns.index:
            try:
                port_ret = portfolio_returns.loc[date]
                bench_ret = benchmark_returns.loc[date]
                port_weights = portfolio_weights.loc[date]
                bench_weights = benchmark_weights.loc[date]
                
                # Remove NaN values
                valid_assets = port_ret.dropna().index.intersection(
                    bench_ret.dropna().index
                ).intersection(
                    port_weights.dropna().index
                ).intersection(
                    bench_weights.dropna().index
                )
                
                if len(valid_assets) == 0:
                    continue
                
                result = attributor.analyze(
                    port_ret[valid_assets],
                    bench_ret[valid_assets],
                    port_weights[valid_assets],
                    bench_weights[valid_assets]
                )
                
                results.append({
                    'date': date,
                    'total_return': result.total_return,
                    'benchmark_return': result.benchmark_return,
                    'excess_return': result.excess_return,
                    'allocation_effect': result.allocation_effect,
                    'selection_effect': result.selection_effect,
                    'interaction_effect': result.interaction_effect
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing date {date}: {e}")
                continue
        
        return pd.DataFrame(results).set_index('date')
    
    def compound_attribution(self, attribution_df: pd.DataFrame) -> AttributionResult:
        """Compound multi-period attribution results"""
        
        if self.compounding_method == 'geometric':
            # Geometric compounding
            total_return = (1 + attribution_df['total_return']).prod() - 1
            benchmark_return = (1 + attribution_df['benchmark_return']).prod() - 1
            
            # For attribution effects, use arithmetic sum (common practice)
            allocation_effect = attribution_df['allocation_effect'].sum()
            selection_effect = attribution_df['selection_effect'].sum()
            interaction_effect = attribution_df['interaction_effect'].sum()
            
        else:  # arithmetic
            total_return = attribution_df['total_return'].sum()
            benchmark_return = attribution_df['benchmark_return'].sum()
            allocation_effect = attribution_df['allocation_effect'].sum()
            selection_effect = attribution_df['selection_effect'].sum()
            interaction_effect = attribution_df['interaction_effect'].sum()
        
        excess_return = total_return - benchmark_return
        
        return AttributionResult(
            total_return=total_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect
        )


class CurrencyAttributor(BaseAttributor):
    """Currency attribution analysis for international portfolios"""
    
    def __init__(self, base_currency: str = 'USD'):
        super().__init__("CurrencyAttribution")
        self.base_currency = base_currency
        
    def analyze_currency_attribution(self, 
                                   local_returns: pd.Series,
                                   currency_returns: pd.Series,
                                   portfolio_weights: pd.Series,
                                   benchmark_weights: pd.Series,
                                   currency_mapping: Dict[str, str]) -> Dict[str, float]:
        """Analyze currency attribution effects"""
        
        currency_attribution = {}
        
        # Group assets by currency
        currencies = set(currency_mapping.values())
        
        for currency in currencies:
            if currency == self.base_currency:
                continue  # No currency effect for base currency
            
            # Get assets in this currency
            currency_assets = [asset for asset, curr in currency_mapping.items() 
                             if curr == currency]
            currency_assets = [asset for asset in currency_assets 
                             if asset in portfolio_weights.index]
            
            if not currency_assets:
                continue
            
            # Calculate currency exposure difference
            port_currency_weight = portfolio_weights[currency_assets].sum()
            bench_currency_weight = benchmark_weights[currency_assets].sum()
            currency_exposure_diff = port_currency_weight - bench_currency_weight
            
            # Get currency return
            currency_return = currency_returns.get(currency, 0.0)
            
            # Calculate currency attribution
            currency_attribution[currency] = currency_exposure_diff * currency_return
        
        return currency_attribution


class AttributionManager:
    """Manager class for coordinating attribution analysis"""
    
    def __init__(self):
        self.logger = Logger.get_logger("attribution_manager")
        self.attributors = {
            'brinson': BrinsonAttributor(),
            'sector': SectorAttributor(),
            'factor': FactorAttributor(),
            'multi_period': MultiPeriodAttributor(),
            'currency': CurrencyAttributor()
        }
        
    def comprehensive_attribution(self, 
                                portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                portfolio_weights: pd.Series,
                                benchmark_weights: pd.Series,
                                sector_mapping: Optional[Dict[str, str]] = None,
                                factor_exposures: Optional[pd.DataFrame] = None,
                                factor_returns: Optional[pd.Series] = None,
                                currency_mapping: Optional[Dict[str, str]] = None,
                                currency_returns: Optional[pd.Series] = None) -> Dict[str, any]:
        """Perform comprehensive attribution analysis"""
        
        results = {}
        
        # Brinson attribution
        try:
            brinson_result = self.attributors['brinson'].analyze(
                portfolio_returns, benchmark_returns,
                portfolio_weights, benchmark_weights
            )
            results['brinson'] = brinson_result
        except Exception as e:
            self.logger.error(f"Brinson attribution failed: {e}")
        
        # Sector attribution
        if sector_mapping:
            try:
                sector_results = self.attributors['sector'].analyze_by_sector(
                    portfolio_returns, benchmark_returns,
                    portfolio_weights, benchmark_weights,
                    sector_mapping
                )
                results['sector'] = sector_results
                results['sector_summary'] = self.attributors['sector'].summarize_sector_attribution(sector_results)
            except Exception as e:
                self.logger.error(f"Sector attribution failed: {e}")
        
        # Factor attribution
        if factor_exposures is not None and factor_returns is not None:
            try:
                factor_results = self.attributors['factor'].analyze_factor_attribution(
                    portfolio_returns, benchmark_returns,
                    portfolio_weights, benchmark_weights,
                    factor_exposures, factor_returns
                )
                results['factor'] = factor_results
            except Exception as e:
                self.logger.error(f"Factor attribution failed: {e}")
        
        # Currency attribution
        if currency_mapping and currency_returns is not None:
            try:
                currency_results = self.attributors['currency'].analyze_currency_attribution(
                    portfolio_returns, currency_returns,
                    portfolio_weights, benchmark_weights,
                    currency_mapping
                )
                results['currency'] = currency_results
            except Exception as e:
                self.logger.error(f"Currency attribution failed: {e}")
        
        return results
    
    def create_attribution_report(self, attribution_results: Dict[str, any]) -> pd.DataFrame:
        """Create a summary attribution report"""
        
        report_data = []
        
        if 'brinson' in attribution_results:
            brinson = attribution_results['brinson']
            report_data.append({
                'Analysis': 'Overall',
                'Total Return': brinson.total_return,
                'Benchmark Return': brinson.benchmark_return,
                'Excess Return': brinson.excess_return,
                'Allocation Effect': brinson.allocation_effect,
                'Selection Effect': brinson.selection_effect,
                'Interaction Effect': brinson.interaction_effect
            })
        
        if 'sector' in attribution_results:
            for sector, result in attribution_results['sector'].items():
                report_data.append({
                    'Analysis': f'Sector: {sector}',
                    'Portfolio Weight': result.portfolio_weight,
                    'Benchmark Weight': result.benchmark_weight,
                    'Portfolio Return': result.portfolio_return,
                    'Benchmark Return': result.benchmark_return,
                    'Allocation Effect': result.allocation_effect,
                    'Selection Effect': result.selection_effect,
                    'Interaction Effect': result.interaction_effect
                })
        
        return pd.DataFrame(report_data)