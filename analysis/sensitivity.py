"""
Sensitivity Analysis Engine for HotTakes Financial Model

Provides comprehensive sensitivity analysis capabilities:
- Single parameter sensitivity
- Multi-parameter sensitivity (tornado charts)
- Scenario analysis
- Monte Carlo simulation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Callable, Optional, Any
from copy import deepcopy
import itertools

from models.financial_model import (
    SystemParameters,
    UserSegment,
    UserSegmentDistribution,
    PortfolioAnalyzer,
    ProjectionEngine,
    TokenIssuanceEngine,
    CostCalculator
)


@dataclass
class ParameterRange:
    """Defines a parameter's sensitivity range."""
    name: str
    display_name: str
    base_value: float
    min_value: float
    max_value: float
    step: Optional[float] = None
    num_steps: int = 20
    is_percentage: bool = False
    unit: str = ""
    category: str = "General"

    def get_values(self) -> np.ndarray:
        """Generate array of values to test."""
        if self.step:
            return np.arange(self.min_value, self.max_value + self.step, self.step)
        return np.linspace(self.min_value, self.max_value, self.num_steps)


class ParameterRegistry:
    """Registry of all sensitizable parameters."""

    @staticmethod
    def get_all_parameters(params: SystemParameters) -> List[ParameterRange]:
        """Get all sensitizable parameters with their ranges."""
        return [
            # Token Issuance Parameters
            ParameterRange(
                name="floor_tokens",
                display_name="Floor Tokens/Day",
                base_value=params.floor_tokens,
                min_value=20,
                max_value=80,
                category="Token Issuance",
                unit="tokens"
            ),
            ParameterRange(
                name="base_tokens",
                display_name="Base Tokens/Day",
                base_value=params.base_tokens,
                min_value=80,
                max_value=200,
                category="Token Issuance",
                unit="tokens"
            ),
            ParameterRange(
                name="theta",
                display_name="Profitability Threshold (θ)",
                base_value=params.theta,
                min_value=1.0,
                max_value=6.0,
                category="Token Issuance",
                unit="$/month"
            ),
            ParameterRange(
                name="beta",
                display_name="Log Growth Rate (β)",
                base_value=params.beta,
                min_value=10,
                max_value=50,
                category="Token Issuance",
                unit=""
            ),
            ParameterRange(
                name="cap_tokens",
                display_name="Token Cap/Day",
                base_value=params.cap_tokens,
                min_value=200,
                max_value=500,
                category="Token Issuance",
                unit="tokens"
            ),

            # Token Economics
            ParameterRange(
                name="coins_per_dollar",
                display_name="Coins per Dollar",
                base_value=params.coins_per_dollar,
                min_value=500,
                max_value=2000,
                category="Token Economics",
                unit="coins"
            ),
            ParameterRange(
                name="coin_redemption_rate",
                display_name="Coin Redemption Rate",
                base_value=params.coin_redemption_rate,
                min_value=0.5,
                max_value=1.0,
                is_percentage=True,
                category="Token Economics",
                unit="%"
            ),
            ParameterRange(
                name="prize_cost_discount",
                display_name="Prize Cost Discount",
                base_value=params.prize_cost_discount,
                min_value=0.0,
                max_value=0.30,
                is_percentage=True,
                category="Token Economics",
                unit="%"
            ),

            # User Costs
            ParameterRange(
                name="kyc_cost_per_user",
                display_name="KYC Cost per User",
                base_value=params.kyc_cost_per_user,
                min_value=0.0,
                max_value=2.0,
                category="User Costs",
                unit="$"
            ),
            ParameterRange(
                name="linking_cost_per_month",
                display_name="Account Linking Cost/Month",
                base_value=params.linking_cost_per_month,
                min_value=0.3,
                max_value=1.5,
                category="User Costs",
                unit="$/month"
            ),
            ParameterRange(
                name="cloud_cost_per_mau",
                display_name="Cloud Cost per MAU",
                base_value=params.cloud_cost_per_mau,
                min_value=0.05,
                max_value=0.50,
                category="User Costs",
                unit="$/month"
            ),

            # Revenue Parameters
            ParameterRange(
                name="cpa_average",
                display_name="Average CPA",
                base_value=params.cpa_average,
                min_value=75,
                max_value=250,
                category="Revenue",
                unit="$"
            ),
            ParameterRange(
                name="retention_rev_per_bet",
                display_name="Retention Revenue per Bet",
                base_value=params.retention_rev_per_bet,
                min_value=0.20,
                max_value=1.00,
                category="Revenue",
                unit="$"
            ),

            # New User Economics
            ParameterRange(
                name="new_user_week1_tokens",
                display_name="New User Week 1 Tokens",
                base_value=params.new_user_week1_tokens,
                min_value=100,
                max_value=300,
                category="New User",
                unit="tokens/day"
            ),
            ParameterRange(
                name="new_user_week2_tokens",
                display_name="New User Week 2 Tokens",
                base_value=params.new_user_week2_tokens,
                min_value=100,
                max_value=250,
                category="New User",
                unit="tokens/day"
            ),
            ParameterRange(
                name="new_user_week3_tokens",
                display_name="New User Week 3 Tokens",
                base_value=params.new_user_week3_tokens,
                min_value=80,
                max_value=200,
                category="New User",
                unit="tokens/day"
            ),

            # Growth Parameters
            ParameterRange(
                name="monthly_growth_rate",
                display_name="Monthly MAU Growth Rate",
                base_value=params.monthly_growth_rate,
                min_value=-0.10,
                max_value=0.30,
                is_percentage=True,
                category="Growth",
                unit="%"
            ),
        ]


@dataclass
class SensitivityResult:
    """Result of a single sensitivity test."""
    parameter_name: str
    parameter_value: float
    metrics: Dict[str, float]


class SensitivityAnalyzer:
    """Performs sensitivity analysis on the financial model."""

    def __init__(self, base_params: Optional[SystemParameters] = None):
        self.base_params = base_params or SystemParameters()
        self.segments = UserSegmentDistribution.get_default_segments()

    def set_segments(self, segments: List[UserSegment]):
        """Update segment distribution."""
        self.segments = segments

    def _create_modified_params(self, param_name: str, value: float) -> SystemParameters:
        """Create a copy of params with one parameter modified."""
        params = deepcopy(self.base_params)
        setattr(params, param_name, value)
        params.update_derived()
        return params

    def _calculate_metrics(self, params: SystemParameters,
                           mau: int = 10000) -> Dict[str, float]:
        """Calculate key metrics for given parameters."""
        analyzer = PortfolioAnalyzer(params)
        portfolio = analyzer.analyze_portfolio(self.segments, mau)
        breakeven = analyzer.calculate_breakeven(self.segments)
        new_user_inv = CostCalculator(params).calculate_new_user_investment()

        return {
            'blended_margin_pct': portfolio['blended_margin_pct'],
            'net_monthly': portfolio['net'],
            'revenue_per_mau': portfolio['revenue_per_mau'],
            'cost_per_mau': portfolio['cost_per_mau'],
            'ltv': breakeven['blended_ltv'],
            'ltv_to_cac': breakeven['ltv_to_cac_ratio'],
            'new_user_investment': new_user_inv['total_investment'],
            'net_per_user': breakeven['net_per_user'],
            'total_revenue': portfolio['total_revenue'],
            'total_cost': portfolio['total_cost']
        }

    def single_parameter_sensitivity(self,
                                     param_range: ParameterRange,
                                     mau: int = 10000) -> pd.DataFrame:
        """Analyze sensitivity to a single parameter."""
        results = []
        values = param_range.get_values()

        for value in values:
            params = self._create_modified_params(param_range.name, value)
            metrics = self._calculate_metrics(params, mau)
            metrics['parameter_value'] = value
            metrics['parameter_name'] = param_range.display_name
            results.append(metrics)

        df = pd.DataFrame(results)
        return df

    def tornado_analysis(self,
                        param_ranges: Optional[List[ParameterRange]] = None,
                        target_metric: str = 'blended_margin_pct',
                        swing_pct: float = 0.20,
                        mau: int = 10000) -> pd.DataFrame:
        """
        Perform tornado analysis - show impact of each parameter at ±swing_pct.

        Returns DataFrame sorted by impact magnitude.
        """
        if param_ranges is None:
            param_ranges = ParameterRegistry.get_all_parameters(self.base_params)

        base_metrics = self._calculate_metrics(self.base_params, mau)
        base_value = base_metrics[target_metric]

        results = []

        for param in param_ranges:
            # Calculate low and high values
            low_value = param.base_value * (1 - swing_pct)
            high_value = param.base_value * (1 + swing_pct)

            # Ensure within bounds
            low_value = max(low_value, param.min_value)
            high_value = min(high_value, param.max_value)

            # Calculate metrics at low and high
            low_params = self._create_modified_params(param.name, low_value)
            high_params = self._create_modified_params(param.name, high_value)

            low_metrics = self._calculate_metrics(low_params, mau)
            high_metrics = self._calculate_metrics(high_params, mau)

            results.append({
                'parameter': param.display_name,
                'category': param.category,
                'base_value': param.base_value,
                'low_value': low_value,
                'high_value': high_value,
                'low_result': low_metrics[target_metric],
                'high_result': high_metrics[target_metric],
                'base_result': base_value,
                'low_delta': low_metrics[target_metric] - base_value,
                'high_delta': high_metrics[target_metric] - base_value,
                'impact_range': abs(high_metrics[target_metric] - low_metrics[target_metric])
            })

        df = pd.DataFrame(results)
        df = df.sort_values('impact_range', ascending=False)
        return df

    def two_way_sensitivity(self,
                           param1: ParameterRange,
                           param2: ParameterRange,
                           target_metric: str = 'blended_margin_pct',
                           mau: int = 10000,
                           resolution: int = 15) -> pd.DataFrame:
        """Perform two-way sensitivity analysis (heat map data)."""
        values1 = np.linspace(param1.min_value, param1.max_value, resolution)
        values2 = np.linspace(param2.min_value, param2.max_value, resolution)

        results = []

        for v1 in values1:
            for v2 in values2:
                params = deepcopy(self.base_params)
                setattr(params, param1.name, v1)
                setattr(params, param2.name, v2)
                params.update_derived()

                metrics = self._calculate_metrics(params, mau)

                results.append({
                    param1.display_name: v1,
                    param2.display_name: v2,
                    target_metric: metrics[target_metric]
                })

        return pd.DataFrame(results)

    def scenario_analysis(self,
                         scenarios: Dict[str, Dict[str, float]],
                         mau: int = 10000) -> pd.DataFrame:
        """
        Analyze multiple predefined scenarios.

        scenarios: Dict mapping scenario name to parameter overrides
        Example: {'Conservative': {'cpa_average': 100, 'monthly_growth_rate': 0.02}}
        """
        results = []

        # Add base case
        base_metrics = self._calculate_metrics(self.base_params, mau)
        base_metrics['scenario'] = 'Base Case'
        results.append(base_metrics)

        for scenario_name, overrides in scenarios.items():
            params = deepcopy(self.base_params)
            for param_name, value in overrides.items():
                setattr(params, param_name, value)
            params.update_derived()

            metrics = self._calculate_metrics(params, mau)
            metrics['scenario'] = scenario_name
            results.append(metrics)

        return pd.DataFrame(results)

    def breakeven_analysis(self,
                          param_name: str,
                          target_metric: str = 'net_monthly',
                          target_value: float = 0.0,
                          mau: int = 10000) -> Optional[float]:
        """Find parameter value that achieves target metric value."""
        param_range = None
        for p in ParameterRegistry.get_all_parameters(self.base_params):
            if p.name == param_name:
                param_range = p
                break

        if not param_range:
            return None

        # Binary search for breakeven
        low = param_range.min_value
        high = param_range.max_value
        tolerance = (high - low) / 1000

        while high - low > tolerance:
            mid = (low + high) / 2
            params = self._create_modified_params(param_name, mid)
            metrics = self._calculate_metrics(params, mau)

            if metrics[target_metric] > target_value:
                # Depends on parameter direction - assume higher param = higher metric for now
                high = mid
            else:
                low = mid

        return (low + high) / 2


class MonteCarloSimulator:
    """Monte Carlo simulation for risk analysis."""

    def __init__(self, base_params: Optional[SystemParameters] = None):
        self.base_params = base_params or SystemParameters()
        self.segments = UserSegmentDistribution.get_default_segments()

    def set_segments(self, segments: List[UserSegment]):
        """Update segment distribution."""
        self.segments = segments

    @dataclass
    class ParameterDistribution:
        """Defines probability distribution for a parameter."""
        name: str
        distribution: str  # 'normal', 'uniform', 'triangular'
        mean: Optional[float] = None
        std: Optional[float] = None
        low: Optional[float] = None
        high: Optional[float] = None
        mode: Optional[float] = None  # For triangular

        def sample(self, n: int = 1) -> np.ndarray:
            """Generate random samples from distribution."""
            if self.distribution == 'normal':
                return np.random.normal(self.mean, self.std, n)
            elif self.distribution == 'uniform':
                return np.random.uniform(self.low, self.high, n)
            elif self.distribution == 'triangular':
                return np.random.triangular(self.low, self.mode, self.high, n)
            else:
                raise ValueError(f"Unknown distribution: {self.distribution}")

    def get_default_distributions(self) -> List['MonteCarloSimulator.ParameterDistribution']:
        """Get default parameter distributions for simulation."""
        return [
            self.ParameterDistribution(
                name='cpa_average',
                distribution='triangular',
                low=100, mode=150, high=200
            ),
            self.ParameterDistribution(
                name='retention_rev_per_bet',
                distribution='triangular',
                low=0.30, mode=0.50, high=0.75
            ),
            self.ParameterDistribution(
                name='coin_redemption_rate',
                distribution='normal',
                mean=0.85, std=0.05
            ),
            self.ParameterDistribution(
                name='linking_cost_per_month',
                distribution='uniform',
                low=0.60, high=1.00
            ),
            self.ParameterDistribution(
                name='monthly_growth_rate',
                distribution='triangular',
                low=-0.02, mode=0.05, high=0.15
            ),
        ]

    def run_simulation(self,
                      param_distributions: Optional[List['MonteCarloSimulator.ParameterDistribution']] = None,
                      n_simulations: int = 1000,
                      mau: int = 10000,
                      projection_months: int = 12) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.

        Returns DataFrame with simulation results and statistics.
        """
        if param_distributions is None:
            param_distributions = self.get_default_distributions()

        results = []

        for i in range(n_simulations):
            # Sample parameters
            params = deepcopy(self.base_params)
            param_values = {}

            for dist in param_distributions:
                value = dist.sample(1)[0]
                # Clip to reasonable bounds
                if dist.low is not None:
                    value = max(value, dist.low * 0.5)
                if dist.high is not None:
                    value = min(value, dist.high * 1.5)
                setattr(params, dist.name, value)
                param_values[dist.name] = value

            params.update_derived()

            # Calculate metrics
            analyzer = PortfolioAnalyzer(params)
            portfolio = analyzer.analyze_portfolio(self.segments, mau)

            # Project over time
            projector = ProjectionEngine(params)
            projections = projector.project_monthly(
                self.segments, mau, projection_months
            )

            # Aggregate projection results
            total_revenue = sum(p['revenue'] for p in projections)
            total_cost = sum(p['cost'] for p in projections)
            total_net = sum(p['net'] for p in projections)
            final_mau = projections[-1]['mau']

            result = {
                'simulation_id': i,
                'blended_margin_pct': portfolio['blended_margin_pct'],
                'monthly_net': portfolio['net'],
                'total_revenue_12m': total_revenue,
                'total_cost_12m': total_cost,
                'total_net_12m': total_net,
                'final_mau': final_mau,
                **param_values
            }
            results.append(result)

        return pd.DataFrame(results)

    def analyze_results(self, results: pd.DataFrame) -> Dict:
        """Analyze Monte Carlo simulation results."""
        metrics = ['blended_margin_pct', 'monthly_net', 'total_net_12m']

        analysis = {}

        for metric in metrics:
            if metric not in results.columns:
                continue

            values = results[metric]
            analysis[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'p5': values.quantile(0.05),
                'p25': values.quantile(0.25),
                'p50': values.quantile(0.50),
                'p75': values.quantile(0.75),
                'p95': values.quantile(0.95),
                'prob_positive': (values > 0).mean(),
                'prob_above_20pct': (values > 20).mean() if 'margin' in metric else None
            }

        return analysis


class SegmentSensitivityAnalyzer:
    """Analyze sensitivity to user segment distribution changes."""

    def __init__(self, base_params: Optional[SystemParameters] = None):
        self.base_params = base_params or SystemParameters()
        self.base_segments = UserSegmentDistribution.get_default_segments()

    def vary_segment_percentage(self,
                                segment_name: str,
                                percentage_range: Tuple[float, float],
                                num_steps: int = 20,
                                mau: int = 10000) -> pd.DataFrame:
        """
        Vary one segment's percentage and redistribute to others proportionally.
        """
        results = []
        percentages = np.linspace(percentage_range[0], percentage_range[1], num_steps)

        # Find target segment
        target_idx = None
        for i, seg in enumerate(self.base_segments):
            if seg.name == segment_name:
                target_idx = i
                break

        if target_idx is None:
            raise ValueError(f"Segment {segment_name} not found")

        base_pct = self.base_segments[target_idx].percentage

        for pct in percentages:
            # Create modified segments
            segments = deepcopy(self.base_segments)
            segments[target_idx].percentage = pct

            # Redistribute remaining percentage proportionally
            delta = pct - base_pct
            other_total = 1 - base_pct

            for i, seg in enumerate(segments):
                if i != target_idx:
                    original_pct = self.base_segments[i].percentage
                    seg.percentage = original_pct - (delta * original_pct / other_total)

            # Calculate metrics
            analyzer = PortfolioAnalyzer(self.base_params)
            portfolio = analyzer.analyze_portfolio(segments, mau)

            results.append({
                'segment': segment_name,
                'percentage': pct,
                'blended_margin_pct': portfolio['blended_margin_pct'],
                'net_monthly': portfolio['net'],
                'revenue_per_mau': portfolio['revenue_per_mau'],
                'cost_per_mau': portfolio['cost_per_mau']
            })

        return pd.DataFrame(results)

    def vary_segment_revenue(self,
                             segment_name: str,
                             revenue_range: Tuple[float, float],
                             num_steps: int = 20,
                             mau: int = 10000) -> pd.DataFrame:
        """Vary one segment's monthly revenue."""
        results = []
        revenues = np.linspace(revenue_range[0], revenue_range[1], num_steps)

        target_idx = None
        for i, seg in enumerate(self.base_segments):
            if seg.name == segment_name:
                target_idx = i
                break

        if target_idx is None:
            raise ValueError(f"Segment {segment_name} not found")

        for rev in revenues:
            segments = deepcopy(self.base_segments)
            segments[target_idx].monthly_revenue = rev
            # Also update avg_ups since revenue drives UPS
            segments[target_idx].avg_ups = rev * 0.8  # Rough approximation

            analyzer = PortfolioAnalyzer(self.base_params)
            portfolio = analyzer.analyze_portfolio(segments, mau)

            results.append({
                'segment': segment_name,
                'monthly_revenue': rev,
                'blended_margin_pct': portfolio['blended_margin_pct'],
                'net_monthly': portfolio['net'],
                'revenue_per_mau': portfolio['revenue_per_mau']
            })

        return pd.DataFrame(results)

    def vary_segment_churn(self,
                          segment_name: str,
                          churn_range: Tuple[float, float],
                          num_steps: int = 20,
                          mau: int = 10000) -> pd.DataFrame:
        """Vary one segment's monthly churn rate and see LTV impact."""
        results = []
        churns = np.linspace(churn_range[0], churn_range[1], num_steps)

        target_idx = None
        for i, seg in enumerate(self.base_segments):
            if seg.name == segment_name:
                target_idx = i
                break

        if target_idx is None:
            raise ValueError(f"Segment {segment_name} not found")

        for churn in churns:
            segments = deepcopy(self.base_segments)
            segments[target_idx].monthly_churn_rate = churn

            analyzer = PortfolioAnalyzer(self.base_params)
            ltv_result = analyzer.calculate_ltv(segments[target_idx], months=12)
            blended_ltv = analyzer.calculate_blended_ltv(segments, months=12)

            results.append({
                'segment': segment_name,
                'monthly_churn_rate': churn,
                'segment_ltv': ltv_result['ltv'],
                'blended_ltv': blended_ltv,
                'final_retention': ltv_result['final_retention']
            })

        return pd.DataFrame(results)
