"""
HotTakes User Profitability Maximization Engine - Core Financial Model

This module implements the mathematical framework for:
- User Profitability Score (UPS) calculation
- Token issuance functions
- Margin analysis
- Portfolio-level economics
- Lifetime value calculations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math


@dataclass
class SystemParameters:
    """Core system parameters that can be sensitized."""

    # Token Issuance Parameters
    floor_tokens: int = 40  # Minimum tokens/day for unprofitable users
    base_tokens: int = 120  # Standard engaged user rate
    theta: float = 3.0  # UPS threshold for "profitable" classification ($/month)
    beta: float = 25.0  # Logarithmic growth rate for high-value users
    cap_tokens: int = 350  # Maximum tokens/day

    # Confidence Parameters
    confidence_k: int = 15  # Bayesian confidence constant
    prior_ups: float = 2.50  # Prior UPS for new users ($/month)

    # Token Economics
    tokens_per_coin: float = 1.0  # Tokens to coins ratio
    coins_per_dollar: float = 1000.0  # Coins to dollar ratio (prize value)
    token_to_dollar: float = field(init=False)  # Derived: tokens to dollar

    # Prize Economics
    coin_redemption_rate: float = 0.85  # % of coins that get redeemed
    prize_cost_discount: float = 0.0  # Discount on prize costs (0-0.3 typical)

    # User Costs
    kyc_cost_per_user: float = 0.50  # One-time KYC cost
    linking_cost_per_month: float = 0.80  # Monthly account linking cost
    cloud_cost_per_mau: float = 0.15  # Monthly cloud computing per MAU
    user_acquisition_cost: float = 5.0  # Marketing cost to acquire user (CAC)

    # Revenue Parameters
    cpa_average: float = 150.0  # Average CPA per sportsbook conversion
    retention_rev_per_bet: float = 0.50  # Revenue per deeplink bet
    avg_bets_per_active_user: float = 4.0  # Average monthly bets for active users

    # New User Economics
    new_user_week1_tokens: int = 200
    new_user_week2_tokens: int = 175
    new_user_week3_tokens: int = 150

    # Growth Parameters
    monthly_growth_rate: float = 0.05  # 5% monthly MAU growth

    # Revenue Assumption Sliders (0.0 = pessimistic, 1.0 = optimistic)
    segment_optimism: float = 0.5  # User quality mix (default: balanced)
    cpa_optimism: float = 0.2  # CPA conversion rate (default: conservative)
    deeplink_optimism: float = 0.5  # Deeplink engagement (default: balanced)

    def __post_init__(self):
        self.token_to_dollar = 1.0 / (self.tokens_per_coin * self.coins_per_dollar)

    def update_derived(self):
        """Recalculate derived values after parameter changes."""
        self.token_to_dollar = 1.0 / (self.tokens_per_coin * self.coins_per_dollar)


@dataclass
class UserSegment:
    """Represents a user segment with its characteristics."""
    name: str
    percentage: float  # % of MAU
    ups_range: Tuple[float, float]  # (min, max) UPS
    avg_ups: float
    monthly_revenue: float
    monthly_churn_rate: float
    linking_rate: float  # % of segment that links accounts
    deeplink_rate: float  # Avg deeplinks per month
    conversion_rate: float  # CPA conversion rate


class UserSegmentDistribution:
    """Default user segment distribution based on the document."""

    # Segment percentage ranges: [pessimistic, base, optimistic]
    SEGMENT_PERCENTAGES = {
        'Unprofitable': [0.55, 0.40, 0.25],
        'Low-Value': [0.25, 0.25, 0.25],
        'Moderate': [0.12, 0.20, 0.28],
        'High-Value': [0.06, 0.12, 0.17],
        'Power': [0.02, 0.03, 0.05],
    }

    # CPA conversion rate ranges: [very_conservative, conservative, base, optimistic]
    # Unprofitable and Low-Value are always 0% at conservative settings
    CPA_RATES = {
        'Unprofitable': [0.0, 0.0, 0.0, 0.0],  # Never converts
        'Low-Value': [0.0, 0.0, 0.02, 0.05],
        'Moderate': [0.0, 0.01, 0.05, 0.10],
        'High-Value': [0.01, 0.03, 0.10, 0.20],
        'Power': [0.03, 0.05, 0.15, 0.30],
    }

    # Deeplink rates: [pessimistic, base, optimistic]
    DEEPLINK_RATES = {
        'Unprofitable': [0.0, 0.0, 0.0],  # Never uses deeplinks
        'Low-Value': [0.5, 1.0, 2.0],
        'Moderate': [1.5, 3.0, 6.0],
        'High-Value': [3.0, 6.0, 12.0],
        'Power': [6.0, 12.0, 24.0],
    }

    @staticmethod
    def _interpolate(value: float, points: List[float]) -> float:
        """Interpolate between points based on value (0.0 to 1.0)."""
        if len(points) == 3:
            # [pessimistic, base, optimistic] - base is at 0.5
            if value <= 0.5:
                # Interpolate between pessimistic (0) and base (0.5)
                t = value / 0.5
                return points[0] + t * (points[1] - points[0])
            else:
                # Interpolate between base (0.5) and optimistic (1.0)
                t = (value - 0.5) / 0.5
                return points[1] + t * (points[2] - points[1])
        elif len(points) == 4:
            # [very_conservative, conservative, base, optimistic]
            # 0.0 = very_conservative, 0.2 = conservative, 0.5 = base, 1.0 = optimistic
            if value <= 0.2:
                t = value / 0.2
                return points[0] + t * (points[1] - points[0])
            elif value <= 0.5:
                t = (value - 0.2) / 0.3
                return points[1] + t * (points[2] - points[1])
            else:
                t = (value - 0.5) / 0.5
                return points[2] + t * (points[3] - points[2])
        return points[0]

    @classmethod
    def get_segments_with_assumptions(cls,
                                       segment_optimism: float = 0.5,
                                       cpa_optimism: float = 0.2,
                                       deeplink_optimism: float = 0.5) -> List['UserSegment']:
        """
        Generate segments based on optimism sliders.

        Args:
            segment_optimism: 0.0 (pessimistic) to 1.0 (optimistic) - controls user quality mix
            cpa_optimism: 0.0 (very conservative) to 1.0 (optimistic) - controls CPA conversion rates
            deeplink_optimism: 0.0 (pessimistic) to 1.0 (optimistic) - controls deeplink engagement

        Returns:
            List of UserSegment with interpolated values
        """
        # Get base segments as template
        base_segments = cls.get_default_segments()

        # Calculate interpolated percentages
        raw_percentages = {}
        for name in cls.SEGMENT_PERCENTAGES:
            raw_percentages[name] = cls._interpolate(
                segment_optimism,
                cls.SEGMENT_PERCENTAGES[name]
            )

        # Normalize percentages to sum to 1.0
        total = sum(raw_percentages.values())
        percentages = {k: v / total for k, v in raw_percentages.items()}

        # Build new segments with interpolated values
        new_segments = []
        for seg in base_segments:
            new_seg = UserSegment(
                name=seg.name,
                percentage=percentages[seg.name],
                ups_range=seg.ups_range,
                avg_ups=seg.avg_ups,
                monthly_revenue=seg.monthly_revenue,
                monthly_churn_rate=seg.monthly_churn_rate,
                linking_rate=seg.linking_rate,
                deeplink_rate=cls._interpolate(
                    deeplink_optimism,
                    cls.DEEPLINK_RATES[seg.name]
                ),
                conversion_rate=cls._interpolate(
                    cpa_optimism,
                    cls.CPA_RATES[seg.name]
                )
            )
            new_segments.append(new_seg)

        return new_segments

    @staticmethod
    def get_default_segments() -> List[UserSegment]:
        return [
            UserSegment(
                name="Unprofitable",
                percentage=0.40,
                ups_range=(-5.0, 0.0),
                avg_ups=-1.0,
                monthly_revenue=0.0,
                monthly_churn_rate=0.12,  # ~60% retain at month 3
                linking_rate=0.0,
                deeplink_rate=0.0,
                conversion_rate=0.0
            ),
            UserSegment(
                name="Low-Value",
                percentage=0.25,
                ups_range=(0.0, 3.0),
                avg_ups=1.5,
                monthly_revenue=1.5,
                monthly_churn_rate=0.08,
                linking_rate=0.20,
                deeplink_rate=1.0,
                conversion_rate=0.02
            ),
            UserSegment(
                name="Moderate",
                percentage=0.20,
                ups_range=(3.0, 8.0),
                avg_ups=5.0,
                monthly_revenue=5.0,
                monthly_churn_rate=0.05,
                linking_rate=0.50,
                deeplink_rate=3.0,
                conversion_rate=0.05
            ),
            UserSegment(
                name="High-Value",
                percentage=0.12,
                ups_range=(8.0, 20.0),
                avg_ups=12.0,
                monthly_revenue=12.0,
                monthly_churn_rate=0.03,
                linking_rate=0.80,
                deeplink_rate=6.0,
                conversion_rate=0.10
            ),
            UserSegment(
                name="Power",
                percentage=0.03,
                ups_range=(20.0, 100.0),
                avg_ups=25.0,
                monthly_revenue=25.0,
                monthly_churn_rate=0.02,
                linking_rate=0.95,
                deeplink_rate=12.0,
                conversion_rate=0.15
            ),
        ]


class TokenIssuanceEngine:
    """Calculates token issuance rates based on UPS."""

    def __init__(self, params: SystemParameters):
        self.params = params

    def calculate_effective_ups(self, ups_observed: float, data_points: int) -> float:
        """Calculate confidence-weighted UPS."""
        alpha = data_points / (data_points + self.params.confidence_k)
        return (alpha * ups_observed) + ((1 - alpha) * self.params.prior_ups)

    def calculate_tokens_per_day(self, ups_effective: float) -> int:
        """
        Calculate daily token issuance using piecewise function:
        - FLOOR if UPS <= 0
        - Linear ramp from FLOOR to BASE if 0 < UPS < theta
        - Logarithmic growth if UPS >= theta
        """
        if ups_effective <= 0:
            tokens = self.params.floor_tokens
        elif ups_effective < self.params.theta:
            # Linear ramp from FLOOR to BASE
            tokens = self.params.floor_tokens + \
                     (ups_effective / self.params.theta) * \
                     (self.params.base_tokens - self.params.floor_tokens)
        else:
            # Logarithmic growth above threshold
            tokens = self.params.base_tokens + \
                     self.params.beta * math.log(1 + ups_effective - self.params.theta)

        return min(int(tokens), self.params.cap_tokens)

    def calculate_new_user_tokens(self, days_since_signup: int) -> int:
        """Calculate token rate for new users (first 21 days)."""
        if days_since_signup < 7:
            return self.params.new_user_week1_tokens
        elif days_since_signup < 14:
            return self.params.new_user_week2_tokens
        elif days_since_signup < 21:
            return self.params.new_user_week3_tokens
        else:
            return None  # Use normal UPS-based calculation


class CostCalculator:
    """Calculates various costs in the system."""

    def __init__(self, params: SystemParameters):
        self.params = params

    def tokens_to_prize_cost(self, tokens_per_day: int, active_days: int = 23) -> float:
        """
        Convert tokens to prize cost.

        Formula: tokens/day * redemption_rate * (1/coins_per_dollar) * active_days * (1 - discount)
        """
        monthly_tokens = tokens_per_day * active_days
        monthly_coins = monthly_tokens * self.params.tokens_per_coin
        redeemed_coins = monthly_coins * self.params.coin_redemption_rate
        prize_cost = redeemed_coins / self.params.coins_per_dollar
        discounted_cost = prize_cost * (1 - self.params.prize_cost_discount)
        return discounted_cost

    def calculate_monthly_user_cost(self,
                                    tokens_per_day: int,
                                    is_linked: bool = False,
                                    active_days: int = 23) -> Dict[str, float]:
        """Calculate total monthly cost for a user."""
        prize_cost = self.tokens_to_prize_cost(tokens_per_day, active_days)
        linking_cost = self.params.linking_cost_per_month if is_linked else 0
        cloud_cost = self.params.cloud_cost_per_mau

        return {
            'prize_cost': prize_cost,
            'linking_cost': linking_cost,
            'cloud_cost': cloud_cost,
            'total': prize_cost + linking_cost + cloud_cost
        }

    def calculate_new_user_investment(self) -> Dict[str, float]:
        """Calculate the total investment in a new user over 21 days."""
        week1_tokens = self.params.new_user_week1_tokens * 7
        week2_tokens = self.params.new_user_week2_tokens * 7
        week3_tokens = self.params.new_user_week3_tokens * 7

        total_tokens = week1_tokens + week2_tokens + week3_tokens

        # Calculate prize cost for new user tokens
        week1_cost = self.tokens_to_prize_cost(self.params.new_user_week1_tokens, 7)
        week2_cost = self.tokens_to_prize_cost(self.params.new_user_week2_tokens, 7)
        week3_cost = self.tokens_to_prize_cost(self.params.new_user_week3_tokens, 7)

        total_cost = week1_cost + week2_cost + week3_cost

        return {
            'total_tokens': total_tokens,
            'week1_cost': week1_cost,
            'week2_cost': week2_cost,
            'week3_cost': week3_cost,
            'total_cost': total_cost,
            'kyc_cost': self.params.kyc_cost_per_user,
            'total_investment': total_cost + self.params.kyc_cost_per_user
        }


class RevenueCalculator:
    """Calculates revenue streams."""

    def __init__(self, params: SystemParameters):
        self.params = params

    def calculate_acquisition_revenue(self, conversion_rate: float,
                                       num_sportsbooks: int = 3) -> float:
        """Calculate expected acquisition revenue from CPA conversions."""
        return conversion_rate * self.params.cpa_average * num_sportsbooks

    def calculate_retention_revenue(self, monthly_bets: float) -> float:
        """Calculate monthly retention revenue from deeplink bets."""
        return monthly_bets * self.params.retention_rev_per_bet

    def calculate_segment_revenue(self, segment: UserSegment) -> Dict[str, float]:
        """Calculate revenue breakdown for a user segment."""
        # Amortize acquisition revenue over 12 months
        acq_monthly = self.calculate_acquisition_revenue(segment.conversion_rate) / 12
        ret_monthly = self.calculate_retention_revenue(segment.deeplink_rate)

        return {
            'acquisition_monthly': acq_monthly,
            'retention_monthly': ret_monthly,
            'total_monthly': acq_monthly + ret_monthly
        }


class PortfolioAnalyzer:
    """Analyzes portfolio-level economics."""

    def __init__(self, params: SystemParameters):
        self.params = params
        self.token_engine = TokenIssuanceEngine(params)
        self.cost_calc = CostCalculator(params)
        self.revenue_calc = RevenueCalculator(params)

    def analyze_segment(self, segment: UserSegment, mau: int = 1000) -> Dict:
        """Analyze economics for a single segment."""
        segment_users = int(mau * segment.percentage)

        # Calculate tokens per day based on average UPS
        tokens_per_day = self.token_engine.calculate_tokens_per_day(segment.avg_ups)

        # Calculate costs
        linked_users = int(segment_users * segment.linking_rate)
        unlinked_users = segment_users - linked_users

        linked_costs = self.cost_calc.calculate_monthly_user_cost(
            tokens_per_day, is_linked=True
        )
        unlinked_costs = self.cost_calc.calculate_monthly_user_cost(
            tokens_per_day, is_linked=False
        )

        total_cost = (linked_costs['total'] * linked_users +
                     unlinked_costs['total'] * unlinked_users)

        # Calculate revenue
        total_revenue = segment.monthly_revenue * segment_users

        # Net margin
        net = total_revenue - total_cost
        margin = (net / total_revenue * 100) if total_revenue > 0 else float('-inf')

        return {
            'segment': segment.name,
            'users': segment_users,
            'tokens_per_day': tokens_per_day,
            'prize_cost_per_user': linked_costs['prize_cost'],
            'total_cost': total_cost,
            'revenue': total_revenue,
            'net': net,
            'margin_pct': margin
        }

    def analyze_portfolio(self, segments: List[UserSegment],
                         mau: int = 1000) -> Dict:
        """Analyze full portfolio economics."""
        segment_results = []
        total_cost = 0
        total_revenue = 0
        total_users = 0

        for segment in segments:
            result = self.analyze_segment(segment, mau)
            segment_results.append(result)
            total_cost += result['total_cost']
            total_revenue += result['revenue']
            total_users += result['users']

        net = total_revenue - total_cost
        margin = (net / total_revenue * 100) if total_revenue > 0 else 0

        return {
            'segments': segment_results,
            'total_mau': total_users,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'net': net,
            'blended_margin_pct': margin,
            'cost_per_mau': total_cost / total_users if total_users > 0 else 0,
            'revenue_per_mau': total_revenue / total_users if total_users > 0 else 0,
            'arpu': total_revenue / total_users if total_users > 0 else 0
        }

    def calculate_ltv(self, segment: UserSegment,
                     months: int = 12) -> Dict[str, float]:
        """Calculate lifetime value for a segment over specified months."""
        tokens_per_day = self.token_engine.calculate_tokens_per_day(segment.avg_ups)
        monthly_cost = self.cost_calc.calculate_monthly_user_cost(
            tokens_per_day,
            is_linked=(segment.linking_rate > 0.5)
        )['total']

        # Simulate cohort over time with churn
        retention_curve = []
        cumulative_revenue = 0
        cumulative_cost = 0

        retained = 1.0
        for month in range(1, months + 1):
            retained *= (1 - segment.monthly_churn_rate)
            retention_curve.append(retained)

            monthly_rev = segment.monthly_revenue * retained
            monthly_cost_adj = monthly_cost * retained

            cumulative_revenue += monthly_rev
            cumulative_cost += monthly_cost_adj

        ltv = cumulative_revenue - cumulative_cost

        return {
            'segment': segment.name,
            'months': months,
            'final_retention': retained,
            'cumulative_revenue': cumulative_revenue,
            'cumulative_cost': cumulative_cost,
            'ltv': ltv,
            'retention_curve': retention_curve
        }

    def calculate_blended_ltv(self, segments: List[UserSegment],
                              months: int = 12) -> float:
        """Calculate blended LTV across all segments."""
        total_ltv = 0
        for segment in segments:
            segment_ltv = self.calculate_ltv(segment, months)
            total_ltv += segment_ltv['ltv'] * segment.percentage
        return total_ltv

    def calculate_breakeven(self, segments: List[UserSegment]) -> Dict:
        """Calculate break-even metrics."""
        new_user_investment = self.cost_calc.calculate_new_user_investment()
        blended_ltv = self.calculate_blended_ltv(segments, months=12)

        # Total CAC = prize investment + KYC + marketing acquisition cost
        total_cac = new_user_investment['total_investment'] + self.params.user_acquisition_cost

        # Calculate what % profitable users needed for break-even
        profitable_segments = [s for s in segments if s.avg_ups > 0]
        unprofitable_segments = [s for s in segments if s.avg_ups <= 0]

        profitable_contribution = sum(
            self.calculate_ltv(s)['ltv'] * s.percentage
            for s in profitable_segments
        )
        unprofitable_drain = sum(
            abs(self.calculate_ltv(s)['ltv']) * s.percentage
            for s in unprofitable_segments
        )

        return {
            'new_user_investment': total_cac,  # Now includes marketing CAC
            'prize_investment': new_user_investment['total_investment'],
            'marketing_cac': self.params.user_acquisition_cost,
            'blended_ltv': blended_ltv,
            'ltv_to_cac_ratio': blended_ltv / total_cac if total_cac > 0 else 0,
            'profitable_contribution': profitable_contribution,
            'unprofitable_drain': unprofitable_drain,
            'net_per_user': blended_ltv - total_cac
        }


class ProjectionEngine:
    """Projects financial metrics over time."""

    def __init__(self, params: SystemParameters):
        self.params = params
        self.portfolio_analyzer = PortfolioAnalyzer(params)

    def project_monthly(self,
                        segments: List[UserSegment],
                        initial_mau: int,
                        months: int = 12,
                        growth_rate: Optional[float] = None) -> List[Dict]:
        """Project monthly financials."""
        if growth_rate is None:
            growth_rate = self.params.monthly_growth_rate

        projections = []
        current_mau = initial_mau

        for month in range(1, months + 1):
            portfolio = self.portfolio_analyzer.analyze_portfolio(segments, int(current_mau))

            projections.append({
                'month': month,
                'mau': int(current_mau),
                'revenue': portfolio['total_revenue'],
                'cost': portfolio['total_cost'],
                'net': portfolio['net'],
                'margin_pct': portfolio['blended_margin_pct'],
                'arpu': portfolio['arpu']
            })

            current_mau *= (1 + growth_rate)

        return projections

    def project_annual(self,
                       segments: List[UserSegment],
                       initial_mau: int,
                       years: int = 3,
                       growth_rate: Optional[float] = None) -> List[Dict]:
        """Project annual financials."""
        monthly = self.project_monthly(segments, initial_mau, years * 12, growth_rate)

        annual = []
        for year in range(1, years + 1):
            year_months = monthly[(year-1)*12 : year*12]

            annual.append({
                'year': year,
                'avg_mau': sum(m['mau'] for m in year_months) / 12,
                'end_mau': year_months[-1]['mau'],
                'total_revenue': sum(m['revenue'] for m in year_months),
                'total_cost': sum(m['cost'] for m in year_months),
                'total_net': sum(m['net'] for m in year_months),
                'avg_margin_pct': sum(m['margin_pct'] for m in year_months) / 12
            })

        return annual
