"""
HotTakes Profitability Maximization Engine - Interactive Dashboard

A comprehensive sensitivity analysis tool for exploring the financial
health of the HotTakes token/prize economy.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.financial_model import (
    SystemParameters,
    UserSegment,
    UserSegmentDistribution,
    PortfolioAnalyzer,
    ProjectionEngine,
    TokenIssuanceEngine,
    CostCalculator,
    RevenueCalculator
)
from analysis.sensitivity import (
    SensitivityAnalyzer,
    MonteCarloSimulator,
    SegmentSensitivityAnalyzer,
    ParameterRegistry,
    ParameterRange
)

# Page configuration
st.set_page_config(
    page_title="HotTakes Profitability Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive { color: #00ff88; }
    .negative { color: #ff4444; }
    .neutral { color: #ffaa00; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def create_sidebar():
    """Create the parameter control sidebar."""
    advanced_mode = st.session_state.get('advanced_mode', False)

    if advanced_mode:
        st.sidebar.title("🎛️ System Parameters")
    else:
        st.sidebar.title("⚙️ Settings")
        st.sidebar.caption("Adjust assumptions to see how they affect profitability")

    # Initialize session state for parameters
    if 'params' not in st.session_state:
        st.session_state.params = SystemParameters()
    if 'mau' not in st.session_state:
        st.session_state.mau = 24000

    params = st.session_state.params
    current_mau = st.session_state.mau

    # Revenue Assumptions Section - Always visible with mode-appropriate labels
    st.sidebar.markdown("### 📊 Revenue Assumptions")
    if not advanced_mode:
        st.sidebar.caption("These control our key revenue projections")

    # User Quality Mix slider with mode-appropriate help
    if advanced_mode:
        params.segment_optimism = st.sidebar.slider(
            "User Quality Mix",
            min_value=0.0, max_value=1.0, value=params.segment_optimism, step=0.1,
            help="Controls user segment distribution. Pessimistic (0) = more floor/unprofitable users. Optimistic (1) = higher quality user mix with better conversion into profitable segments."
        )
    else:
        params.segment_optimism = st.sidebar.slider(
            "User Quality",
            min_value=0.0, max_value=1.0, value=params.segment_optimism, step=0.1,
            help="What percentage of users will become paying/profitable vs staying on free tier?"
        )
    # Display interpretation
    if params.segment_optimism < 0.3:
        st.sidebar.caption("📉 Pessimistic: ~55% free-tier users")
    elif params.segment_optimism > 0.7:
        st.sidebar.caption("📈 Optimistic: ~25% free-tier users")
    else:
        st.sidebar.caption("⚖️ Balanced: ~40% free-tier users")

    # CPA Conversion slider
    if advanced_mode:
        params.cpa_optimism = st.sidebar.slider(
            "CPA Conversion Rate",
            min_value=0.0, max_value=1.0, value=params.cpa_optimism, step=0.1,
            help="Controls CPA conversion success rate per segment. Very Conservative (0) = only power users occasionally convert. We recommend conservative assumptions (0.2) until validated by real data."
        )
    else:
        params.cpa_optimism = st.sidebar.slider(
            "Sportsbook Signups",
            min_value=0.0, max_value=1.0, value=params.cpa_optimism, step=0.1,
            help="How often do users sign up for partner sportsbooks? This generates one-time CPA revenue (~$150 each)."
        )
    # Calculate and display estimated CPAs
    temp_segments = UserSegmentDistribution.get_segments_with_assumptions(
        params.segment_optimism, params.cpa_optimism, params.deeplink_optimism
    )
    est_cpas = sum(seg.percentage * seg.conversion_rate * current_mau for seg in temp_segments)
    if params.cpa_optimism < 0.2:
        st.sidebar.caption(f"🔒 Very Conservative: minimal signups")
    elif params.cpa_optimism < 0.4:
        st.sidebar.caption(f"🔒 Conservative: ~{est_cpas:.0f} signups/mo")
    else:
        st.sidebar.caption(f"⚠️ Optimistic: ~{est_cpas:.0f} signups/mo")

    # Deeplink Engagement slider
    if advanced_mode:
        params.deeplink_optimism = st.sidebar.slider(
            "Deeplink Engagement",
            min_value=0.0, max_value=1.0, value=params.deeplink_optimism, step=0.1,
            help="Controls how frequently users engage with retention bets (deeplinks). Pessimistic (0) = low adoption. Optimistic (1) = strong engagement across all profitable segments."
        )
    else:
        params.deeplink_optimism = st.sidebar.slider(
            "Bet Link Usage",
            min_value=0.0, max_value=1.0, value=params.deeplink_optimism, step=0.1,
            help="How often do users place bets through our links? This generates ongoing revenue ($0.50 per bet)."
        )
    # Calculate estimated deeplink revenue
    est_bets = sum(seg.percentage * seg.deeplink_rate * current_mau for seg in temp_segments)
    est_deeplink_rev = est_bets * params.retention_rev_per_bet
    if params.deeplink_optimism < 0.3:
        st.sidebar.caption(f"📉 Low usage: ~${est_deeplink_rev:,.0f}/mo")
    elif params.deeplink_optimism > 0.7:
        st.sidebar.caption(f"📈 High usage: ~${est_deeplink_rev:,.0f}/mo")
    else:
        st.sidebar.caption(f"⚖️ Moderate: ~${est_deeplink_rev:,.0f}/mo")

    st.sidebar.divider()

    # Basic Settings section - visible in both modes but simplified
    if advanced_mode:
        # Advanced mode: All parameter expanders

        with st.sidebar.expander("💰 Token Issuance", expanded=False):
            params.floor_tokens = st.slider(
                "Floor Tokens/Day",
                min_value=20, max_value=100, value=params.floor_tokens,
                help="Minimum tokens for unprofitable users"
            )
            params.base_tokens = st.slider(
                "Base Tokens/Day",
                min_value=80, max_value=200, value=params.base_tokens,
                help="Standard engaged user rate"
            )
            params.theta = st.slider(
                "Profitability Threshold (θ)",
                min_value=1.0, max_value=8.0, value=params.theta, step=0.5,
                help="UPS threshold for 'profitable' classification ($/month)"
            )
            params.beta = st.slider(
                "Log Growth Rate (β)",
                min_value=10.0, max_value=60.0, value=params.beta, step=5.0,
                help="Logarithmic growth rate for high-value users"
            )
            params.cap_tokens = st.slider(
                "Token Cap/Day",
                min_value=200, max_value=600, value=params.cap_tokens,
                help="Maximum daily token issuance"
            )

        with st.sidebar.expander("🪙 Token Economics"):
            params.coins_per_dollar = st.slider(
                "Coins per Dollar",
                min_value=500, max_value=2000, value=int(params.coins_per_dollar), step=100,
                help="Exchange rate for prize redemption"
            )
            params.coin_redemption_rate = st.slider(
                "Coin Redemption Rate",
                min_value=0.50, max_value=1.0, value=params.coin_redemption_rate, step=0.05,
                help="Percentage of coins that get redeemed"
            )
            params.prize_cost_discount = st.slider(
                "Prize Cost Discount",
                min_value=0.0, max_value=0.30, value=params.prize_cost_discount, step=0.05,
                help="Bulk discount on prize costs"
            )

        with st.sidebar.expander("💵 User Costs"):
            params.user_acquisition_cost = st.slider(
                "User Acquisition Cost (CAC)",
                min_value=0.0, max_value=30.0, value=params.user_acquisition_cost, step=1.0,
                help="Marketing cost to acquire each new user (ads, referrals, etc.)"
            )
            params.kyc_cost_per_user = st.slider(
                "KYC Cost per User",
                min_value=0.0, max_value=3.0, value=params.kyc_cost_per_user, step=0.10,
                help="One-time identity verification cost"
            )
            params.linking_cost_per_month = st.slider(
                "Account Linking Cost/Month",
                min_value=0.30, max_value=2.0, value=params.linking_cost_per_month, step=0.10,
                help="Monthly cost for sportsbook account linking"
            )
            params.cloud_cost_per_mau = st.slider(
                "Cloud Cost per MAU",
                min_value=0.05, max_value=0.50, value=params.cloud_cost_per_mau, step=0.05,
                help="Monthly infrastructure cost per active user"
            )

        with st.sidebar.expander("📈 Revenue"):
            params.cpa_average = st.slider(
                "Average CPA",
                min_value=50.0, max_value=300.0, value=params.cpa_average, step=10.0,
                help="Average acquisition revenue per sportsbook conversion"
            )
            params.retention_rev_per_bet = st.slider(
                "Retention Rev per Bet",
                min_value=0.20, max_value=1.50, value=params.retention_rev_per_bet, step=0.10,
                help="Revenue per deeplink bet"
            )

        with st.sidebar.expander("🆕 New User Economics"):
            params.new_user_week1_tokens = st.slider(
                "Week 1 Tokens/Day",
                min_value=100, max_value=350, value=params.new_user_week1_tokens,
                help="Daily tokens for new users in week 1"
            )
            params.new_user_week2_tokens = st.slider(
                "Week 2 Tokens/Day",
                min_value=100, max_value=300, value=params.new_user_week2_tokens,
                help="Daily tokens for new users in week 2"
            )
            params.new_user_week3_tokens = st.slider(
                "Week 3 Tokens/Day",
                min_value=80, max_value=250, value=params.new_user_week3_tokens,
                help="Daily tokens for new users in week 3"
            )

        with st.sidebar.expander("📊 Growth & Scale"):
            mau = st.number_input(
                "Monthly Active Users (MAU)",
                min_value=1000, max_value=1000000, value=st.session_state.mau, step=1000,
                help="Current or target MAU for analysis"
            )
            st.session_state.mau = mau
            params.monthly_growth_rate = st.slider(
                "Monthly Growth Rate",
                min_value=-0.10, max_value=0.30, value=params.monthly_growth_rate, step=0.01,
                format="%.0f%%",
                help="Expected monthly MAU growth rate"
            )

    else:
        # Simple mode: Minimal settings in one collapsed section
        with st.sidebar.expander("⚙️ Basic Settings", expanded=False):
            mau = st.number_input(
                "Monthly Active Users",
                min_value=1000, max_value=1000000, value=st.session_state.mau, step=1000,
                help="How many active users do we have?"
            )
            st.session_state.mau = mau

            params.cpa_average = st.slider(
                "Revenue per Signup",
                min_value=50.0, max_value=300.0, value=params.cpa_average, step=10.0,
                help="How much we earn when a user signs up for a sportsbook"
            )
            params.retention_rev_per_bet = st.slider(
                "Revenue per Bet",
                min_value=0.20, max_value=1.50, value=params.retention_rev_per_bet, step=0.10,
                help="How much we earn per bet placed through our links"
            )
            params.user_acquisition_cost = st.slider(
                "Cost to Acquire User",
                min_value=0.0, max_value=30.0, value=params.user_acquisition_cost, step=1.0,
                help="Marketing cost to acquire each new user"
            )

    # Update derived values
    params.update_derived()

    # Reset button
    if st.sidebar.button("🔄 Reset to Defaults"):
        st.session_state.params = SystemParameters()
        st.session_state.mau = 24000
        st.rerun()

    return params, st.session_state.mau


def render_overview_tab(params: SystemParameters, mau: int):
    """Render the main overview dashboard."""
    st.header("📊 Financial Overview")

    # Get segments using optimism parameters
    segments = UserSegmentDistribution.get_segments_with_assumptions(
        params.segment_optimism,
        params.cpa_optimism,
        params.deeplink_optimism
    )
    analyzer = PortfolioAnalyzer(params)
    portfolio = analyzer.analyze_portfolio(segments, mau)
    breakeven = analyzer.calculate_breakeven(segments)
    cost_calc = CostCalculator(params)
    revenue_calc = RevenueCalculator(params)
    new_user_inv = cost_calc.calculate_new_user_investment()

    # Calculate revenue breakdown by source
    total_cpa_revenue = 0
    total_deeplink_revenue = 0
    total_deeplink_bets = 0
    total_cpa_conversions = 0
    for seg in segments:
        seg_users = mau * seg.percentage
        # CPA revenue (amortized monthly)
        cpa_conversions = seg_users * seg.conversion_rate
        cpa_rev = revenue_calc.calculate_acquisition_revenue(seg.conversion_rate) * seg_users / 12
        total_cpa_revenue += cpa_rev
        total_cpa_conversions += cpa_conversions
        # Deeplink revenue
        deeplink_bets = seg_users * seg.deeplink_rate
        deeplink_rev = deeplink_bets * params.retention_rev_per_bet
        total_deeplink_revenue += deeplink_rev
        total_deeplink_bets += deeplink_bets

    total_revenue = total_cpa_revenue + total_deeplink_revenue
    cpa_pct = (total_cpa_revenue / total_revenue * 100) if total_revenue > 0 else 0
    deeplink_pct = (total_deeplink_revenue / total_revenue * 100) if total_revenue > 0 else 0

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        margin_color = "normal" if portfolio['blended_margin_pct'] >= 35 else "inverse"
        st.metric(
            "Blended Margin",
            f"{portfolio['blended_margin_pct']:.1f}%",
            delta=f"{portfolio['blended_margin_pct'] - 35:.1f}% vs target",
            delta_color=margin_color
        )

    with col2:
        st.metric(
            "Monthly Revenue",
            f"${portfolio['total_revenue']:,.0f}",
            delta=f"${portfolio['revenue_per_mau']:.2f}/MAU"
        )

    with col3:
        st.metric(
            "Monthly Cost",
            f"${portfolio['total_cost']:,.0f}",
            delta=f"${portfolio['cost_per_mau']:.2f}/MAU",
            delta_color="inverse"
        )

    with col4:
        net_color = "normal" if portfolio['net'] > 0 else "inverse"
        st.metric(
            "Monthly Net",
            f"${portfolio['net']:,.0f}",
            delta_color=net_color
        )

    with col5:
        ltv_color = "normal" if breakeven['ltv_to_cac_ratio'] >= 3 else "inverse"
        st.metric(
            "LTV:CAC Ratio",
            f"{breakeven['ltv_to_cac_ratio']:.2f}x",
            delta=f"Target: 3.0x",
            delta_color=ltv_color
        )

    st.divider()

    # Revenue Sources Breakdown
    st.subheader("💵 Revenue Sources")
    st.caption("Where revenue comes from - CPA is one-time sportsbook signup commissions, Deeplinks are ongoing retention bet commissions")

    rev_col1, rev_col2, rev_col3 = st.columns(3)

    with rev_col1:
        st.metric(
            "CPA Revenue",
            f"${total_cpa_revenue:,.0f}/mo",
            delta=f"{cpa_pct:.0f}% of total"
        )
        st.caption(f"~{total_cpa_conversions:.0f} conversions @ ${params.cpa_average:.0f} avg (amortized)")

    with rev_col2:
        st.metric(
            "Deeplink Revenue",
            f"${total_deeplink_revenue:,.0f}/mo",
            delta=f"{deeplink_pct:.0f}% of total"
        )
        st.caption(f"~{total_deeplink_bets:.0f} bets/mo @ ${params.retention_rev_per_bet:.2f}")

    with rev_col3:
        st.metric(
            "Total Revenue",
            f"${total_revenue:,.0f}/mo",
            delta=f"${total_revenue/mau:.2f}/MAU"
        )

    st.divider()

    # Two column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Token Issuance Curve")
        # Generate issuance curve data
        token_engine = TokenIssuanceEngine(params)
        ups_values = np.linspace(-3, 30, 100)
        tokens = [token_engine.calculate_tokens_per_day(ups) for ups in ups_values]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ups_values, y=tokens,
            mode='lines',
            name='Token Rate',
            line=dict(color='#00ff88', width=3)
        ))

        # Add reference lines
        fig.add_hline(y=params.floor_tokens, line_dash="dash",
                      annotation_text="Floor", line_color="red")
        fig.add_hline(y=params.base_tokens, line_dash="dash",
                      annotation_text="Base", line_color="yellow")
        fig.add_vline(x=params.theta, line_dash="dash",
                      annotation_text="θ threshold", line_color="cyan")

        fig.update_layout(
            xaxis_title="User Profitability Score ($/month)",
            yaxis_title="Tokens per Day",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("👥 User Segment Distribution")
        # Segment pie chart
        segment_data = pd.DataFrame([
            {'Segment': s['segment'], 'Users': s['users'], 'Net': s['net']}
            for s in portfolio['segments']
        ])

        fig = px.pie(
            segment_data,
            values='Users',
            names='Segment',
            color='Segment',
            color_discrete_map={
                'Unprofitable': '#ff4444',
                'Low-Value': '#ffaa00',
                'Moderate': '#00aaff',
                'High-Value': '#00ff88',
                'Power': '#aa00ff'
            }
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, width="stretch")

    # Segment economics table
    st.subheader("💰 Segment Economics")
    segment_df = pd.DataFrame(portfolio['segments'])
    segment_df['margin_pct'] = segment_df['margin_pct'].apply(
        lambda x: f"{x:.1f}%" if x != float('-inf') else "N/A"
    )
    segment_df['prize_cost_per_user'] = segment_df['prize_cost_per_user'].apply(
        lambda x: f"${x:.2f}"
    )
    segment_df['total_cost'] = segment_df['total_cost'].apply(
        lambda x: f"${x:,.0f}"
    )
    segment_df['revenue'] = segment_df['revenue'].apply(
        lambda x: f"${x:,.0f}"
    )
    segment_df['net'] = segment_df['net'].apply(
        lambda x: f"${x:,.0f}"
    )

    st.dataframe(
        segment_df[['segment', 'users', 'tokens_per_day', 'prize_cost_per_user',
                    'revenue', 'total_cost', 'net', 'margin_pct']],
        width="stretch",
        hide_index=True
    )

    # New user economics
    st.subheader("🆕 New User Investment")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Token Investment", f"{new_user_inv['total_tokens']:,} tokens")
        st.metric("Week 1 Cost", f"${new_user_inv['week1_cost']:.2f}")

    with col2:
        st.metric("Prize Cost (21 days)", f"${new_user_inv['total_cost']:.2f}")
        st.metric("Week 2 Cost", f"${new_user_inv['week2_cost']:.2f}")

    with col3:
        st.metric("Total Investment/User", f"${new_user_inv['total_investment']:.2f}")
        st.metric("Week 3 Cost", f"${new_user_inv['week3_cost']:.2f}")


def render_sensitivity_tab(params: SystemParameters, mau: int):
    """Render the sensitivity analysis tab."""
    st.header("🔬 Sensitivity Analysis")

    segments = UserSegmentDistribution.get_segments_with_assumptions(
        params.segment_optimism, params.cpa_optimism, params.deeplink_optimism
    )
    analyzer = SensitivityAnalyzer(params)
    analyzer.set_segments(segments)

    analysis_type = st.radio(
        "Analysis Type",
        ["Single Parameter", "Tornado Chart", "Two-Way Analysis"],
        horizontal=True
    )

    if analysis_type == "Single Parameter":
        st.subheader("📊 Single Parameter Sensitivity")

        col1, col2 = st.columns([1, 3])

        with col1:
            # Get all parameters grouped by category
            all_params = ParameterRegistry.get_all_parameters(params)
            categories = list(set(p.category for p in all_params))

            selected_category = st.selectbox("Category", sorted(categories))
            category_params = [p for p in all_params if p.category == selected_category]

            selected_param_name = st.selectbox(
                "Parameter",
                [p.display_name for p in category_params]
            )
            selected_param = next(p for p in category_params if p.display_name == selected_param_name)

            target_metric = st.selectbox(
                "Target Metric",
                ["blended_margin_pct", "net_monthly", "revenue_per_mau",
                 "cost_per_mau", "ltv", "ltv_to_cac"]
            )

        with col2:
            # Run sensitivity analysis
            results = analyzer.single_parameter_sensitivity(selected_param, mau)

            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['parameter_value'],
                y=results[target_metric],
                mode='lines+markers',
                name=target_metric,
                line=dict(color='#00ff88', width=3)
            ))

            # Add current value marker
            current_val = getattr(params, selected_param.name)
            closest_idx = (results['parameter_value'] - current_val).abs().idxmin()
            current_metric = results.loc[closest_idx, target_metric]

            fig.add_trace(go.Scatter(
                x=[current_val],
                y=[current_metric],
                mode='markers',
                name='Current',
                marker=dict(color='red', size=15, symbol='star')
            ))

            fig.update_layout(
                xaxis_title=selected_param.display_name,
                yaxis_title=target_metric.replace('_', ' ').title(),
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, width="stretch")

    elif analysis_type == "Tornado Chart":
        st.subheader("🌪️ Tornado Analysis")

        col1, col2 = st.columns([1, 4])

        with col1:
            target_metric = st.selectbox(
                "Target Metric",
                ["blended_margin_pct", "net_monthly", "ltv", "ltv_to_cac"],
                key="tornado_metric"
            )
            swing_pct = st.slider(
                "Swing %",
                min_value=10, max_value=50, value=20,
                help="Percentage variation from base value"
            ) / 100

        with col2:
            # Run tornado analysis
            tornado_df = analyzer.tornado_analysis(
                target_metric=target_metric,
                swing_pct=swing_pct,
                mau=mau
            )

            # Take top 15 most impactful parameters
            tornado_df = tornado_df.head(15)

            # Create tornado chart
            fig = go.Figure()

            # Sort by impact for visual
            tornado_df = tornado_df.sort_values('impact_range')

            fig.add_trace(go.Bar(
                y=tornado_df['parameter'],
                x=tornado_df['low_delta'],
                name='Low Value',
                orientation='h',
                marker_color='#ff4444'
            ))

            fig.add_trace(go.Bar(
                y=tornado_df['parameter'],
                x=tornado_df['high_delta'],
                name='High Value',
                orientation='h',
                marker_color='#00ff88'
            ))

            fig.update_layout(
                barmode='overlay',
                xaxis_title=f"Change in {target_metric.replace('_', ' ').title()}",
                template="plotly_dark",
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig, width="stretch")

    else:  # Two-Way Analysis
        st.subheader("🔥 Two-Way Sensitivity (Heat Map)")

        col1, col2, col3 = st.columns(3)

        all_params = ParameterRegistry.get_all_parameters(params)

        with col1:
            param1_name = st.selectbox(
                "Parameter 1 (X-axis)",
                [p.display_name for p in all_params],
                index=0
            )
            param1 = next(p for p in all_params if p.display_name == param1_name)

        with col2:
            param2_name = st.selectbox(
                "Parameter 2 (Y-axis)",
                [p.display_name for p in all_params],
                index=1
            )
            param2 = next(p for p in all_params if p.display_name == param2_name)

        with col3:
            target_metric = st.selectbox(
                "Target Metric",
                ["blended_margin_pct", "net_monthly", "ltv"],
                key="twoway_metric"
            )

        if param1.name != param2.name:
            # Run two-way analysis
            results = analyzer.two_way_sensitivity(
                param1, param2, target_metric, mau, resolution=12
            )

            # Pivot for heatmap
            pivot = results.pivot(
                index=param2.display_name,
                columns=param1.display_name,
                values=target_metric
            )

            fig = px.imshow(
                pivot,
                labels=dict(x=param1.display_name, y=param2.display_name,
                           color=target_metric),
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Please select two different parameters.")


def render_monte_carlo_tab(params: SystemParameters, mau: int):
    """Render the Monte Carlo simulation tab."""
    st.header("🎲 Monte Carlo Simulation")

    segments = UserSegmentDistribution.get_segments_with_assumptions(
        params.segment_optimism, params.cpa_optimism, params.deeplink_optimism
    )
    simulator = MonteCarloSimulator(params)
    simulator.set_segments(segments)

    col1, col2 = st.columns([1, 3])

    with col1:
        n_simulations = st.slider(
            "Number of Simulations",
            min_value=100, max_value=5000, value=1000, step=100
        )
        projection_months = st.slider(
            "Projection Months",
            min_value=6, max_value=36, value=12
        )

        st.markdown("### Parameter Distributions")
        st.caption("Using default triangular/normal distributions for key parameters")

        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                results = simulator.run_simulation(
                    n_simulations=n_simulations,
                    mau=mau,
                    projection_months=projection_months
                )
                st.session_state.mc_results = results
                st.session_state.mc_analysis = simulator.analyze_results(results)

    with col2:
        if 'mc_results' in st.session_state:
            results = st.session_state.mc_results
            analysis = st.session_state.mc_analysis

            # Summary metrics
            st.subheader("📊 Simulation Results")

            metric_cols = st.columns(4)

            with metric_cols[0]:
                st.metric(
                    "Expected Margin",
                    f"{analysis['blended_margin_pct']['mean']:.1f}%",
                    delta=f"±{analysis['blended_margin_pct']['std']:.1f}%"
                )

            with metric_cols[1]:
                st.metric(
                    "Expected Monthly Net",
                    f"${analysis['monthly_net']['mean']:,.0f}",
                    delta=f"±${analysis['monthly_net']['std']:,.0f}"
                )

            with metric_cols[2]:
                st.metric(
                    f"Expected {projection_months}M Net",
                    f"${analysis['total_net_12m']['mean']:,.0f}"
                )

            with metric_cols[3]:
                prob_positive = analysis['monthly_net']['prob_positive'] * 100
                st.metric(
                    "P(Profitable)",
                    f"{prob_positive:.1f}%"
                )

            # Distribution charts
            tab1, tab2, tab3 = st.tabs(["Margin Distribution", "Net Distribution", "Percentiles"])

            with tab1:
                fig = px.histogram(
                    results,
                    x='blended_margin_pct',
                    nbins=50,
                    title="Blended Margin Distribution"
                )
                fig.add_vline(x=35, line_dash="dash", line_color="red",
                             annotation_text="Target (35%)")
                fig.add_vline(x=analysis['blended_margin_pct']['mean'],
                             line_dash="solid", line_color="green",
                             annotation_text=f"Mean ({analysis['blended_margin_pct']['mean']:.1f}%)")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, width="stretch")

            with tab2:
                fig = px.histogram(
                    results,
                    x='monthly_net',
                    nbins=50,
                    title="Monthly Net Profit Distribution"
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red",
                             annotation_text="Breakeven")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, width="stretch")

            with tab3:
                percentile_data = []
                for metric in ['blended_margin_pct', 'monthly_net', 'total_net_12m']:
                    percentile_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'P5 (Downside)': f"{analysis[metric]['p5']:,.1f}",
                        'P25': f"{analysis[metric]['p25']:,.1f}",
                        'P50 (Median)': f"{analysis[metric]['p50']:,.1f}",
                        'P75': f"{analysis[metric]['p75']:,.1f}",
                        'P95 (Upside)': f"{analysis[metric]['p95']:,.1f}"
                    })
                st.dataframe(pd.DataFrame(percentile_data), hide_index=True,
                            width="stretch")
        else:
            st.info("👈 Configure simulation parameters and click 'Run Simulation' to start.")


def render_scenarios_tab(params: SystemParameters, mau: int):
    """Render the scenario comparison tab."""
    st.header("📋 Scenario Analysis")

    segments = UserSegmentDistribution.get_segments_with_assumptions(
        params.segment_optimism, params.cpa_optimism, params.deeplink_optimism
    )
    analyzer = SensitivityAnalyzer(params)
    analyzer.set_segments(segments)

    # Predefined scenarios
    scenarios = {
        "Conservative": {
            "cpa_average": 100,
            "retention_rev_per_bet": 0.35,
            "monthly_growth_rate": 0.02,
            "coin_redemption_rate": 0.90
        },
        "Optimistic": {
            "cpa_average": 200,
            "retention_rev_per_bet": 0.75,
            "monthly_growth_rate": 0.15,
            "prize_cost_discount": 0.15
        },
        "High Growth": {
            "monthly_growth_rate": 0.20,
            "new_user_week1_tokens": 250,
            "cpa_average": 175
        },
        "Cost Pressure": {
            "linking_cost_per_month": 1.20,
            "cloud_cost_per_mau": 0.30,
            "prize_cost_discount": -0.10  # Price increases
        },
        "Revenue Decline": {
            "cpa_average": 100,
            "retention_rev_per_bet": 0.30,
            "coin_redemption_rate": 0.95
        }
    }

    # Run scenario analysis
    results = analyzer.scenario_analysis(scenarios, mau)

    # Display comparison table
    st.subheader("📊 Scenario Comparison")

    display_cols = ['scenario', 'blended_margin_pct', 'net_monthly',
                   'revenue_per_mau', 'cost_per_mau', 'ltv', 'ltv_to_cac']

    display_df = results[display_cols].copy()
    display_df.columns = ['Scenario', 'Margin %', 'Monthly Net', 'Rev/MAU',
                         'Cost/MAU', 'LTV', 'LTV:CAC']

    # Format values
    display_df['Margin %'] = display_df['Margin %'].apply(lambda x: f"{x:.1f}%")
    display_df['Monthly Net'] = display_df['Monthly Net'].apply(lambda x: f"${x:,.0f}")
    display_df['Rev/MAU'] = display_df['Rev/MAU'].apply(lambda x: f"${x:.2f}")
    display_df['Cost/MAU'] = display_df['Cost/MAU'].apply(lambda x: f"${x:.2f}")
    display_df['LTV'] = display_df['LTV'].apply(lambda x: f"${x:.2f}")
    display_df['LTV:CAC'] = display_df['LTV:CAC'].apply(lambda x: f"{x:.2f}x")

    st.dataframe(display_df, width="stretch", hide_index=True)

    # Visualization
    st.subheader("📈 Scenario Comparison Charts")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            results,
            x='scenario',
            y='blended_margin_pct',
            title="Blended Margin by Scenario",
            color='blended_margin_pct',
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=35, line_dash="dash", line_color="white",
                     annotation_text="Target (35%)")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(
            results,
            x='scenario',
            y='net_monthly',
            title="Monthly Net by Scenario",
            color='net_monthly',
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white",
                     annotation_text="Breakeven")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, width="stretch")

    # Custom scenario builder
    st.subheader("🛠️ Custom Scenario Builder")

    with st.expander("Build Custom Scenario"):
        custom_name = st.text_input("Scenario Name", "My Custom Scenario")

        col1, col2, col3 = st.columns(3)

        with col1:
            custom_cpa = st.number_input("CPA Average", value=params.cpa_average)
            custom_retention = st.number_input("Retention Rev/Bet", value=params.retention_rev_per_bet)

        with col2:
            custom_growth = st.number_input("Monthly Growth", value=params.monthly_growth_rate)
            custom_redemption = st.number_input("Redemption Rate", value=params.coin_redemption_rate)

        with col3:
            custom_linking = st.number_input("Linking Cost", value=params.linking_cost_per_month)
            custom_discount = st.number_input("Prize Discount", value=params.prize_cost_discount)

        if st.button("Add Custom Scenario"):
            custom_scenario = {
                custom_name: {
                    "cpa_average": custom_cpa,
                    "retention_rev_per_bet": custom_retention,
                    "monthly_growth_rate": custom_growth,
                    "coin_redemption_rate": custom_redemption,
                    "linking_cost_per_month": custom_linking,
                    "prize_cost_discount": custom_discount
                }
            }
            custom_results = analyzer.scenario_analysis(custom_scenario, mau)
            st.success(f"Custom scenario '{custom_name}' analyzed!")
            st.dataframe(custom_results, hide_index=True)


def render_projections_tab(params: SystemParameters, mau: int):
    """Render the financial projections tab."""
    st.header("📅 Financial Projections")

    segments = UserSegmentDistribution.get_segments_with_assumptions(
        params.segment_optimism, params.cpa_optimism, params.deeplink_optimism
    )
    projector = ProjectionEngine(params)

    col1, col2 = st.columns([1, 3])

    with col1:
        projection_years = st.slider(
            "Projection Years",
            min_value=1, max_value=5, value=3
        )
        custom_growth = st.checkbox("Custom Growth Rate")
        if custom_growth:
            growth_rate = st.slider(
                "Monthly Growth %",
                min_value=-10, max_value=30, value=int(params.monthly_growth_rate * 100)
            ) / 100
        else:
            growth_rate = params.monthly_growth_rate

    with col2:
        # Monthly projections
        monthly = projector.project_monthly(
            segments, mau, projection_years * 12, growth_rate
        )
        monthly_df = pd.DataFrame(monthly)

        # Annual projections
        annual = projector.project_annual(
            segments, mau, projection_years, growth_rate
        )
        annual_df = pd.DataFrame(annual)

        # Charts
        tab1, tab2, tab3 = st.tabs(["Monthly Trend", "Annual Summary", "Cumulative"])

        with tab1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=monthly_df['month'], y=monthly_df['revenue'],
                          name='Revenue', line=dict(color='#00ff88')),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=monthly_df['month'], y=monthly_df['cost'],
                          name='Cost', line=dict(color='#ff4444')),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=monthly_df['month'], y=monthly_df['mau'],
                          name='MAU', line=dict(color='#00aaff', dash='dot')),
                secondary_y=True
            )

            fig.update_layout(
                title="Monthly Revenue, Cost & MAU",
                template="plotly_dark",
                height=400
            )
            fig.update_yaxes(title_text="$ (Revenue/Cost)", secondary_y=False)
            fig.update_yaxes(title_text="MAU", secondary_y=True)

            st.plotly_chart(fig, width="stretch")

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=annual_df['year'],
                y=annual_df['total_revenue'],
                name='Revenue',
                marker_color='#00ff88'
            ))
            fig.add_trace(go.Bar(
                x=annual_df['year'],
                y=annual_df['total_cost'],
                name='Cost',
                marker_color='#ff4444'
            ))
            fig.add_trace(go.Bar(
                x=annual_df['year'],
                y=annual_df['total_net'],
                name='Net Profit',
                marker_color='#00aaff'
            ))

            fig.update_layout(
                title="Annual Financial Summary",
                barmode='group',
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, width="stretch")

            # Annual table
            display_annual = annual_df.copy()
            display_annual['avg_mau'] = display_annual['avg_mau'].apply(lambda x: f"{x:,.0f}")
            display_annual['end_mau'] = display_annual['end_mau'].apply(lambda x: f"{x:,.0f}")
            display_annual['total_revenue'] = display_annual['total_revenue'].apply(lambda x: f"${x:,.0f}")
            display_annual['total_cost'] = display_annual['total_cost'].apply(lambda x: f"${x:,.0f}")
            display_annual['total_net'] = display_annual['total_net'].apply(lambda x: f"${x:,.0f}")
            display_annual['avg_margin_pct'] = display_annual['avg_margin_pct'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(display_annual, width="stretch", hide_index=True)

        with tab3:
            # Cumulative chart
            monthly_df['cumulative_revenue'] = monthly_df['revenue'].cumsum()
            monthly_df['cumulative_cost'] = monthly_df['cost'].cumsum()
            monthly_df['cumulative_net'] = monthly_df['net'].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_df['month'],
                y=monthly_df['cumulative_net'],
                fill='tozeroy',
                name='Cumulative Net',
                line=dict(color='#00ff88')
            ))

            fig.update_layout(
                title="Cumulative Net Profit",
                xaxis_title="Month",
                yaxis_title="Cumulative Net ($)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, width="stretch")

            # Summary metrics
            total_revenue = monthly_df['revenue'].sum()
            total_cost = monthly_df['cost'].sum()
            total_net = monthly_df['net'].sum()
            final_mau = monthly_df['mau'].iloc[-1]

            st.markdown(f"""
            ### {projection_years}-Year Summary
            - **Total Revenue:** ${total_revenue:,.0f}
            - **Total Cost:** ${total_cost:,.0f}
            - **Total Net Profit:** ${total_net:,.0f}
            - **Final MAU:** {final_mau:,.0f}
            - **Avg Monthly Net:** ${total_net / (projection_years * 12):,.0f}
            """)


def render_segments_tab(params: SystemParameters, mau: int):
    """Render the segment analysis tab."""
    st.header("👥 User Segment Analysis")

    segments = UserSegmentDistribution.get_segments_with_assumptions(
        params.segment_optimism, params.cpa_optimism, params.deeplink_optimism
    )
    seg_analyzer = SegmentSensitivityAnalyzer(params)
    portfolio_analyzer = PortfolioAnalyzer(params)

    # Segment editor
    st.subheader("📝 Segment Configuration")

    with st.expander("Edit Segment Distributions", expanded=False):
        edited_segments = []
        cols = st.columns(len(segments))

        for i, (col, seg) in enumerate(zip(cols, segments)):
            with col:
                st.markdown(f"**{seg.name}**")
                new_pct = st.slider(
                    f"{seg.name} %",
                    min_value=0.0, max_value=0.8,
                    value=seg.percentage,
                    key=f"seg_pct_{i}"
                )
                new_rev = st.number_input(
                    f"{seg.name} Rev",
                    value=seg.monthly_revenue,
                    key=f"seg_rev_{i}"
                )
                edited_seg = UserSegment(
                    name=seg.name,
                    percentage=new_pct,
                    ups_range=seg.ups_range,
                    avg_ups=seg.avg_ups,
                    monthly_revenue=new_rev,
                    monthly_churn_rate=seg.monthly_churn_rate,
                    linking_rate=seg.linking_rate,
                    deeplink_rate=seg.deeplink_rate,
                    conversion_rate=seg.conversion_rate
                )
                edited_segments.append(edited_seg)

        # Normalize percentages
        total_pct = sum(s.percentage for s in edited_segments)
        if abs(total_pct - 1.0) > 0.01:
            st.warning(f"Segment percentages sum to {total_pct*100:.1f}% (should be 100%)")

    # Segment sensitivity analysis
    st.subheader("📊 Segment Sensitivity")

    col1, col2 = st.columns([1, 3])

    with col1:
        selected_segment = st.selectbox(
            "Select Segment",
            [s.name for s in segments]
        )
        analysis_type = st.selectbox(
            "Vary",
            ["Percentage", "Revenue", "Churn Rate"]
        )

    with col2:
        if analysis_type == "Percentage":
            results = seg_analyzer.vary_segment_percentage(
                selected_segment,
                (0.05, 0.60),
                mau=mau
            )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=results['percentage']*100, y=results['blended_margin_pct'],
                          name='Margin %', line=dict(color='#00ff88')),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=results['percentage']*100, y=results['net_monthly'],
                          name='Monthly Net', line=dict(color='#00aaff')),
                secondary_y=True
            )

            fig.update_layout(
                title=f"Impact of {selected_segment} Segment Size",
                xaxis_title="Segment Percentage (%)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, width="stretch")

        elif analysis_type == "Revenue":
            seg = next(s for s in segments if s.name == selected_segment)
            results = seg_analyzer.vary_segment_revenue(
                selected_segment,
                (0, seg.monthly_revenue * 3),
                mau=mau
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['monthly_revenue'],
                y=results['blended_margin_pct'],
                mode='lines',
                name='Margin %',
                line=dict(color='#00ff88', width=3)
            ))

            fig.update_layout(
                title=f"Impact of {selected_segment} Revenue",
                xaxis_title="Monthly Revenue per User ($)",
                yaxis_title="Blended Margin %",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, width="stretch")

        else:  # Churn Rate
            results = seg_analyzer.vary_segment_churn(
                selected_segment,
                (0.01, 0.25),
                mau=mau
            )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=results['monthly_churn_rate']*100, y=results['segment_ltv'],
                          name='Segment LTV', line=dict(color='#00ff88')),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=results['monthly_churn_rate']*100, y=results['final_retention']*100,
                          name='12M Retention %', line=dict(color='#ffaa00', dash='dot')),
                secondary_y=True
            )

            fig.update_layout(
                title=f"Impact of {selected_segment} Churn Rate",
                xaxis_title="Monthly Churn Rate (%)",
                template="plotly_dark",
                height=400
            )
            fig.update_yaxes(title_text="LTV ($)", secondary_y=False)
            fig.update_yaxes(title_text="12-Month Retention (%)", secondary_y=True)
            st.plotly_chart(fig, width="stretch")

    # LTV breakdown
    st.subheader("📈 Lifetime Value by Segment")

    ltv_data = []
    for seg in segments:
        ltv = portfolio_analyzer.calculate_ltv(seg, months=12)
        ltv_data.append({
            'Segment': seg.name,
            'LTV': ltv['ltv'],
            'Cumulative Revenue': ltv['cumulative_revenue'],
            'Cumulative Cost': ltv['cumulative_cost'],
            '12M Retention': ltv['final_retention'] * 100
        })

    ltv_df = pd.DataFrame(ltv_data)

    fig = px.bar(
        ltv_df,
        x='Segment',
        y='LTV',
        color='LTV',
        color_continuous_scale='RdYlGn',
        title="12-Month LTV by Segment"
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, width="stretch")

    # Display table
    display_ltv = ltv_df.copy()
    display_ltv['LTV'] = display_ltv['LTV'].apply(lambda x: f"${x:.2f}")
    display_ltv['Cumulative Revenue'] = display_ltv['Cumulative Revenue'].apply(lambda x: f"${x:.2f}")
    display_ltv['Cumulative Cost'] = display_ltv['Cumulative Cost'].apply(lambda x: f"${x:.2f}")
    display_ltv['12M Retention'] = display_ltv['12M Retention'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_ltv, width="stretch", hide_index=True)


def main():
    """Main application entry point."""
    # Initialize mode toggle session state
    if 'advanced_mode' not in st.session_state:
        st.session_state.advanced_mode = False

    # Header with mode toggle
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        if st.session_state.advanced_mode:
            st.title("🔬 HotTakes Profitability Engine")
            st.caption("Advanced sensitivity analysis and financial modeling")
        else:
            st.title("📊 HotTakes Financial Dashboard")
            st.caption("A simplified view of our token economy's financial health")
    with header_col2:
        st.session_state.advanced_mode = st.toggle(
            "Advanced Mode",
            value=st.session_state.advanced_mode,
            help="Toggle for detailed analysis tools and parameters"
        )

    # Create sidebar and get parameters
    params, mau = create_sidebar()

    if st.session_state.advanced_mode:
        # Advanced mode: All 6 tabs
        tabs = st.tabs([
            "📊 Overview",
            "🔬 Sensitivity",
            "🎲 Monte Carlo",
            "📋 Scenarios",
            "📅 Projections",
            "👥 Segments"
        ])

        with tabs[0]:
            render_overview_tab(params, mau)
        with tabs[1]:
            render_sensitivity_tab(params, mau)
        with tabs[2]:
            render_monte_carlo_tab(params, mau)
        with tabs[3]:
            render_scenarios_tab(params, mau)
        with tabs[4]:
            render_projections_tab(params, mau)
        with tabs[5]:
            render_segments_tab(params, mau)
    else:
        # Simple mode: 3 simplified tabs
        tabs = st.tabs([
            "📊 Overview",
            "📋 Scenarios",
            "📅 Forecast"
        ])

        with tabs[0]:
            render_simple_overview_tab(params, mau)
        with tabs[1]:
            render_simple_scenarios_tab(params, mau)
        with tabs[2]:
            render_simple_forecast_tab(params, mau)


if __name__ == "__main__":
    main()
