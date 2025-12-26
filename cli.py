#!/usr/bin/env python3
"""
HotTakes Profitability Maximization Engine - CLI Tool

A command-line interface for running sensitivity analyses and generating reports.

Usage:
    python cli.py overview              # Show current financial overview
    python cli.py sensitivity <param>   # Single parameter sensitivity
    python cli.py tornado               # Tornado analysis
    python cli.py monte-carlo           # Monte Carlo simulation
    python cli.py scenarios             # Scenario comparison
    python cli.py export <format>       # Export full analysis report
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from models.financial_model import (
    SystemParameters,
    UserSegmentDistribution,
    PortfolioAnalyzer,
    ProjectionEngine,
    CostCalculator,
    TokenIssuanceEngine
)
from analysis.sensitivity import (
    SensitivityAnalyzer,
    MonteCarloSimulator,
    SegmentSensitivityAnalyzer,
    ParameterRegistry
)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.ENDC}"


def print_header(text: str):
    """Print a formatted header."""
    print()
    print(colorize("=" * 60, Colors.CYAN))
    print(colorize(f"  {text}", Colors.BOLD + Colors.CYAN))
    print(colorize("=" * 60, Colors.CYAN))
    print()


def print_subheader(text: str):
    """Print a formatted subheader."""
    print()
    print(colorize(f"--- {text} ---", Colors.YELLOW))
    print()


def print_metric(name: str, value: str, delta: Optional[str] = None):
    """Print a formatted metric."""
    if delta:
        delta_color = Colors.GREEN if not delta.startswith('-') else Colors.RED
        print(f"  {colorize(name + ':', Colors.BOLD)} {value}  ({colorize(delta, delta_color)})")
    else:
        print(f"  {colorize(name + ':', Colors.BOLD)} {value}")


def cmd_overview(params: SystemParameters, mau: int):
    """Display financial overview."""
    print_header("HotTakes Profitability Engine - Financial Overview")

    segments = UserSegmentDistribution.get_default_segments()
    analyzer = PortfolioAnalyzer(params)
    portfolio = analyzer.analyze_portfolio(segments, mau)
    breakeven = analyzer.calculate_breakeven(segments)
    cost_calc = CostCalculator(params)
    new_user_inv = cost_calc.calculate_new_user_investment()

    print_subheader("Key Metrics")
    print_metric("MAU", f"{mau:,}")
    print_metric("Blended Margin", f"{portfolio['blended_margin_pct']:.1f}%",
                 f"{portfolio['blended_margin_pct'] - 35:.1f}% vs 35% target")
    print_metric("Monthly Revenue", f"${portfolio['total_revenue']:,.0f}")
    print_metric("Monthly Cost", f"${portfolio['total_cost']:,.0f}")
    print_metric("Monthly Net", f"${portfolio['net']:,.0f}")
    print_metric("Revenue/MAU", f"${portfolio['revenue_per_mau']:.2f}")
    print_metric("Cost/MAU", f"${portfolio['cost_per_mau']:.2f}")

    print_subheader("Unit Economics")
    print_metric("Blended LTV", f"${breakeven['blended_ltv']:.2f}")
    print_metric("New User Investment", f"${new_user_inv['total_investment']:.2f}")
    print_metric("LTV:CAC Ratio", f"{breakeven['ltv_to_cac_ratio']:.2f}x",
                 "Target: 3.0x")
    print_metric("Net per User", f"${breakeven['net_per_user']:.2f}")

    print_subheader("Segment Breakdown")
    print(f"  {'Segment':<15} {'Users':>8} {'Tokens':>8} {'Prize$':>8} {'Rev$':>10} {'Net$':>10} {'Margin':>8}")
    print("  " + "-" * 75)
    for seg in portfolio['segments']:
        margin_str = f"{seg['margin_pct']:.1f}%" if seg['margin_pct'] != float('-inf') else "N/A"
        print(f"  {seg['segment']:<15} {seg['users']:>8,} {seg['tokens_per_day']:>8} "
              f"${seg['prize_cost_per_user']:>7.2f} ${seg['revenue']:>9,.0f} "
              f"${seg['net']:>9,.0f} {margin_str:>8}")

    print_subheader("New User Economics (21-day)")
    print_metric("Week 1 Cost", f"${new_user_inv['week1_cost']:.2f}")
    print_metric("Week 2 Cost", f"${new_user_inv['week2_cost']:.2f}")
    print_metric("Week 3 Cost", f"${new_user_inv['week3_cost']:.2f}")
    print_metric("Total Prize Cost", f"${new_user_inv['total_cost']:.2f}")
    print_metric("KYC Cost", f"${new_user_inv['kyc_cost']:.2f}")
    print_metric("Total Investment", f"${new_user_inv['total_investment']:.2f}")

    print()


def cmd_sensitivity(params: SystemParameters, mau: int, param_name: str):
    """Run single parameter sensitivity analysis."""
    print_header(f"Single Parameter Sensitivity: {param_name}")

    segments = UserSegmentDistribution.get_default_segments()
    analyzer = SensitivityAnalyzer(params)
    analyzer.set_segments(segments)

    # Find parameter
    all_params = ParameterRegistry.get_all_parameters(params)
    param = None
    for p in all_params:
        if p.name == param_name or p.display_name.lower() == param_name.lower():
            param = p
            break

    if not param:
        print(colorize(f"Error: Parameter '{param_name}' not found.", Colors.RED))
        print("\nAvailable parameters:")
        for p in all_params:
            print(f"  - {p.name}: {p.display_name}")
        return

    results = analyzer.single_parameter_sensitivity(param, mau)

    print(f"Parameter: {param.display_name}")
    print(f"Range: {param.min_value} to {param.max_value}")
    print(f"Current: {param.base_value}")
    print()

    print(f"  {'Value':>12} {'Margin%':>10} {'Net/Month':>12} {'LTV':>10} {'LTV:CAC':>10}")
    print("  " + "-" * 56)

    for _, row in results.iterrows():
        print(f"  {row['parameter_value']:>12.2f} "
              f"{row['blended_margin_pct']:>10.1f} "
              f"${row['net_monthly']:>11,.0f} "
              f"${row['ltv']:>9.2f} "
              f"{row['ltv_to_cac']:>10.2f}")

    print()


def cmd_tornado(params: SystemParameters, mau: int, swing_pct: float = 0.20):
    """Run tornado analysis."""
    print_header("Tornado Analysis")

    segments = UserSegmentDistribution.get_default_segments()
    analyzer = SensitivityAnalyzer(params)
    analyzer.set_segments(segments)

    results = analyzer.tornado_analysis(
        target_metric='blended_margin_pct',
        swing_pct=swing_pct,
        mau=mau
    )

    print(f"Target Metric: Blended Margin %")
    print(f"Swing: ±{swing_pct*100:.0f}%")
    print()

    print(f"  {'Parameter':<30} {'Low':>10} {'High':>10} {'Impact':>10}")
    print("  " + "-" * 62)

    for _, row in results.head(15).iterrows():
        low_delta = row['low_delta']
        high_delta = row['high_delta']
        low_str = f"{low_delta:+.1f}%"
        high_str = f"{high_delta:+.1f}%"

        print(f"  {row['parameter']:<30} {low_str:>10} {high_str:>10} {row['impact_range']:>10.1f}")

    print()


def cmd_monte_carlo(params: SystemParameters, mau: int, n_simulations: int = 1000):
    """Run Monte Carlo simulation."""
    print_header("Monte Carlo Simulation")

    segments = UserSegmentDistribution.get_default_segments()
    simulator = MonteCarloSimulator(params)
    simulator.set_segments(segments)

    print(f"Running {n_simulations} simulations...")
    results = simulator.run_simulation(n_simulations=n_simulations, mau=mau)
    analysis = simulator.analyze_results(results)

    print_subheader("Summary Statistics")

    for metric in ['blended_margin_pct', 'monthly_net', 'total_net_12m']:
        stats = analysis[metric]
        metric_name = metric.replace('_', ' ').title()
        print(f"\n  {colorize(metric_name, Colors.BOLD)}")
        print(f"    Mean:   {stats['mean']:>12,.1f}")
        print(f"    Std:    {stats['std']:>12,.1f}")
        print(f"    P5:     {stats['p5']:>12,.1f}")
        print(f"    P50:    {stats['p50']:>12,.1f}")
        print(f"    P95:    {stats['p95']:>12,.1f}")
        if stats.get('prob_positive') is not None:
            print(f"    P(>0):  {stats['prob_positive']*100:>11.1f}%")

    print()


def cmd_scenarios(params: SystemParameters, mau: int):
    """Run scenario analysis."""
    print_header("Scenario Analysis")

    segments = UserSegmentDistribution.get_default_segments()
    analyzer = SensitivityAnalyzer(params)
    analyzer.set_segments(segments)

    scenarios = {
        "Conservative": {
            "cpa_average": 100,
            "retention_rev_per_bet": 0.35,
            "monthly_growth_rate": 0.02,
        },
        "Optimistic": {
            "cpa_average": 200,
            "retention_rev_per_bet": 0.75,
            "monthly_growth_rate": 0.15,
        },
        "High Growth": {
            "monthly_growth_rate": 0.20,
            "cpa_average": 175,
        },
        "Cost Pressure": {
            "linking_cost_per_month": 1.20,
            "cloud_cost_per_mau": 0.30,
        },
    }

    results = analyzer.scenario_analysis(scenarios, mau)

    print(f"  {'Scenario':<20} {'Margin%':>10} {'Net/Month':>12} {'Rev/MAU':>10} {'LTV:CAC':>10}")
    print("  " + "-" * 64)

    for _, row in results.iterrows():
        print(f"  {row['scenario']:<20} "
              f"{row['blended_margin_pct']:>10.1f} "
              f"${row['net_monthly']:>11,.0f} "
              f"${row['revenue_per_mau']:>9.2f} "
              f"{row['ltv_to_cac']:>10.2f}")

    print()


def cmd_projections(params: SystemParameters, mau: int, years: int = 3):
    """Show financial projections."""
    print_header(f"{years}-Year Financial Projections")

    segments = UserSegmentDistribution.get_default_segments()
    projector = ProjectionEngine(params)

    annual = projector.project_annual(segments, mau, years)

    print(f"  {'Year':<6} {'Avg MAU':>12} {'Revenue':>14} {'Cost':>14} {'Net':>14} {'Margin%':>10}")
    print("  " + "-" * 72)

    for year in annual:
        print(f"  {year['year']:<6} "
              f"{year['avg_mau']:>12,.0f} "
              f"${year['total_revenue']:>13,.0f} "
              f"${year['total_cost']:>13,.0f} "
              f"${year['total_net']:>13,.0f} "
              f"{year['avg_margin_pct']:>10.1f}")

    print()

    # Summary
    total_net = sum(y['total_net'] for y in annual)
    final_mau = annual[-1]['end_mau']
    print(f"  Total {years}-year Net: ${total_net:,.0f}")
    print(f"  Final MAU: {final_mau:,.0f}")
    print()


def cmd_export(params: SystemParameters, mau: int, format: str, output_path: Optional[str] = None):
    """Export comprehensive analysis to file."""
    print_header("Exporting Analysis Report")

    segments = UserSegmentDistribution.get_default_segments()
    analyzer = SensitivityAnalyzer(params)
    analyzer.set_segments(segments)
    portfolio_analyzer = PortfolioAnalyzer(params)
    projector = ProjectionEngine(params)
    simulator = MonteCarloSimulator(params)
    simulator.set_segments(segments)

    # Generate all analyses
    print("Generating portfolio analysis...")
    portfolio = portfolio_analyzer.analyze_portfolio(segments, mau)
    breakeven = portfolio_analyzer.calculate_breakeven(segments)

    print("Running tornado analysis...")
    tornado = analyzer.tornado_analysis(mau=mau)

    print("Running Monte Carlo simulation (500 iterations)...")
    mc_results = simulator.run_simulation(n_simulations=500, mau=mau)
    mc_analysis = simulator.analyze_results(mc_results)

    print("Generating projections...")
    projections = projector.project_annual(segments, mau, years=3)

    # Build report
    report = {
        'generated_at': datetime.now().isoformat(),
        'parameters': {
            'mau': mau,
            'floor_tokens': params.floor_tokens,
            'base_tokens': params.base_tokens,
            'theta': params.theta,
            'beta': params.beta,
            'cap_tokens': params.cap_tokens,
            'coins_per_dollar': params.coins_per_dollar,
            'coin_redemption_rate': params.coin_redemption_rate,
            'prize_cost_discount': params.prize_cost_discount,
            'cpa_average': params.cpa_average,
            'retention_rev_per_bet': params.retention_rev_per_bet,
            'linking_cost_per_month': params.linking_cost_per_month,
            'cloud_cost_per_mau': params.cloud_cost_per_mau,
            'monthly_growth_rate': params.monthly_growth_rate,
        },
        'portfolio': {
            'blended_margin_pct': portfolio['blended_margin_pct'],
            'total_revenue': portfolio['total_revenue'],
            'total_cost': portfolio['total_cost'],
            'net': portfolio['net'],
            'revenue_per_mau': portfolio['revenue_per_mau'],
            'cost_per_mau': portfolio['cost_per_mau'],
        },
        'breakeven': {
            'blended_ltv': breakeven['blended_ltv'],
            'new_user_investment': breakeven['new_user_investment'],
            'ltv_to_cac_ratio': breakeven['ltv_to_cac_ratio'],
            'net_per_user': breakeven['net_per_user'],
        },
        'segments': portfolio['segments'],
        'tornado': tornado.to_dict('records'),
        'monte_carlo': {
            'n_simulations': 500,
            'blended_margin_pct': mc_analysis['blended_margin_pct'],
            'monthly_net': mc_analysis['monthly_net'],
            'total_net_12m': mc_analysis['total_net_12m'],
        },
        'projections': projections,
    }

    # Determine output path
    if not output_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"hottakes_analysis_{timestamp}.{format}"

    # Export based on format
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    elif format == 'csv':
        # Export multiple CSVs in a directory
        output_dir = Path(output_path).with_suffix('')
        output_dir.mkdir(exist_ok=True)

        # Portfolio summary
        pd.DataFrame([report['portfolio']]).to_csv(output_dir / 'portfolio.csv', index=False)

        # Segments
        pd.DataFrame(report['segments']).to_csv(output_dir / 'segments.csv', index=False)

        # Tornado
        pd.DataFrame(report['tornado']).to_csv(output_dir / 'tornado.csv', index=False)

        # Projections
        pd.DataFrame(report['projections']).to_csv(output_dir / 'projections.csv', index=False)

        output_path = str(output_dir)
    else:
        print(colorize(f"Error: Unknown format '{format}'", Colors.RED))
        return

    print(colorize(f"\nReport exported to: {output_path}", Colors.GREEN))
    print()


def cmd_list_params(params: SystemParameters):
    """List all available parameters for sensitivity analysis."""
    print_header("Available Parameters")

    all_params = ParameterRegistry.get_all_parameters(params)
    categories = {}

    for p in all_params:
        if p.category not in categories:
            categories[p.category] = []
        categories[p.category].append(p)

    for category, params_list in sorted(categories.items()):
        print(f"\n  {colorize(category, Colors.BOLD)}")
        for p in params_list:
            print(f"    {p.name:<30} = {p.base_value:>10} ({p.min_value} - {p.max_value})")

    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HotTakes Profitability Engine - Sensitivity Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py overview
  python cli.py sensitivity cpa_average
  python cli.py tornado --swing 25
  python cli.py monte-carlo --simulations 2000
  python cli.py scenarios
  python cli.py projections --years 5
  python cli.py export json --output report.json
  python cli.py params

Parameter Overrides:
  Use --param=value to override default parameters:
  python cli.py overview --cpa-average=200 --monthly-growth-rate=0.10
        """
    )

    parser.add_argument('command', choices=[
        'overview', 'sensitivity', 'tornado', 'monte-carlo',
        'scenarios', 'projections', 'export', 'params'
    ], help='Command to run')

    parser.add_argument('args', nargs='*', help='Command arguments')

    # MAU override
    parser.add_argument('--mau', type=int, default=24000,
                        help='Monthly Active Users (default: 24000)')

    # Parameter overrides
    parser.add_argument('--floor-tokens', type=int)
    parser.add_argument('--base-tokens', type=int)
    parser.add_argument('--theta', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--cap-tokens', type=int)
    parser.add_argument('--coins-per-dollar', type=float)
    parser.add_argument('--coin-redemption-rate', type=float)
    parser.add_argument('--prize-cost-discount', type=float)
    parser.add_argument('--cpa-average', type=float)
    parser.add_argument('--retention-rev-per-bet', type=float)
    parser.add_argument('--linking-cost-per-month', type=float)
    parser.add_argument('--cloud-cost-per-mau', type=float)
    parser.add_argument('--monthly-growth-rate', type=float)

    # Command-specific options
    parser.add_argument('--swing', type=int, default=20,
                        help='Tornado swing percentage (default: 20)')
    parser.add_argument('--simulations', type=int, default=1000,
                        help='Monte Carlo simulations (default: 1000)')
    parser.add_argument('--years', type=int, default=3,
                        help='Projection years (default: 3)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path')

    args = parser.parse_args()

    # Create parameters with overrides
    params = SystemParameters()
    param_overrides = {
        'floor_tokens': args.floor_tokens,
        'base_tokens': args.base_tokens,
        'theta': args.theta,
        'beta': args.beta,
        'cap_tokens': args.cap_tokens,
        'coins_per_dollar': args.coins_per_dollar,
        'coin_redemption_rate': args.coin_redemption_rate,
        'prize_cost_discount': args.prize_cost_discount,
        'cpa_average': args.cpa_average,
        'retention_rev_per_bet': args.retention_rev_per_bet,
        'linking_cost_per_month': args.linking_cost_per_month,
        'cloud_cost_per_mau': args.cloud_cost_per_mau,
        'monthly_growth_rate': args.monthly_growth_rate,
    }

    for key, value in param_overrides.items():
        if value is not None:
            setattr(params, key, value)
    params.update_derived()

    # Route to command
    if args.command == 'overview':
        cmd_overview(params, args.mau)
    elif args.command == 'sensitivity':
        if not args.args:
            print(colorize("Error: Please specify a parameter name", Colors.RED))
            cmd_list_params(params)
            return
        cmd_sensitivity(params, args.mau, args.args[0])
    elif args.command == 'tornado':
        cmd_tornado(params, args.mau, args.swing / 100)
    elif args.command == 'monte-carlo':
        cmd_monte_carlo(params, args.mau, args.simulations)
    elif args.command == 'scenarios':
        cmd_scenarios(params, args.mau)
    elif args.command == 'projections':
        cmd_projections(params, args.mau, args.years)
    elif args.command == 'export':
        if not args.args:
            print(colorize("Error: Please specify export format (json, csv)", Colors.RED))
            return
        cmd_export(params, args.mau, args.args[0], args.output)
    elif args.command == 'params':
        cmd_list_params(params)


if __name__ == '__main__':
    main()
