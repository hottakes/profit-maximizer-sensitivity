#!/bin/bash

# HotTakes Profitability Maximization Engine - Run Script

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Determine what to run
if [ "$1" == "cli" ]; then
    shift
    python cli.py "$@"
elif [ "$1" == "help" ]; then
    echo ""
    echo "HotTakes Profitability Maximization Engine"
    echo "==========================================="
    echo ""
    echo "Usage:"
    echo "  ./run.sh              # Launch interactive dashboard"
    echo "  ./run.sh cli <cmd>    # Run CLI commands"
    echo "  ./run.sh help         # Show this help"
    echo ""
    echo "CLI Commands:"
    echo "  ./run.sh cli overview              # Financial overview"
    echo "  ./run.sh cli sensitivity <param>   # Single param sensitivity"
    echo "  ./run.sh cli tornado               # Tornado analysis"
    echo "  ./run.sh cli monte-carlo           # Monte Carlo simulation"
    echo "  ./run.sh cli scenarios             # Scenario comparison"
    echo "  ./run.sh cli projections           # Financial projections"
    echo "  ./run.sh cli export json           # Export report"
    echo "  ./run.sh cli params                # List all parameters"
    echo ""
    echo "Dashboard will be available at: http://localhost:8501"
    echo ""
else
    echo ""
    echo "Starting HotTakes Profitability Dashboard..."
    echo "Dashboard will be available at: http://localhost:8501"
    echo ""
    streamlit run dashboard/app.py
fi
