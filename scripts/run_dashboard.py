#!/usr/bin/env python3
"""Run the monitoring dashboard.

Usage:
  python scripts/run_dashboard.py
  bitdaytrader-dashboard
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Run dashboard."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: streamlit not installed")
        print("Install with: pip install -e '.[dashboard]'")
        sys.exit(1)

    dashboard_path = project_root / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)

    sys.argv = ["streamlit", "run", str(dashboard_path), "--server.port=8501"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
