"""
Wrapper for PSI monitoring to ensure consistent execution.

Delegates logic to `api_drift_test.py`, which uses:
- paths from params.yaml
- proper exit code on drift
- feature list from monitoring config
"""

import sys

from .api_drift_test import main as run_drift

if __name__ == "__main__":
    sys.exit(run_drift())
