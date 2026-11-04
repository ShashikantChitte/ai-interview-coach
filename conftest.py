"""Pytest configuration — adds project root to sys.path.

Without this, pytest only adds `tests/` to sys.path, so
`from src.schemas import ...` fails with ModuleNotFoundError.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
