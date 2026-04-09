"""
Compatibility entrypoint for older launch commands.

This wrapper executes the canonical dashboard script on every rerun so older
deployment paths behave the same way as the current entrypoint.
"""

from pathlib import Path
from runpy import run_path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

run_path(str(HERE / "streamlit_benford_esser.py"), run_name="__main__")
