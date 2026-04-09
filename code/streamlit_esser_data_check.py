"""
Canonical entrypoint for the ESSER data-check dashboard.

This wrapper executes the stable app script on every Streamlit rerun.
Using `run_path` avoids Python import caching, which can otherwise leave the
page blank after the first widget interaction.
"""

from pathlib import Path
from runpy import run_path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

run_path(str(HERE / "streamlit_benford_esser.py"), run_name="__main__")
