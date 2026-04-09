"""
Compatibility entrypoint for the renamed Streamlit app.

If Streamlit Community Cloud or a local launch command still points at
`code/streamlit_benford_esser.py`, importing the canonical app module below
will execute the current dashboard.
"""

from streamlit_esser_data_check import *  # noqa: F401,F403
