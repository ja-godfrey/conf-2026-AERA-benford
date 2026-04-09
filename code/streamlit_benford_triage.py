"""
Compatibility entrypoint for older launch commands.

The canonical dashboard now lives at `code/streamlit_esser_data_check.py`.
Importing that module here preserves older Streamlit deployment paths.
"""

from streamlit_esser_data_check import *  # noqa: F401,F403
