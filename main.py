"""Streamlit entrypoint for the VRP dashboard.

Streamlit reruns this file after every widget interaction. Importing ``src.app``
only executes the dashboard on the first run because Python caches imported
modules, so use ``runpy`` to execute the app script on every rerun.
"""

from pathlib import Path
import runpy


APP_PATH = Path(__file__).parent / "src" / "app.py"

runpy.run_path(str(APP_PATH), run_name="__main__")
