"""Macro regime dashboard.

    streamlit run app.py

Entry point: page setup and navigation only. The pages live in views/.

The "data as known on" control on the dashboard is the point of the whole
thing. Move it back a year and the dashboard shows what you would have seen
then, not what you know now.
"""

import hashlib
import pathlib
import sys

import streamlit as st

# A deploy pulls new code into the running process. Streamlit re-runs the page
# files, but can keep the modules they import and every cache from before: a
# page then calls a function its old module lacks, or reads a cached result
# missing a field it now needs. So on the first run after the code changes,
# drop the project's modules and all caches, and rebuild from the new code.
_ROOT = pathlib.Path(__file__).resolve().parent


def _code_stamp() -> str:
    h = hashlib.sha1()
    for p in sorted([*(_ROOT / "src").rglob("*.py"), *(_ROOT / "views").glob("*.py"),
                     *(_ROOT / "scripts").glob("*.py"), *(_ROOT / "config").glob("*.yml")]):
        s = p.stat()
        h.update(f"{p.relative_to(_ROOT)}:{s.st_mtime_ns}:{s.st_size}".encode())
    return h.hexdigest()


_stamp = _code_stamp()
# A process that has not recorded a stamp may be one this guard is new to, so
# purge then too: in a fresh process there is nothing to drop.
if getattr(sys, "_macro_regime_code", None) != _stamp:
    for _name in [m for m in sys.modules
                  if m.split(".")[0] in ("src", "views", "scripts")]:
        del sys.modules[_name]
    st.cache_data.clear()
    st.cache_resource.clear()
sys._macro_regime_code = _stamp

from src import ui  # noqa: E402  (after the purge, so it is the new code)

st.set_page_config(page_title="Macro Regime Monitor", layout="wide",
                   initial_sidebar_state="collapsed")
st.html(ui.CSS)

st.navigation(
    [st.Page("views/dashboard.py", title="Dashboard", default=True),
     st.Page("views/control.py", title="Control room", url_path="control"),
     st.Page("views/methodology.py", title="Methodology", url_path="methodology"),
     st.Page("views/bench.py", title="Bench", url_path="bench"),
     # Draft benchmark: the regime each month turned out to be in, since 1968,
     # laid out so a reader can check every call against the inputs.
     st.Page("views/history.py", title="Regime history", url_path="history"),
     # In the menu but marked work in progress: is energy demand outrunning
     # supply? Scored on its own, not yet feeding the regime model.
     st.Page("views/energy.py", title="Energy (WIP)", url_path="energy"),
     # Hidden: reachable at /twin, never in the nav, so the published model is
     # untouched while the proposed indicator set is being weighed.
     st.Page("views/twin.py", title="Twin", url_path="twin", visibility="hidden"),
     # Also hidden while its shape is argued over. /forecast asks a different
     # question from every other page: not what regime this month is, but what
     # is already determined about the next three years.
     st.Page("views/forecast.py", title="Forecast", url_path="forecast",
             visibility="hidden"),
     # Regime shapes, any month since 1968, and the v2 three-year call.
     # Kept at /outlook so links from when it was the three-year outlook still work.
     st.Page("views/outlook.py", title="Regime explorer", url_path="outlook")],
    position="top",
).run()
