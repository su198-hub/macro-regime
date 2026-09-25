"""Macro regime dashboard.

    streamlit run app.py

Entry point: page setup and navigation only. The pages live in views/.

The "data as known on" control on the dashboard is the point of the whole
thing. Move it back a year and the dashboard shows what you would have seen
then, not what you know now.
"""

import streamlit as st

from src import ui

st.set_page_config(page_title="Macro Regime Monitor", layout="wide",
                   initial_sidebar_state="collapsed")
st.html(ui.CSS)

st.navigation(
    [st.Page("views/dashboard.py", title="Dashboard", default=True),
     st.Page("views/control.py", title="Control room", url_path="control"),
     st.Page("views/methodology.py", title="Methodology", url_path="methodology"),
     st.Page("views/bench.py", title="Bench", url_path="bench"),
     # Hidden: reachable at /twin, never in the nav, so the presentation is
     # untouched while the proposed indicator set is being weighed.
     st.Page("views/twin.py", title="Twin", url_path="twin", visibility="hidden"),
     # Also hidden while its shape is argued over. /forecast asks a different
     # question from every other page: not what regime this month is, but what
     # is already determined about the next three years.
     st.Page("views/forecast.py", title="Forecast", url_path="forecast",
             visibility="hidden")],
    position="top",
).run()
