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
     st.Page("views/methodology.py", title="Methodology", url_path="methodology")],
    position="top",
).run()
