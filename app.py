import os
import sys
import streamlit as st

sys.path.insert(0, 'dashboardd')
from applications import exploration
from multiapp import MultiApp

st.set_page_config(page_title="African language Speech Recognition - Speech-to-Text ", layout="wide")

app = MultiApp()

st.sidebar.markdown("""
# Speech-to-Text
""")

# Add all your application here
app.add_app("visualizations", exploration.app)


# The main app
app.run()

