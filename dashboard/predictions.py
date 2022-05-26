import pandas as pd
import numpy as np
import pickle
import streamlit as st

def make_prediction(page=None):
    
    if st.session_state.page_select == page:
        st.title('Page 1')
        
    