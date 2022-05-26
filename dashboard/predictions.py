import pandas as pd
import numpy as np
import pickle
import streamlit as st
from io import StringIO

def make_prediction(page=None):
    
    if st.session_state.page_select == page:
        st.subheader('Regressing Forecasts')
        
        if st.checkbox("Upload file in order to make batch predictions"):
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                st.write(bytes_data)

                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                st.write(stringio)

                # To read file as string:
                string_data = stringio.read()
                st.write(string_data)

                # Can be used wherever a "file-like" object is accepted:
                dataframe = pd.read_csv(uploaded_file)
                st.write(dataframe)
                

        
    