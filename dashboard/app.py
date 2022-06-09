import sounddevice as sd
import streamlit as st
import pandas as pd
from PIL import Image
import os
 


def run():
    
    st.title("African Language Transcription")
    
    with st.sidebar:
        option = st.selectbox(
        "CHOOSE A LANGUAGE",
        ("Swahili", "Amharic")
    )

    st.subheader('Upload audio file')
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

    st.subheader('Transcription of audio')
    

if __name__=='__main__':
    run()