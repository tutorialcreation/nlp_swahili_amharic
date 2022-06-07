import streamlit as st
import pandas as pd
import sys
import pickle


def app():

    st.title("African language Speech Recognition - Speech-to-Text ")

    st.header("Visualization")

    st.subheader("distributon of words for Amharic")
    st.image('data/amh distribution.png')

    st.subheader("distributon of words for Swahili")
    st.image('data/swahili distribution.png')

    st.subheader("Data Augmentaion")
    st.image('data/data augmentation.png')
    
    st.subheader("Feature Extraction")
    st.image('data/feat extraction.png')

    st.subheader("spectogram")
    st.image('data/spectogram.png')

    st.subheader("Mel-Frequency Cepstral Coefficients")
    st.image('data/mfcc.png')

