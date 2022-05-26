import pandas as pd
import numpy as np
import pickle
import streamlit as st
from io import StringIO

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



def predict(**kwargs):
    pred_arr=np.array(list(kwargs.values()))
    preds=pred_arr.reshape(1,-1)
    preds=preds.astype(float)
    # model_prediction=lr_model.predict(preds)
    return preds

def make_prediction(page=None):
    prediction  = None
    if st.session_state.page_select == page:
        st.subheader('Regressing Forecasts')
        if st.checkbox("Do you wish to upload file in order to make batch predictions,\
            if not then please proceed to make a single sales prediction"):
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                # Can be used wherever a "file-like" object is accepted:
                dataframe = pd.read_csv(uploaded_file)
                if st.button("Predict"):
                    st.dataframe(dataframe.head())
                    st.success("Predicted successfully")
                    csv = convert_df(dataframe)

                    st.download_button(
                        label="Download predictions",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )
        else:
            params = {
                'var_1':st.text_input("Variable 1"),
                'var_2':st.text_input("Variable 2"),
                'var_3':st.text_input("Variable 3"),
                'var_4':st.text_input("Variable 4"),
                'var_5':st.text_input("Variable 5")
            }
            if st.button("Predict"):
                prediction=predict(**params)
                st.success("The prediction by Model :{}".
                            format(prediction.shape))
                    

        
    