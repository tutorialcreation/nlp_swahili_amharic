import pandas as pd
import numpy as np
import pickle
import streamlit as st
from io import StringIO




def predict(*args,**kwargs):

    pred_arr=np.array([kwargs])
    preds=pred_arr.reshape(1,-1)
    preds=preds.astype(float)
    # model_prediction=lr_model.predict(preds)
    return preds

def make_prediction(page=None):
    prediction  = None
    if st.session_state.page_select == page:
        st.subheader('Regressing Forecasts')
        if st.checkbox("Do you wish to upload file in order to make batch predictions,\
            if not then please proceed to make a prediction"):
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
                        format(prediction))
                

        
    