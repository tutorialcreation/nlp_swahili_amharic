import pandas as pd
import numpy as np
import pickle
import streamlit as st
from io import StringIO
from scripts.modeling import Modeler
from scripts.clean import Clean
from scripts.database import DBOps
from sklearn.ensemble import RandomForestRegressor
from dashboard.plots import view_predictions
import os
import subprocess

def clean(train_path):
    df = pd.read_csv(train_path)
    clean_df = Clean(df)
    train = df
    clean_df.transfrom_time_series('Date')
    clean_df.label_encoding(train)
    train.to_csv("data/cleaned_train_single.csv",index=False)


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


@st.cache
def predict(train,column='Sales',data=None,type='batch'):
    clean = Clean(data)
    clean.remove_unnamed_cols()
    data = clean.get_df()
    analyzer = Modeler(train)
    if type == 'single':
        forecast = analyzer.regr_models(model_=RandomForestRegressor,column=column,inputs=data,
                                    connect=True,n_estimators=10)
    elif type == 'batch':
        forecast = analyzer.regr_models(model_=RandomForestRegressor,column=column,inputs=data,
                                    connect=True,n_estimators=10)

    _,forecast_ = forecast
    return forecast_


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
                    prediction = dataframe
                    st.success("Predicted successfully")
                    csv = convert_df(dataframe)

                    st.download_button(
                        label="Download predictions",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )
                view_charts = st.button('View Predictions Chart')
                if view_charts:
                    st.session_state.page_select = 'view_forecast_chart'
                
        else:
            db = DBOps(is_online=True)
            train = pd.read_sql('select * from pharmaceuticalData',db.get_engine())
            train.reset_index(drop=True)
            train.drop('index',axis =1, inplace=True)
            train.set_index('Date',inplace=True)
            columns = train.columns.difference(['Sales','DayOfYear',
                            'WeekOfYear','Year','Month','Day'])
            params={}
            for _,x in enumerate(columns):
                params.update({f'{x}':st.text_input(f"{x}")})
            params.update({'Date':st.text_input("Date")})
            if st.button("Predict"):
                unclean_data = pd.DataFrame([params])
                unclean_data.to_csv("data/training_single.csv")
                clean("data/training_single.csv")
                cleaned_data = pd.read_csv("data/cleaned_train_single.csv")
                prediction=predict(train,'Sales',data = cleaned_data,type='single').tolist()
                for i in prediction:
                    st.success("The prediction by Model :{}".
                                format(i))

                view_charts = st.button('View Prediction Charts')
                if view_charts:
                    st.session_state.page_select = 'view_forecast_chart'

                    

        
    