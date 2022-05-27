import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.database import DBOps
sns.set()

def view_predictions(page=None,results=None):
    if st.session_state.page_select == page:
        st.subheader('Predictions Chart')
        db = DBOps(is_online=True)
        df = pd.read_sql('select * from pharmaceuticalData',db.get_engine())
        df.set_index('Date',inplace=True)
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=df['Sales'],hue=df['Customers'])
        st.pyplot(fig)
        # fig = plt.figure(figsize=(10, 4))
        # sns.lineplot(data=df['Customers'])
        # st.pyplot(fig)
        