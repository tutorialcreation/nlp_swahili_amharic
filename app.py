
from dashboard.routes import router
import streamlit as st
# """
# - the purpose of this file
# Use one of the platforms of your choice (Flask, Streamlit, pure javascript, etc.) to design, and build 
# a backend to make inference using your trained model and input parameters collected through a frontend interface.

# Your dashboard should provide an easy way for a user (in this case managers of the stores) to enter required 
# input parameters, and output the predicted sales amount and customer numbers.

# The input fields in the frontend are for example

#     Store_id
#     Upload csv file with a columns name

#     Date
#     IsHoliday
#     IsWeekend
#     IsPromo

#     Any other parameter which is dependent on the date
#     Any any other parameter requires as input for your model that is not dependent on date

# Finally your dashboard should show a plot that shows the predicted sales amount and number of customers. It 
# should also allow the user to download the prediction in the form of a csv table.
# """

def run():
    st.title("Pharmaceutical Sales Prediction")
    st.session_state.page_select = st.sidebar.radio('SiteMap', 
            [
                'make_forecast', 
                'view_forecast_chart'
            ])
    # routes
    router(page=st.session_state.page_select)    
    

if __name__=='__main__':
    run()

