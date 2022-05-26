# importing of libraries
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from statsmodels.tsa.stattools import adfuller, acf, pacf
import os
print(os.getcwd())
from logger import logger
import mlflow
import seaborn as sns
sns.set()


class TimeSeries:
    """
    - this class is for carrying out time series
    """

    def __init__(self,df) -> None:
        """
        - initialization of the class
        """
        self.df = df
        logger.info("Initialized the time series class")
    
    def get_df(self):
        """returns the df"""
        return self.df

    def perform_adfuller(self,column):
        """
        - this algorithm performs the adfuller test
        """
        result = adfuller(self.df[column].values, autolag='AIC')
        if result:
            logger.info("succesfully performed adfuller test")
        return result

    def remove_stationarity(self,df, interval=1):
        """
        - this algorithm removes stationarity
        """
        diff = list()
        for i in range(interval, len(df)):
            value = df[i] - df[i - interval]
            diff.append(value)
        return pd.Series(diff)

    def corrPlots(self,array: np.array, prefix: str):
        """
        - this algorigthm configures plots for
        correlation
        """
        plt.figure(figsize=(30, 5))
        plt.title(f"{prefix}  Autocorrelations Plots")
        plt.bar(range(len(array)), array)
        plt.grid(True)
        plt.show()
        logger.info("Successfully displayed the autocorelation plots")


if __name__=='__main__':
    """data collection"""
    train_ = pd.read_csv("data/cleaned_train.csv")
    test = pd.read_csv("data/cleaned_test.csv")
    train_.drop(['DayOfWeek','DayOfYear','WeekOfYear',
                'Customers',"Month","Day"],axis=1,inplace=True)
    test.drop(["Id",'DayOfWeek','DayOfYear','WeekOfYear'
            ,"Month","Day"],axis=1,inplace=True)
    train=train_.loc[:,train_.columns!='Sales']
    train['Sales']=train_['Sales']
    train.sort_values(["Year"], ascending=False ,ignore_index=True, inplace=True)
    test.sort_values(["Year"], ascending=False ,ignore_index=True, inplace=True)
    train.index.name = 'Year'
    train = train.set_index('Year')
    test.index.name = 'Year'
    test = test.set_index('Year')


    """deep learning model"""
    timeseries = TimeSeries(train)
    