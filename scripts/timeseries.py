# importing of libraries
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from pandas.plotting import scatter_matrix
import os
print(os.getcwd())
from logger import logger
import sys
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

if __name__=='__main__':
    train_path = sys.argv[1]
    train = pd.read_csv(train_path)
    timeseries = TimeSeries(train)