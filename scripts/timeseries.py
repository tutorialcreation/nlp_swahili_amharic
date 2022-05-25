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

    def remove_stationarity(self,df, interval=1):
        """
        - this algorithm removes stationarity
        """
        diff = list()
        for i in range(interval, len(df)):
            value = df[i] - df[i - interval]
            diff.append(value)
        return pd.Series(diff)

    def split_dataset(self,series, window_size=48, batch_size=200): 
        """initialize variables"""
        series = tf.expand_dims(series, axis=-1)
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True) 
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
        dataset = dataset.batch(batch_size).prefetch(1)
        logger.info("Successfully windowed the dataset")
        return dataset

    def model(self,x=8,y=4,neurons=1):
        """model the lstm class"""
        model = Sequential()
        model.add(LSTM(x, input_shape=[None, 1], return_sequences=True))
        model.add(LSTM(y, input_shape=[None, 1]))
        model.add(Dense(neurons))
        model.compile(loss="huber_loss", optimizer='adam')
        logger.info("Successfully modeled the neural network")
        return model

    def model_forecast(self,model, series, window_size,SIZE=0):
        """for returning the forecasts"""
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True) 
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(SIZE).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    def view_forecast(self,DateValid,XValid1,Results1,Results,WINDOW_SIZE=48):
        """for viewing the forecast"""
        plt.figure(figsize=(30, 8))
        plt.title("LSTM Model Forecast Compared to Validation Data")
        plt.plot(DateValid.astype('datetime64'), Results1, label='Forecast series')
        plt.plot(DateValid.astype('datetime64'), np.reshape(XValid1, (2*WINDOW_SIZE, 1)), label='Validation series')
        plt.xlabel('Date')
        plt.ylabel('Thousands of Units')
        plt.xticks(DateValid.astype('datetime64')[:,-1], rotation = 90) 
        plt.legend(loc="upper right")
        MAE = tf.keras.metrics.mean_absolute_error(XValid1[:,-1], Results[:,-1]).numpy()
        RMSE = np.sqrt(tf.keras.metrics.mean_squared_error(XValid1[:,-1], Results[:,-1]).numpy())
        textstr = "MAE = " + "{:.3f}".format(MAE) + "  RMSE = " + "{:.3f}".format(RMSE)
        plt.annotate(textstr, xy=(0.87, 0.05), xycoords='axes fraction')
        plt.grid(True)
        plt.show()
        logger.info("Successfully ploted time series chart")
        return textstr


if __name__=='__main__':
    train_path = sys.argv[1]
    train = pd.read_csv(train_path)
    timeseries = TimeSeries(train)