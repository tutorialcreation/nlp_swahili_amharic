# importing of libraries
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from logger import logger
import sys
import seaborn as sns
sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams["figure.figsize"] = (12, 8)
pd.set_option('display.max_columns', None)


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


if __name__=='__main__':
    train_path = sys.argv[1]
    train = pd.read_csv(train_path)
    timeseries = TimeSeries(df)