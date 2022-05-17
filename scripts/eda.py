import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    """
    - this class is responsible for performing 
    Exploratory Data Analysis
    """

    def __init__(self,df) -> None:
        """initialize the eda class"""
        self.df = df

    def descriptive_stats(self,describe=False,info=False,size=False):
        """
        expects: 
            - boolean
        returns:
            - summary
        """
        summary = None
        if describe:
            summary=self.df.describe()            
        elif info:
            summary=self.df.info
        elif size:
            summary=self.df.shape
        return summary

    

    