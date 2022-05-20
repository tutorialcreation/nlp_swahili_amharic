import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys


class EDA:
    """
    - this class is responsible for performing 
    Exploratory Data Analysis
    """

    def __init__(self, df):
        """initialize the eda class"""
        self.df = df

    def descriptive_stats(self, describe=False, info=False, size=False):
        """
        expects: 
            - boolean
        returns:
            - summary
        """
        summary = None
        if describe:
            summary = self.df.describe()
        elif info:
            summary = self.df.info
        elif size:
            summary = self.df.shape
        return summary

    def has_missing_values(self):
        """
        expects:
            -   nothing
        returns:
            -   boolean
        """
        has_missing_values = False
        if True in self.df.isnull().any().to_list():
            has_missing_values = True
        return has_missing_values

    # def plot_counts(self, column, second_column=None, type=None):
    #     """
    #     expects:
    #         -   string
    #     returns:
    #         -   plot
    #     """
    #     if type == "univariate":
    #         plt.figure(figsize=(12, 6))
    #         sns.countplot(data=self.df, x=column)
    #         plt.title(f"Unique value counts of the {column} columns")
    #         plt.show()
    #     elif type == "bivariate":
    #         plt.figure(figsize=(12, 6))
    #         sns.countplot(data=self.df, x=second_column, hue=column)
    #         plt.title(f"{column} vs {second_column}")
    #         plt.show()
    #     return

    # def correlation_analysis(self):
    #     """
    #     expects:
    #         - nothing
    #     returns:
    #         - dataframe
    #     """
    #     corr = self.df.corr()
    #     fig, ax = plt.subplots()
    #     sns.heatmap(corr, annot=True)
    #     plt.title('Heatmap of correlation for the numerical columns')
    #     plt.show()
    #     return fig

    def get_df(self):
        """
        - returns the dataframes
        """
        return self.df


if __name__ == '__main__':
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    eda = EDA(df)
    eda_df = eda.get_df()
    eda_df.to_csv("data/eda.csv", index=False)
