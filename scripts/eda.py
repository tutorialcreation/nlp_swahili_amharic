import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pandas.plotting import scatter_matrix


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

    def plot_counts(self, column, second_column=None, type=None):
        """
        expects:
            -   string
        returns:
            -   plot
        """
        if type == "univariate":
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.df, x=column)
            plt.title(f"Unique value counts of the {column} columns")
            plt.show()
        elif type == "bivariate":
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.df, x=second_column, hue=column)
            plt.title(f"{column} vs {second_column}")
            plt.show()
        elif type == "bivariate_line":
            plt.figure(figsize=(28,10))
            sns.lineplot(self.df.index, y= self.df[column], hue=self.df[second_column]);
            plt.show()
        elif type == "bivariate_scatter":
            plt.figure(figsize=(12, 8))
            plt.scatter(self.df[column],self.df[second_column], alpha=0.1)
            plt.xlabel(column)
            plt.ylabel(second_column)
            plt.plot()
        elif type == "bivariate_hist":
            cols = ['purple','green']
            labels = ['With','Without']
            for i in reversed(range(0,2)):
                promos = self.df[self.df[column] == i][second_column]
                plt.hist(promos, 
                color=cols[i], alpha=0.3, label =labels[i])
            
            plt.ylabel(column)
            plt.xlabel(second_column)
            plt.legend()
            plt.plot()
        elif type == "bivariate_count":
            sns.countplot( x=column, data=self.df, hue=second_column, palette="Set1")
            plt.show()

        return


    def between(self,column,start,stop):
        """
        - get values in between certain range
        """
        return self.df[self.df[column].between(start,stop)]

    
    def correlation_analysis(self,column=None,second_column=None,type="all"):
        """
        expects:
            - nothing
        returns:
            - dataframe
        """
        if type=="all":
            corr = self.df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True)
            plt.title('Heatmap of correlation for the numerical columns')
            plt.show()
        elif type=="bivariate":
            attributes = [column,second_column]
            scatter_matrix(self.df[attributes], alpha=0.1);
            plt.show()
        return 

    def plot_distributions(self):
        """
        - this algorithm is responsible for plotting distributions
        """
        num_feats = list(self.df.select_dtypes(include=['int64', 'float64', 'int32']).columns)
        self.df[num_feats].hist(figsize=(20,15))

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
