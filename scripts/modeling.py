import pandas as pd
import numpy as np

# To Preproccesing our data
from sklearn.preprocessing import LabelEncoder

# To fill missing values
from sklearn.impute import SimpleImputer

# To Split our train data
from sklearn.model_selection import train_test_split

# To Visualize Data
import matplotlib.pyplot as plt
import seaborn as sns

# To Train our data
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB

# To evaluate end result we have
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

class Modeler:

    """
    - this class is responsible for modeling
    """

    def __init__(self,df):
        """
        - Initialization of the class
        """
        self.df = df

    def generate_pipeline(self,type_="numeric",x=1):
        """
        purpose:
            - generate_pipelines for the data
        input:
            - string and int
        returns:
            - pipeline
        """
        pipeline = None
        if type_ == "numeric":
            pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', MinMaxScaler())
            ])
        elif type_ == "categorical":
            pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        else:
            pipeline = np.zeros(x)
        return pipeline

    def generate_transformation(self,pipeline,type_,value,trim=None,key=None):
        """
        purpose:
            - generates transformations for the data
        input:
            - string,int and df
        returns:
            - transformation
        """
        transformation = None
        if type_=="numeric":
            transformation=pipeline.fit_transform(self.df.select_dtypes(include=value))
            if trim:
                transformation=pipeline.fit_transform(pd.DataFrame(self.split_data(key,0.3,trim)).select_dtypes(include=value))
        elif type_ == "categorical":
            transformation=pipeline.fit_transform(self.df.select_dtypes(exclude=value))
            if trim:
                transformation=pipeline.fit_transform(pd.DataFrame(self.split_data(key,0.3,trim)).select_dtypes(exclude=value))
        return transformation


if __name__=="__main__":
    df = pd.read_csv("data/data.csv")
    analyzer = Modeler(df)
    numeric_pipeline = analyzer.generate_pipeline("numeric")
    numeric_transformation =  analyzer.generate_transformation(numeric_pipeline,"numeric","number")
    
    print(numeric_transformation)