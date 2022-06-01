import pandas as pd
import numpy as np
import math
import time
import pickle
import mlflow
# To Preproccesing our data
from sklearn.preprocessing import LabelEncoder,StandardScaler

# To fill missing values
from sklearn.impute import SimpleImputer

# To Split our train data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
# sensitivity analysis of k in k-fold cross-validation
from numpy import mean
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB

import os,sys
# sys.path.append(os.path.abspath(os.path.join('../scripts')))
from scripts.logger import logger
from scripts.model_serializer import ModelSerializer
# To evaluate end result we have
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

class Modeler:

    """
    - this class is responsible for modeling
    """

    def __init__(self,df=None):
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

    def make_last(self, data_,target_variable):
        """this functions allows one choose which column to be last"""
        data=data_.loc[:,data_.columns!=target_variable]
        data[target_variable]=data_[target_variable]
        return data
    
    def preprocessor_audio(self,data,columns_to_drop,target_variable):
        """
        this function is for preprocessing audio data
        """
        data = data.drop([columns_to_drop],axis=1)#Encoding the Labels
        data=self.make_last(data,target_variable)
        genre_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(genre_list)#Scaling the Feature columns
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return (X_train, X_test, y_train, y_test)

    def preprocessing_learn(self,train,columns_to_drop,target_variable):
        """
        this function creates preprocessed data for deep learning
        """
        train = train.drop([columns_to_drop],axis=1)#Encoding the Labels
        train=self.make_last(train,target_variable)
        # genre_list = train.iloc[:, -1]
        # encoder = LabelEncoder()
        # y = encoder.fit_transform(genre_list)#Scaling the Feature columns
        # train[target_variable] = y
        n = len(train)
        train_df = train[0:int(n*0.7)]
        val_df = train[int(n*0.7):int(n*0.9)]
        test_df = train[int(n*0.9):]
        return (train_df,val_df,test_df)


    def store_features(self,type_,value):
        """
        purpose:
            - stores features for the data set
        input:
            - string,int,dataframe
        returns:
            - dataframe
        """
        features = [None]
        if type_ == "numeric":
            features = self.df.select_dtypes(include=value).columns.tolist()
        elif type_ == "categorical":
            features = self.df.select_dtypes(exclude=value).columns.tolist()
        return features

    def encoding_data(self):
        """
        - responsible for encoding the columns

        """
        categorical_features = self.store_features("categorical","number")
        to_one_hot_encoding = [col for col in categorical_features if self.df[col].nunique() <= 10 and self.df[col].nunique() > 2]
        # Get Categorical Column names thoose are not in "to_one_hot_encoding"
        to_label_encoding = [col for col in categorical_features if not col in to_one_hot_encoding]
        return to_one_hot_encoding,to_label_encoding

    def hot_encode(self):
        """
        - responsible one hot encoding the columns
        """
        to_one_hot_encoding,_= self.encoding_data()
        one_hot_encoded_columns = pd.get_dummies(self.df[to_one_hot_encoding],ignore_index=True) if len(to_one_hot_encoding) > 0 else pd.DataFrame(columns=to_one_hot_encoding)
        return one_hot_encoded_columns

    def label_encode(self):
        """
        - responsible for label ecoding the column
        """
        _,to_label_encoding = self.encoding_data()
        label_encoded_columns = []
        # For loop for each columns
        for col in to_label_encoding:
            # We define new label encoder to each new column
            le = LabelEncoder()
            # Encode our data and create new Dataframe of it, 
            # notice that we gave column name in "columns" arguments
            column_dataframe = pd.DataFrame(le.fit_transform(self.df[col]), columns=[col] )
            # and add new DataFrame to "label_encoded_columns" list
            label_encoded_columns.append(column_dataframe)

        # Merge all data frames
        label_encoded_columns = pd.concat(label_encoded_columns, axis=1)
        return label_encoded_columns
    
    def merge_data(self,encoded=False):
        """
        - responsible for bringing all the data together
        """
        # Copy our DataFrame to X variable
        X = self.df.copy()

        # Droping Categorical Columns,
        # "inplace" means replace our data with new one
        # Don't forget to "axis=1"
        categorical_features = self.store_features("categorical","number")
        if categorical_features:
            X.drop(categorical_features, axis=1, inplace=True)
        else:
            X = X

        if not encoded:
            one_hot_encoded_columns = self.hot_encode()
            label_encoded_columns = self.label_encode()
            if one_hot_encoded_columns and label_encoded_columns:
                X = pd.concat([X, one_hot_encoded_columns, label_encoded_columns], axis=1)
        else:
            X = X

        return X
    


    def groupby_column(self,column="browser_Chrome Mobile",index=1):
        """
        - group according to the different columns
        """
        grouped_data = self.df[self.df[column]==index]
        return grouped_data

    def get_columns(self,column="yes",encoded=False):
        """
        - responsible for getting the columns
        """

        # Droping "class" from X
        X = self.merge_data(encoded)
        y = X[column]
        X.drop([column], axis=1, inplace=True)
        return X,y



    def split_data(self,column="yes",encoded=False):
        """
        - responsible for splitting the data
        """
        X,y =self.get_columns(column,encoded)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=42)
        return X_train, X_test, X_val, y_train, y_test,y_val

    def model(self,model,column=None,encoded=False,**kwargs):
        """
        - model the dataset
        """
        X_train, X_test, X_val, y_train, y_test,y_val = self.split_data(column,encoded)
        # Define Random Forest Model
        model = model(**kwargs)
        # We fit our model with our train data
        model.fit(X_train, y_train)
        # Then predict results from X_test data
        predicted_data = model.predict(X_test)
        # generate a confusion matrix
        # get accuracy score
        return predicted_data

    
    def get_model(self,model=LogisticRegression,**kwargs):
        """
        - this method does simple returning of the model
        """
        model_=model(**kwargs)
        return model_

    def custom_log_loss(self,actual,predicted):
        """
        - this algorithm finds the log loss
        """
        losses = []
        for i in range(len(actual)):
            if actual[i] != 0:
                losses.append(actual[i]*math.log(predicted[i])+(1-actual[i])*math.log(1-predicted[i]))
        return sum(losses)/len(actual) 
        
    
    #loss function for models
    def log_loss(self, model = LogisticRegression,column="yes",custom=True,**kwargs):
        """
        - loss function
        """
        X_train, X_test, X_val, y_train, y_test,y_val = self.split_data(column,True)
        model_ = model(random_state=0)
        fitted= model_.fit(X_train,y_train)
        pred_proba =fitted.predict(X_test)
        
        # Running Log loss on training
        if custom:
            validation_loss = self.custom_log_loss(y_test.to_numpy(), pred_proba)
        else:
            validation_loss = log_loss(y_test, pred_proba)
        return validation_loss


    def feature_importance(self,model_,column="yes",**kwargs):
        """
        - an algorithm for checking feature importance
        """
        #initialization
        model = model_(**kwargs)
        X,y =self.get_columns(column,True)
        model.fit(X,y)
        #plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.show()
        return feat_importances

    
    def get_folds(self,fold):
        """
        - get cross validation
        """
        cv = KFold(n_splits=fold, random_state=1, shuffle=True)
        return cv

    def regr_models(self,model_=None,column="yes",inputs=None,
                connect=True,serialize=True,**kwargs):
        """
        - evaluates the algorithm
        """
        # get the dataset
        # get the model
        if column:
            X_train, X_test, X_val, y_train, y_test,y_val = self.split_data(column,True)
        scores = 0.0
        mlflow.sklearn.autolog()
        with mlflow.start_run(run_name="regression-modeling") as run:
            mlflow.set_tag("mlflow.runName", "regression-modeling")
            model = model_(**kwargs)
            model.fit(X_train,y_train)
            mlflow.sklearn.log_model(model,"model_random_forest_regressor")
            logger.info(f"fitted a {model} model")
            # Then predict results from X_test data
            if connect:
                inputs_ = inputs.to_numpy()
                predicted_data=model.predict(inputs_)
                logger.info("predicting for a single instance")
            else:
                predicted_data = model.predict(X_test)
                scores = mean_absolute_error(y_test, predicted_data)
                logger.info("predicting for a group instance")
            mlflow.log_metric("scores",scores)
            # serialize the model
            if serialize:
                serializer = ModelSerializer(model)
                serializer.pickle_serialize()
        return (scores,predicted_data)


    def get_df(self):
        """
        - get the df
        """
        return self.df
    


if __name__=="__main__":
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
    analyzer = Modeler(train)
    # mlflow.sklearn.autolog()
    # with mlflow.start_run():
    forecast = analyzer.regr_models(model_=RandomForestRegressor,column='Sales',
                                connect=False,n_estimators=10)
    scores,forecast = forecast
    # print(scores)
    # mlflow.log_metric("scores",scores)
    