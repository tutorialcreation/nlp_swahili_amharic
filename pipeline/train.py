import pandas as pd
import os,sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.modeling import Modeler
from mlflow import log_metric, log_param, log_artifacts,set_experiment
from random import random, randint

# evaluate a logistic regression model using k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset

if __name__=='__main__':
    experiment_name="abtest"
    set_experiment(experiment_name)
    df = pd.read_csv("../data/AdSmartABdata.csv")
    model_=Modeler(df)
    X_train, X_test, X_val,y_train, y_test,y_val = model_.split_data()
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    X_train.to_csv("outputs/train.csv")
    X_val.to_csv("outputs/validation.csv")
    log_artifacts("outputs")
    