import pandas as pd
import os,sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.modeling import Modeler
from mlflow import log_metric, log_param, log_artifacts
from random import random, randint


if __name__=='__main__':
    df = pd.read_csv("../data/AdSmartABdata.csv")
    model_=Modeler(df)
    print("<<<<<<<<start pipeline>>>>>>>>")
    X_train, X_test, X_val, y_train, y_test,y_val = model_.split_data()
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)
    print(type(X_train))
    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    X_train.to_csv("outputs/train.csv")
    log_artifacts("outputs")
    print("<<<<<<<<<end pipeline>>>>>>>>>>>")
