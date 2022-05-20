import pandas as pd
import os,sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.modeling import Modeler
from mlflow import log_metric, log_param, log_artifact,set_experiment
from random import random, randint

# evaluate a logistic regression model using k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset

if __name__=='__main__':
    experiment_name="abtest"
    set_experiment(experiment_name)
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    model_=Modeler(df)
    X,y = model_.get_columns()
    os_6 = model_.groupby_column(column="platform_os",index=6)
    os_5 = model_.groupby_column(column="platform_os",index=5)
    browser_chrome_mobile_pipeline = model_.groupby_column(column="browser_Chrome Mobile",index=1)
    browser_chrome_mobile_web_pipeline = model_.groupby_column(column="browser_Chrome Mobile WebView",index=1)
    X_train, X_test, X_val,y_train, y_test,y_val = model_.split_data()
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    os_6.to_csv("outputs/os_6.csv")
    os_5.to_csv("outputs/os_5.csv")
    browser_chrome_mobile_pipeline.to_csv("outputs/browser_chrome_mobile_pipeline.csv")
    browser_chrome_mobile_web_pipeline.to_csv("outputs/browser_chrome_mobile_web_pipeline.csv")
    log_artifact("outputs/os_6.csv")
    log_artifact("outputs/os_5.csv")
    log_artifact("outputs/browser_chrome_mobile_pipeline.csv")
    log_artifact("outputs/browser_chrome_mobile_web_pipeline.csv")