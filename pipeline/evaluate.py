# evaluation script
import pandas as pd
import os,sys

from xgboost import XGBClassifier
sys.path.append(os.path.abspath(os.path.join('scripts/')))
print(os.getcwd())
from modeling import Modeler
from mlflow import log_metric, log_param,log_artifact,log_metrics,sklearn,start_run
from random import random, randint
# evaluate a logistic regression model using k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB


if __name__=='__main__':
    file_path = sys.argv[2]
    df = pd.read_csv(file_path)
    model_=Modeler(df)
    model = model_.get_model()
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    score,predictions=model_.regr_models(model_=LogisticRegression,column="yes",inputs=None,
                connect=False,serialize=False)
    metrics = {"score": score}
    results = pd.DataFrame([predictions])
    results.to_csv("data/results.csv")
    results.to_csv("data/predictions.csv")
    log_artifact("data/predictions.csv")
    log_metrics(metrics)
    sklearn.log_model(model, "model")


