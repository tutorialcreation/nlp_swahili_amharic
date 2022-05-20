import pandas as pd
import os,sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.modeling import Modeler

from mlflow import log_metric, log_param, log_artifacts,get_experiment_by_name
from random import random, randint

# evaluate a logistic regression model using k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset

if __name__=='__main__':
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    file_path = sys.argv[2]
    df = pd.read_csv(file_path)
    model_=Modeler(df)
    X,y =model_.get_columns()
    cv = KFold(n_splits=fold, random_state=1, shuffle=True)
    df.to_csv("../data/validation.csv")
    log_param("fold", fold)
    log_metric("folds", cv.get_n_splits())
    log_artifacts("outputs")
    