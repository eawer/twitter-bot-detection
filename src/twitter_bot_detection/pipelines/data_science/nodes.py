import pandas as pd
import catboost as ctb
from kedro.io import PickleLocalDataSet
from catboost import CatBoostClassifier, cv, Pool
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from twitter_bot_detection.helpers import log_running_time


# @log_running_time
# def train_catboost(users: PickleLocalDataSet) -> PickleLocalDataSet:


@log_running_time
def train_catboost(users: PickleLocalDataSet) -> PickleLocalDataSet:
    X = users.drop(columns=["label"])
    y = users["label"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=1)

    params = {
        "iterations": 2000,
        "learning_rate": 0.02,
        "loss_function": 'Logloss',
        "random_seed": 1,
        "od_wait": 30,
        "od_type": "Iter",
        "thread_count": 8,
        "cat_features": ["created_at_time",]
    }
    model = CatBoostClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        verbose=200,
        plot=False,
    )
    
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average="weighted")
        
    print(classification_report(y_test, y_pred))

    return model


