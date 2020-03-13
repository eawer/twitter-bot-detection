import joblib
import catboost as ctb

from kedro.io import PickleLocalDataSet
from catboost import CatBoostClassifier, cv, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score

from twitter_bot_detection.helpers import log_running_time


@log_running_time
def train_catboost(X_train: PickleLocalDataSet, X_test: PickleLocalDataSet, y_train: PickleLocalDataSet, y_test: PickleLocalDataSet, log=False) -> PickleLocalDataSet:
    params = {
        "iterations": 2500,
        "learning_rate": 0.02,
        "loss_function": 'Logloss',
        "random_seed": 1,
        "od_wait": 30,
        "od_type": "Iter",
        "thread_count": 8,
    }
    model = CatBoostClassifier(**params)
    features = X_train.columns.values

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=200,
        plot=False,
    )
    
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average="weighted")        
    joblib.dump(model, 'data/06_models/catboost.pkl')

    print(classification_report(y_test, y_pred, digits=5))
    
    return model
