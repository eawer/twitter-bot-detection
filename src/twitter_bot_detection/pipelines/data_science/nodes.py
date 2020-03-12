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
        # "cat_features": ["created_at_time"],
        # "task_type": "GPU",
        #"text_features": ["description", "screen_name", "name", "location"],
    }
    model = CatBoostClassifier(**params)

#     data = Pool(data=X_train, label=y_train, cat_features=params["cat_features"])

#     params = {"iterations": 100,
#               "depth": 2,
#               "loss_function": "Logloss",
#               "verbose": False}
# 
#     scores = cv(data, params, fold_count=3, verbose=200, plot="True", stratified=True)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=200,
        plot=False,
    )
    
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average="weighted")        
    model.save_model('data/06_models/catboost', format="cbm", export_parameters=None, pool=None)
    
    print(classification_report(y_test, y_pred, digits=5))
    if log:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/firefly.eugene@gmail.com/twitter-bot-detection")

        run_id = mlflow.search_runs(experiment_ids="3889491181315524", filter_string="tags.`mlflow.runName`='catboost'", run_view_type=1)["run_id"][0]
        mlflow.start_run(run_id=run_id, nested=False)
        with mlflow.start_run(nested=True):
            mlflow.set_tags({
                "lib": "catboost",
                "description": "basic tweets features",
                "features": X_train.columns.values,
            })

            mlflow.log_params(params)
            mlflow.log_metric("f1", f1, 1)
            mlflow.log_artifact('data/05_model_input/X_test.pkl')
        mlflow.end_run()

    return model


