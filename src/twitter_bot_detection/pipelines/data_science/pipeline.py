from kedro.pipeline import node, Pipeline
from twitter_bot_detection.pipelines.data_science.nodes import train_catboost


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # node(
            #     func=train_xgboost,
            #     inputs="user_features",
            #     outputs="xgboost_model",
            #     name="Training xgboost classifier",
            # ),
            node(
                func=train_catboost,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs="catboost_model",
                name="Training catboost classifier",
            ),
        ],
    )
