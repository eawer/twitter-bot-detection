from kedro.pipeline import node, Pipeline
from twitter_bot_detection.pipelines.feature_engineering.nodes import (
    extract_user_features, 
    extract_main_tweets_features, 
    get_dominating_hour_share,
    split_dataset
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=extract_user_features,
                inputs="prepared_users",
                outputs="user_features",
                name="Extracting user features",
            ),
            node(
                func=extract_main_tweets_features,
                inputs="tweets",
                outputs="tweets_features",
                name="Extracting main tweets' features",
            ),
            node(
                func=split_dataset,
                inputs=["user_features", "tweets_features", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="Splitting the dataset",
            ),
        ],
    )
