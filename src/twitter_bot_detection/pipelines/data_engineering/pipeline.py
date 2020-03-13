from kedro.pipeline import node, Pipeline
from twitter_bot_detection.pipelines.data_engineering.nodes import label_users, prepare_users

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=label_users,
                inputs=["raw_users", "labels"],
                outputs="labelled_users",
                name="Labelling users",
            ),
            node(
                func=prepare_users,
                inputs="labelled_users",
                outputs="prepared_users",
                name="Preparing users",
            ),
        ],
    )
