from kedro.pipeline import node, Pipeline
from twitter_bot_detection.pipelines.data_engineering.nodes import convert_to_parquet


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=convert_to_parquet,
                inputs="raw_test",
                outputs="parquet_test",
                name="test_to_parquet",
            ),
        ]
    )
