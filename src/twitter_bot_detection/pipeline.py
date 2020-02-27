"""Pipeline construction."""

from typing import Dict

from kedro.pipeline import Pipeline
from twitter_bot_detection.pipelines.data_engineering import pipeline as de
# from twitter_bot_detection.pipelines import data_science as ds


# Here you can define your data-driven pipeline by importing your functions
# and adding them to the pipeline as follows:
#
# from nodes.data_wrangling import clean_data, compute_features
#
# pipeline = Pipeline([
#     node(clean_data, 'customers', 'prepared_customers'),
#     node(compute_features, 'prepared_customers', ['X_train', 'Y_train'])
# ])
#
# Once you have your pipeline defined, you can run it from the root of your
# project by calling:
#
# $ kedro run


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    de_pipeline = de.create_pipeline()
    return {
        "de": de_pipeline,
        "__default__": Pipeline([de_pipeline])
    }

