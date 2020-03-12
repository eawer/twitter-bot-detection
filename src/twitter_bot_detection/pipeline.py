"""Pipeline construction."""

from typing import Dict

from kedro.pipeline import Pipeline
from twitter_bot_detection.pipelines.data_engineering import pipeline as de
from twitter_bot_detection.pipelines.feature_engineering import pipeline as fe
from twitter_bot_detection.pipelines.data_science import pipeline as ds


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
    fe_pipeline = fe.create_pipeline()
    ds_pipeline = ds.create_pipeline()

    return {
        "de": de_pipeline,
        "ds": ds_pipeline,
        "__default__": Pipeline([de_pipeline, fe_pipeline, ds_pipeline])
    }
