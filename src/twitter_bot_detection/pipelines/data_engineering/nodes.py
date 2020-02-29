import time
import logging
import pandas as pd
from functools import wraps
from typing import Callable
from pyspark import SparkContext, SQLContext, sql
from kedro.contrib.io.pyspark import SparkDataSet
from kedro.io import ParquetLocalDataSet, CSVLocalDataSet, PickleLocalDataSet

from twitter_bot_detection.helpers import log_running_time, extract_urls


@log_running_time
def label_users(users: CSVLocalDataSet, labels: PickleLocalDataSet) -> PickleLocalDataSet:
    users.set_index('id_str', inplace=True)

    labels = labels.loc[~labels.index.duplicated(keep='first')]
    labels.index = labels.index.astype(str)
    
    df = pd.concat([users, labels], axis=1, join='inner')
    df.index.name = 'id_str'
    return df

@log_running_time
def prepare_users(users: PickleLocalDataSet) -> PickleLocalDataSet:
    users = users.copy()
    deprecated = ["utc_offset", "time_zone", "lang", "geo_enabled", "following", "follow_request_sent", "has_extended_profile", "notifications", "profile_location", "contributors_enabled", "profile_image_url", "profile_background_color", "profile_background_image_url", "profile_background_image_url_https", "profile_background_tile", "profile_link_color", "profile_sidebar_border_color", "profile_sidebar_fill_color", "profile_text_color", "profile_use_background_image", "is_translator", "is_translation_enabled", "translator_type"]
    
    #extracting date of last tweet
    users["last_status_date"] = pd.to_datetime(users["status"].apply(lambda x: None if pd.isnull(x) else x["created_at"]))
    
    #extracting expanded profile url and urls from the description 
    users[["profile_url", "description_urls"]] = users["entities"].apply(extract_urls)
    
    #removing old accounts with no tweets publicly available
    users = users[~users["last_status_date"].isnull()]

    users.drop(columns=deprecated + ['id', 'status', 'entities', 'url'], inplace=True)

    return users


@log_running_time
def extract_user_features(users: PickleLocalDataSet) -> PickleLocalDataSet:
    features = users[[
        "protected", "followers_count", "friends_count",  "listed_count", "favourites_count", "statuses_count",
        "verified", "default_profile", "default_profile_image",
    ]].copy()

    features["has_location"] = users.location.isnull()
    features["has_description"] = users.description.isnull()
    features["created_at_time"] = pd.cut(users["created_at"].dt.hour, [-1, 6, 11, 18, 23], labels=["night", "morning", "day", "evening"])
    features["account_active_for_days"] = (users["last_status_date"] - users["created_at"]).dt.days
    features["has_banner"] = ~users["profile_banner_url"].isnull()
    features["has_profile_url"] = ~users["profile_url"].isnull()
    features["description_urls_count"] = users["description_urls"].str.len()
    features["label"] = (users["label"] == 'bot').astype(int)

    features["tweets_per_day"] = users["statuses_count"] / features["account_active_for_days"]
    features["favourites_per_day"] = users["favourites_count"] / features["account_active_for_days"]
    features["fr_to_flw_ratio"] = users["friends_count"] / users["followers_count"]
    features["faw_to_tweets_ratio"] = users["favourites_count"] / users["statuses_count"]
    features["tweets_to_faw_ratio"] = users["statuses_count"] / users["favourites_count"]
    # features["listed_count_cat"] = pd.cut(users["listed_count"], [-1, 100, 500, 2000, 5000, 10000, float('inf')], labels=['100', '500', '2000', '5000', '10000', 'inf']).astype(str)

    return features


# sc = SparkContext()
# sqlContext = SQLContext(sc)

# tweets = sqlContext.read.json('./data/01_raw/tweets.jsonl')

# print(tweets.map(lambda x: x.user.id_str))

def convert_to_parquet(tweets: SparkDataSet) -> ParquetLocalDataSet:
    return tweets.coalesce(1)

#     return companies

