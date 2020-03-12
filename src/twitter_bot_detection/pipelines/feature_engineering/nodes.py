import vaex as vx
import pandas as pd
import numpy as np
np.random.seed(1)
from sklearn.model_selection import train_test_split
from kedro.io import CSVLocalDataSet, PickleLocalDataSet

from twitter_bot_detection.io.vaex_hdf5 import VaexHDF5DataSet
from twitter_bot_detection.helpers import log_running_time

def count_char_types(string):
    flags = [0, 0, 0, 0]
    for c in string:
        if c.islower():
            flags[0] = 1
        elif c.isupper():
            flags[1] = 1
        elif c.isdigit():
            flags[2] = 1
        else:
            flags[3] = 1
        
    return sum(flags)

@log_running_time
def extract_user_features(users: PickleLocalDataSet) -> PickleLocalDataSet:
    features = users[[
        "protected", "followers_count", "friends_count",  "listed_count", "favourites_count", "statuses_count",
        "verified", "default_profile", "default_profile_image",
    ]].copy()

    features["protected"] = features["protected"].astype(int)
    features["verified"] = features["verified"].astype(int)
    features["default_profile"] = features["default_profile"].astype(int)
    features["default_profile_image"] = features["default_profile_image"].astype(int)


    features["char_types"] = users["screen_name"].apply(count_char_types)
    features["has_location"] = (users.location != '').astype(int)
    features["has_description"] = (users.description != '').astype(int)
    features["created_at_time"] = pd.cut(users["created_at"].dt.hour, [-1, 6, 11, 18, 23], labels=["night", "morning", "day", "evening"])
    features["account_active_for_days"] = (users["last_status_date"] - users["created_at"]).dt.days
    features["has_banner"] = ~users["profile_banner_url"].isnull().astype(int)
    features["has_profile_url"] = ~users["profile_url"].isnull().astype(int)
    features["description_urls_count"] = users["description_urls"].str.len()
    features["label"] = (users["label"] == 'bot').astype(int)

    features["tweets_per_day"] = (users["statuses_count"] + 0.001) / (features["account_active_for_days"] + 0.001)
    features["favourites_per_day"] = (users["favourites_count"] + 0.001) / (features["account_active_for_days"] + 0.001)
    features["fr_to_flw_ratio"] = (users["friends_count"] + 0.001) / (users["followers_count"] + 0.001)
    features["faw_to_tweets_ratio"] = (users["favourites_count"] + 0.001) / (users["statuses_count"] + 0.001)
    features["tweets_to_faw_ratio"] = (users["statuses_count"] + 0.001) / (users["favourites_count"] + 0.001)
    # features["listed_count_cat"] = pd.cut(users["listed_count"], [-1, 100, 500, 2000, 5000, 10000, float('inf')], labels=['100', '500', '2000', '5000', '10000', 'inf']).astype(str)

    features = pd.get_dummies(features, columns=["created_at_time"])

    return features


@log_running_time
def extract_main_tweets_features(tweets: VaexHDF5DataSet) -> PickleLocalDataSet:
    features = tweets.groupby(by=tweets["user_id"], agg={
        "tweets": vx.agg.count("id"),
        "unique_sources": vx.agg.nunique("source"),
        "replies": [vx.agg.mean("is_reply")],
        "quotes": [vx.agg.mean("is_quote")],
        "hashtags": [vx.agg.mean("hashtags_count"), vx.agg.std("hashtags_count")],
        "mentions": [vx.agg.mean("mentions_count"), vx.agg.std("mentions_count")],
        "urls": [vx.agg.mean("urls_count"), vx.agg.std("urls_count")],
        "symbols": [vx.agg.mean("symbols_count"), vx.agg.std("symbols_count")],
        "sensitive": [vx.agg.mean("sensitive")],
        "truncated": [vx.agg.mean("truncated")],
        "langs": [vx.agg.nunique("lang")],
        "is_retweet": [vx.agg.mean("is_retweet")],
        #retweeted_author
        "media": [vx.agg.mean("media_count"), vx.agg.max("media_count"), vx.agg.std("media_count")],
                

#         "retweeted_by": [vx.agg.mean("retweet_count"), vx.agg.max("retweet_count"), vx.agg.std("retweet_count")],
#         "favorited_by": [vx.agg.mean("favorite_count"), vx.agg.max("favorite_count"), vx.agg.std("favorite_count")],
# #         "reply_to_tweet_ratio": vx.agg.sum("is_reply") / vx.agg.count("is_reply"),
#         "reply_to_unique": vx.agg.nunique("reply_to"),
#         "quotes_count": vx.agg.sum("is_quote"),
    })
    features = features.to_pandas_df()
    features.set_index("user_id", inplace=True)
    
    # features["dominating_hour"] = get_dominating_hour(tweets)
    # features["dominating_day_of_week"] = get_dominating_day_of_week(tweets)
    
    # features["dominating_hour_share"] = get_dominating_hour_tweets(tweets) / features["tweets"] 
    # features["dominating_dayofweek_share"] = get_dominating_dayofweek_tweets(tweets) / features["tweets"]
    return features.drop(columns=["tweets"])


def get_dominating_dayofweek_tweets(df):
    temp = df.groupby(['user_id', "dayofweek"], 'count')
    temp = temp.groupby(['user_id'], agg={
        "max": vx.agg.max("count")
    })
    temp = temp.to_pandas_df()
    temp.set_index("user_id", inplace=True)
    return temp["max"]

def get_dominating_hour_tweets(tweets: VaexHDF5DataSet):
    temp = tweets.groupby(['user_id', "hour"], 'count')
    temp = temp.groupby(['user_id'], agg={
        "max": vx.agg.max("count")
    })
    temp = temp.to_pandas_df()
    temp.set_index("user_id", inplace=True)
    
    return temp["max"]



def get_dominating_hour_share(tweets: VaexHDF5DataSet, tweets_features: PickleLocalDataSet) -> PickleLocalDataSet:
    temp = tweets.groupby(['user_id', "hour"], 'count')
    temp = temp.groupby(['user_id'], agg={
        "max_hour": vx.agg.max("count")
    })
    temp = temp.to_pandas_df()
    temp.set_index("user_id", inplace=True)
    tweets_features["dominating_hour_share"] = temp["max_hour"] / tweets_features["tweets"] 
    
    return tweets_features



def get_dominating_hour(df):
    temp = df.groupby(['user_id', "hour"], 'count')
    temp = temp.to_pandas_df()
    temp = temp.loc[temp.groupby(['user_id'])['count'].idxmax()].drop(columns=["count"])

    return temp.set_index("user_id")

def get_dominating_day_of_week(df):
    temp = df.groupby(['user_id', "dayofweek"], 'count')
    temp = temp.to_pandas_df()
    temp = temp.loc[temp.groupby(['user_id'])['count'].idxmax()].drop(columns=["count"])
    return temp.set_index("user_id")

@log_running_time
def split_dataset(user_features: PickleLocalDataSet, tweets_features: PickleLocalDataSet, test_size: str, random_state: int) -> [PickleLocalDataSet, PickleLocalDataSet, PickleLocalDataSet, PickleLocalDataSet]:
    df = pd.concat([user_features, tweets_features], axis=1, join='inner')
    df.fillna(0, inplace=True)
    # df.fillna(df.mean(), inplace=True)
    X = df.drop(columns=["label"])
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)