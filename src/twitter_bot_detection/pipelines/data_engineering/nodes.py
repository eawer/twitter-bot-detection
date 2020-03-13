import vaex as vx
import pandas as pd
from kedro.io import CSVLocalDataSet, PickleLocalDataSet

from twitter_bot_detection.io.vaex_hdf5 import VaexHDF5DataSet
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

