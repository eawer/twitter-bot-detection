import pandas as pd
import time
import logging
from functools import wraps
from typing import Callable


def extract_urls(entities):
    profile_url = None
    if "url" in entities:
        url = entities["url"]["urls"][0]
        profile_url = url["expanded_url"] if url["expanded_url"] else url["url"] 
    description_urls = [url["expanded_url"] for url in entities["description"]["urls"] if url]
    
    return pd.Series([profile_url, description_urls])


def log_running_time(func: Callable) -> Callable:
    """Decorator for logging node execution time.

        Args:
            func: Function to be executed.

        Returns:
            Decorator for logging the running time.

    """

    @wraps(func)
    def with_time(*args, **kwargs):
        log = logging.getLogger(__name__)
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        elapsed = t_end - t_start
        log.info("Running %r took %.2f seconds", func.__name__, elapsed)
        return result

    return with_time