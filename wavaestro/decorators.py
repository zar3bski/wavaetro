import logging
from functools import wraps
import os
import pickle

logger = logging.getLogger(__name__)


def cached(func):
    """
    Use local pickelized object if available. Else, compute the
    decorated function
    """

    #    def decorator_cached(func):
    # @wraps(func)
    def wrapper_cached(*args, **kwargs):
        path = kwargs.get("path")
        wavelet_name = kwargs.get("wavelet_name")
        if os.path.isfile(f"{path}_{wavelet_name}.pickle"):
            logger.info("loading object from %s", f"{path}_{wavelet_name}.pickle")
            with open(f"{path}_{wavelet_name}.pickle", "rb") as f1:
                df = pickle.load(f1)
        else:
            logger.info("computing from %s", func.__name__)
            df = func(*args, **kwargs)
            with open(f"{path}_{wavelet_name}.pickle", "wb") as f1:
                pickle.dump(df, f1)
        return df

    return wrapper_cached


#    return decorator_cached
