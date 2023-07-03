import logging
from functools import wraps
import os
import pickle

logger = logging.getLogger(__name__)


def cached(path: str):
    """
    Use local pickelized object if available. Else, compute the
    decorated function
    """

    def decorator_repeat(func):
        @wraps(func)
        def wrapper_cached(*args, **kwargs):
            if os.path.isfile(f"{path}.pickle"):
                logger.info("loading object from %s", f"{path}.pickle")
                df = pickle.load(f"{path}.pickle")
            else:
                logger.info("computing from %s", func.__name__)
                df = func(*args, **kwargs)

            return df

        return wrapper_cached

    return decorator_repeat
