from functools import wraps
import pandas as pd
import warnings


def ignore_warnings_decorator(category):
    def ignore_warnings(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(
                    action="ignore",
                    category=category,
                )
                return f(*args, **kwargs)

        return wrapper

    return ignore_warnings


ignore_performance_warnings = ignore_warnings_decorator(pd.errors.PerformanceWarning)
ignore_future_warnings = ignore_warnings_decorator(FutureWarning)
