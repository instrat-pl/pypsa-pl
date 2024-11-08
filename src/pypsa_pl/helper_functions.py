import os
import sys
import warnings
from contextlib import contextmanager
from functools import wraps
import pandas as pd


@contextmanager
def suppress_stdout():
    if ("google.colab" in sys.modules) or (os.name == "nt"):  # quick fix for Windows
        yield
    else:
        with open(os.devnull, "w") as devnull:
            oldstdout_fno = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), 1)
            try:
                yield
            finally:
                os.dup2(oldstdout_fno, 1)


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
ignore_user_warnings = ignore_warnings_decorator(UserWarning)
