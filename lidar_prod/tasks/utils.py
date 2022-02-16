from dataclasses import dataclass
import re
from functools import wraps
from tempfile import TemporaryDirectory
import numpy as np


@dataclass
class BDUniConnectionParams:
    """URL and public credentials to connect to a database - typically the BDUni"""

    host: str
    user: str
    pwd: str
    bd_name: str


def split_idx_by_dim(dim_array):
    """Returns a sequence of arrays of indices of elements sharing the same value in dim_array"""
    idx = np.argsort(dim_array)
    sorted_dim_array = dim_array[idx]
    group_idx = np.array_split(idx, np.where(np.diff(sorted_dim_array) != 0)[0] + 1)
    return group_idx


def extract_coor(las_name: str, x_span: float, y_span: float, buffer: float):
    """
    Extract the X-Y coordinates from standard LAS name, and returns
    a bounding box with a specified spans plus a buffer.
    """
    # get the values with [4,10] digits in the file name
    x_min, y_max = re.findall(r"[0-9]{4,10}", las_name)
    x_min, y_max = int(x_min), int(y_max)
    return (
        x_min - buffer,
        y_max - y_span - buffer,
        x_min + x_span + buffer,
        y_max + buffer,
    )


def tempdir(**kw):
    """
    A decorator for building a temporary directory. The temproary directory is created
    just before the wrapped function is called and is destroyed imediately after the wrapped function returns.
    The functions should have temporary_dict as its first
    Inspired from https://gist.github.com/waylan/50a067d79386aabf9073dbab7beb2950
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with TemporaryDirectory(**kw) as td:
                kwargs.update({"temporary_dir": td})
                return fn(*args, **kwargs)

        return wrapper

    return decorator
