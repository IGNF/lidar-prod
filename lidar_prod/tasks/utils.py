from dataclasses import dataclass
import json
import os
import re
from functools import wraps
import subprocess
from tempfile import TemporaryDirectory
import tempfile
from typing import List, Union
import numpy as np
import pdal
from sqlalchemy import Numeric


@dataclass
class BDUniConnectionParams:
    """URL and public credentials to connect to a database - typically the BDUni"""

    host: str
    user: str
    pwd: str
    bd_name: str


def split_idx_by_dim(dim_array):
    """
    Returns a sequence of arrays of indices of elements sharing the same value in dim_array
    Groups are ordered by ascending value.
    """
    idx = np.argsort(dim_array)
    sorted_dim_array = dim_array[idx]
    group_idx = np.array_split(idx, np.where(np.diff(sorted_dim_array) != 0)[0] + 1)
    return group_idx


def run_pdal_info(in_las: str, out_stats_json: str):
    command = (
        f"pdal info {in_las}"
        " --metadata"
        " --driver readers.las"
        f" > {out_stats_json}"
    )
    subprocess.run(command, shell=True, check=True)


def get_las_metadata(in_las):
    _tmp = tempfile.NamedTemporaryFile().name
    run_pdal_info(in_las, _tmp)
    with open(_tmp) as mtd:
        metadata = json.load(mtd)["metadata"]
    return metadata


def get_bbox(in_las: str, buffer: int = 0):
    """Get XY bounding box of a cloud using pdal info --metadata.

    Args:
        in_las (str): path to input LAS cloud.
        buffer (int): expand bbox with a buffer. Default: no buffer.

    Returns:
        float: coordinates of bounding box : xmin, ymin, xmax, ymax

    """

    metadata = get_las_metadata(in_las)
    return {
        "x_min": metadata["minx"] - buffer,
        "y_min": metadata["miny"] - buffer,
        "x_max": metadata["maxx"] + buffer,
        "y_max": metadata["maxy"] + buffer,
    }
