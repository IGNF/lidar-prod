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


# TODO: replace with get_bbox below, with a buffer! Do that after tests.
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


def run_pdal_info(in_las: str, out_stats_json: str):
    command = f"pdal info {in_las} --metadata > {out_stats_json}"
    subprocess.run(command, shell=True, check=True)


def get_las_metadata(in_las):
    _tmp = tempfile.NamedTemporaryFile().name
    run_pdal_info(in_las, _tmp)
    with open(_tmp) as mtd:
        metadata = json.load(mtd)["metadata"]
    return metadata


def get_bbox(in_las: str):
    """Get XY bounding box of a cloud using pdal info --metadata.

    Args:
        in_las (str): path to input LAS cloud.

    Returns:
        float: coordinates of bounding box : xmin, ymin, xmax, ymax

    """

    metadata = get_las_metadata(in_las)
    return metadata["minx"], metadata["maxx"], metadata["miny"], metadata["maxy"]
