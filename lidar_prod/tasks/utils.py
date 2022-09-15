from dataclasses import dataclass
from typing import Union
import json
import math
from numbers import Number
from typing import Any, Dict, Iterable
import numpy as np
import laspy
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


def get_pipeline(entry_value: Union[pdal.pipeline.Pipeline, str]):
    if type(entry_value) == str:
        pipeline = pdal.Pipeline() | get_pdal_reader(entry_value)
        pipeline.execute()
    else:
        pipeline = entry_value
    return pipeline


def get_las_metadata(entry_value: Union[pdal.pipeline.Pipeline, str]):
    pipeline = get_pipeline(entry_value)
    return json.loads(pipeline.metadata)["metadata"]["readers.las"]


def get_integer_bbox(entry_value: Union[pdal.pipeline.Pipeline, str], buffer: Number = 0) -> Dict[str, int]:
    pipeline = get_pipeline(entry_value)
    """Get XY bounding box of a cloud, cast x/y min/max to integers."""
    metadata = get_las_metadata(pipeline)
    bbox = {
        "x_min": math.floor(metadata["minx"] - buffer),
        "y_min": math.floor(metadata["miny"] - buffer),
        "x_max": math.ceil(metadata["maxx"] + buffer),
        "y_max": math.ceil(metadata["maxy"] + buffer),
    }
    return bbox


def get_pdal_reader(las_path: str) -> pdal.Reader.las:
    """Standard Reader which imposes Lamber 93 SRS.

    Args:
        las_path (str): input LAS path to read.

    Returns:
        pdal.Reader.las: reader to use in a pipeline.

    """
    return pdal.Reader.las(
        filename=las_path,
        nosrs=True,
        override_srs="EPSG:2154",
    )


def get_las_data_from_las(las_path: str) -> laspy.lasdata.LasData:
    """ Load las data from a las file """
    return laspy.read(las_path)


def get_pdal_writer(target_las_path: str, extra_dims: str = "all") -> pdal.Writer.las:
    """Standard LAS Writer which imposes LAS 1.4 specification and dataformat 8.

    Args:
        target_las_path (str): output LAS path to write.
        extra_dims (str): extra dimensions to keep, in the format expected by pdal.Writer.las.

    Returns:
        pdal.Writer.las: writer to use in a pipeline.

    """
    return pdal.Writer.las(
        filename=target_las_path,
        minor_version=4,
        dataformat_id=8,
        forward="all",
        extra_dims=extra_dims,
    )


def save_las_data_to_las(las_path: str, las_data: laspy.lasdata.LasData):
    """ save las data to a las file"""
    las_data.write(las_path)


def get_a_las_to_las_pdal_pipeline(
    src_las_path: str, target_las_path: str, ops: Iterable[Any]
):
    """Create a pdal pipeline, preserving format, forwarding every dimension.

    Args:
        src_las_path (str): input LAS path
        target_las_path (str): output LAS path
        ops (Iterable[Any]): list of pdal operation (e.g. Filter.assign(...))

    """
    pipeline = pdal.Pipeline()
    pipeline |= get_pdal_reader(src_las_path)
    for op in ops:
        pipeline |= op
    pipeline |= get_pdal_writer(target_las_path)
    return pipeline


def pdal_read_las_array(las_path: str):
    """Read LAS as a named array.

    Args:
        las_path (str): input LAS path

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.
    """
    p1 = pdal.Pipeline() | get_pdal_reader(las_path)
    p1.execute()
    return p1.arrays[0]
