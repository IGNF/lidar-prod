from dataclasses import dataclass
import json
import subprocess
import tempfile
from typing import Any, Iterable
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


def get_pdal_reader(in_f: str) -> pdal.Reader.las:
    """Standard Reader which imposes Lamber 93 SRS.

    Args:
        in_f (str): input LAS path to read.

    Returns:
        pdal.Reader.las: reader to use in a pipeline.

    """
    return pdal.Reader.las(
        filename=in_f,
        nosrs=True,
        override_srs="EPSG:2154",
    )


def get_pdal_writer(out_f: str, extra_dims: str = "all") -> pdal.Writer.las:
    """Standard LAS Writer which imposes LAS 1.4 specification and dataformat 8.

    Args:
        in_f (str): output LAS path to write.
        extra_dims (str): extra dimensions to keep, in the format expected by pdal.Writer.las.

    Returns:
        pdal.Writer.las: writer to use in a pipeline.

    """
    return pdal.Writer.las(
        filename=out_f,
        minor_version=4,
        dataformat_id=8,
        forward="all",
        extra_dims=extra_dims,
    )


def get_a_las_to_las_pdal_pipeline(in_f: str, out_f: str, ops: Iterable[Any]):
    """Create a pdal pipeline, preserving format, forwarding every dimension.

    Args:
        in_f (str): input LAS path
        out_f (str): output LAS path
        ops (Iterable[Any]): list of pdal operation (e.g. Filter.assign(...))

    """
    pipeline = pdal.Pipeline()
    pipeline |= get_pdal_reader(in_f)
    for op in ops:
        pipeline |= op
    pipeline |= get_pdal_writer(out_f)
    return pipeline


def pdal_read_las_array(in_f: str):
    """Read LAS as a named array.

    Args:
        in_f (str): input LAS path

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.
    """
    p1 = pdal.Pipeline() | get_pdal_reader(in_f)
    p1.execute()
    return p1.arrays[0]
