from dataclasses import dataclass
from typing import Union
import json
import math
import logging
from numbers import Number
from typing import Any, Dict, Iterable
import numpy as np
import pdal
import laspy
import subprocess
import geopandas

log = logging.getLogger(__name__)


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


def get_pipeline(input_value: Union[pdal.pipeline.Pipeline, str]):
    """If the input value is a pipeline, returns it, if it's a las path return the corresponding pipeline"""
    if type(input_value) == str:
        pipeline = pdal.Pipeline() | get_pdal_reader(input_value)
        pipeline.execute()
    else:
        pipeline = input_value
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


def request_bd_uni_for_building_shapefile(
    bd_params: BDUniConnectionParams,
    shapefile_path: str,
    bbox: Dict[str, int],
):
    """BD Uni request.

    Create a shapefile with non destructed building on the area of interest
    and saves it.
    Also add a "PRESENCE" column filled with 1 for later use by pdal.

    """
    Lambert_93_SRID = 2154
    sql_request = f'SELECT \
        st_setsrid(batiment.geometrie,{Lambert_93_SRID}) AS geometry, \
        1 as presence \
        FROM batiment \
        WHERE batiment.geometrie \
            && \
        ST_MakeEnvelope({bbox["x_min"]}, {bbox["y_min"]}, {bbox["x_max"]}, {bbox["y_max"]}, {Lambert_93_SRID}) \
        and \
        not gcms_detruit'
    cmd = [
        "pgsql2shp",
        "-f",
        shapefile_path,
        "-h",
        bd_params.host,
        "-u",
        bd_params.user,
        "-P",
        bd_params.pwd,
        bd_params.bd_name,
        sql_request,
    ]
    # This call may yield
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # In empty zones, pgsql2shp does not create a shapefile
        if (
            e.output
            == b"Initializing... \nERROR: Could not determine table metadata (empty table)\n"
        ):
            return False
        # Error can be due to something else entirely, like
        # an inability to translate host name to an address.
        # e.g. "could not translate host name "serveurbdudiff.ign.fr" to address: System error"
        raise e
    except ConnectionRefusedError as e:
        log.error(
            "ConnectionRefusedError when requesting BDUni.  \
            This means that the Database cannot be accessed (e.g. due to vpn/proxy reasons, \
            or bad credentials)"
        )
        raise e
    except TimeoutError as e:
        log.error(
            "TimeoutError when requesting BDUni"
        )
        raise e

    # read & write to avoid unnacepted 3D shapefile format.
    gdf = geopandas.read_file(shapefile_path)
    gdf[["PRESENCE", "geometry"]].to_file(shapefile_path)

    return True
