from dataclasses import dataclass
from numbers import Number
import os
from pathlib import Path
import subprocess
import tempfile
import json
import math
import numpy as np
from typing import Any, Dict, Iterable
from osgeo import gdal, ogr, osr
import pdal

LAMBERT_93_SRID = 2154
LAMBERT_93_EPSG_STR = f"EPSG:{LAMBERT_93_SRID}"


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


def get_bbox(las_path: str, buffer: int = 0) -> Dict[str, int]:
    """Get XY bounding box of a cloud using pdal info --metadata.

    Args:
        las_path (str): path to input LAS cloud.
        buffer (int): expand bbox with a buffer. Default: no buffer.

    Returns:
        float: coordinates of bounding box : xmin, ymin, xmax, ymax

    """

    metadata = get_las_metadata(las_path)
    return {
        "x_min": math.floor(metadata["minx"] - buffer),
        "y_min": math.floor(metadata["miny"] - buffer),
        "x_max": math.ceil(metadata["maxx"] + buffer),
        "y_max": math.ceil(metadata["maxy"] + buffer),
    }


def get_integer_bbox(las_path: str, buffer: Number = 0) -> Dict[str, int]:
    """Get XY bounding box of a cloud, cast x/y min/max to integers."""

    bbox = get_bbox(las_path, buffer=buffer)
    return {
        "x_min": math.floor(bbox["x_min"]),
        "y_min": math.floor(bbox["y_min"]),
        "x_max": math.ceil(bbox["x_max"]),
        "y_max": math.ceil(bbox["y_max"]),
    }


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
        override_srs=LAMBERT_93_EPSG_STR,
    )


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


def stem(input_file: str):
    return Path(input_file).stem


# Rasterization and vectorization


def gdal_polygonize(
    fic_mask, fic_output, epsg_out=None, field_name="value", value_to_polygonize=0
):
    """Polygonisation des valeurs non-nulles d'un raster monobande."""

    src_ds = gdal.Open(fic_mask)
    srcband = src_ds.GetRasterBand(1)
    srcband2 = src_ds.GetRasterBand(1)

    srs = get_ogr_srs(epsg_out)
    ext_driver = get_ogr_driver_by_ext(fic_output)
    drv = ogr.GetDriverByName(ext_driver)

    if os.path.exists(fic_output):
        drv.DeleteDataSource(fic_output)

    dst_datasource = drv.CreateDataSource(fic_output)
    dst_layer = dst_datasource.CreateLayer(fic_output, srs)
    field_defn = ogr.FieldDefn(field_name, ogr.OFTString)
    dst_layer.CreateField(field_defn)

    gdal.Polygonize(
        srcband, srcband2, dst_layer, value_to_polygonize, ["8"], callback=None
    )


def get_ogr_srs(epsg):
    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        return srs
    else:
        return None


def get_ogr_driver_by_ext(file):
    _, file_extension = os.path.splitext(file)
    if file_extension == ".json" or file_extension == ".geojson":
        return "GeoJson"
    elif file_extension == ".shp":
        return "ESRI Shapefile"
    else:
        raise Exception("Extension de fichier vecteur non comprise")
        return 0
