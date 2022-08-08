import logging
import os
import os.path as osp
from typing import Iterable, Optional, Union
import pdal
import numpy as np
import numpy.lib.recfunctions as rfn
import laspy

from lidar_prod.tasks.utils import get_pdal_reader, get_pdal_writer

log = logging.getLogger(__name__)


class Cleaner:
    """Keep only necessary extra dimensions channels."""

    def __init__(self, extra_dims: Optional[Union[Iterable[str], str]]):
        """Format extra_dims parameter from config.

        Args:
            extra_dims (Optional[Union[Iterable[str], str]]): each dim should have format dim_name:pdal_type.
            If a string, used directly; if an iterable, dimensions are joined together.

        """
        self.extra_dims = [dimension for dimension in extra_dims] # turn a listconfig into a 'normal' list

    def get_extra_dims_as_str(self):
        """ 'stringify' the extra_dims list and return it, or an empty list if there is no extra dims"""
        return_str = ",".join(self.extra_dims)
        return return_str if return_str else []

    def run(self, src_las_path: str, target_las_path: str):
        """Clean out LAS extra dimensions.

        Args:
            src_las_path (str): input LAS path
            target_las_path (str): output LAS path, with specified extra dims.
        """

        pipeline = pdal.Pipeline()
        pipeline |= get_pdal_reader(src_las_path)
        pipeline |= get_pdal_writer(target_las_path, extra_dims=self.get_extra_dims_as_str())
        pipeline.execute()
        os.makedirs(osp.dirname(target_las_path), exist_ok=True)
        log.info(f"Saved to {target_las_path}")

    def remove_unwanted_dimensions(self, points: np.ndarray):
        """remove the dimensions we don't want to keep in the points array"""
        # list of dimension that are, by default, in a las file 
        default_pdal_dimension_list = [
            'X', 'Y', 'Z', 
            'Intensity', 
            'ReturnNumber', 
            'NumberOfReturns', 
            'ScanDirectionFlag', 
            'EdgeOfFlightLine', 
            'Classification', 
            'ScanAngleRank', 
            'UserData', 
            'PointSourceId', 
            'GpsTime', 
            'Red', 'Green', 'Blue', 'Infrared', 
            'ScanChannel', 
            'ClassFlags'
            ]

        # substract unwanted dimensions from the dimensions from the points array to get the kept dimensions
        dimensions_to_keep = [dimension for dimension in points.dtype.names]
        extra_dim_no_type = [dimension.split('=')[0] for dimension in self.extra_dims] # removing the type of the dimension (int, float, etc.)
        for dimension in reversed(dimensions_to_keep): # reversed because we may remove some dimension, therefore the list may change
            if dimension not in default_pdal_dimension_list + extra_dim_no_type:
                dimensions_to_keep.remove(dimension)

        # return a modified points array only if there are modifications to carry
        if len(dimensions_to_keep) != len(points.dtype.names):
            return points[dimensions_to_keep]
        return points

    def remove_dimensions(self, las_data: laspy.lasdata.LasData):
        """remove existing dimensions we don't want"""
        extra_dim_no_type = [dimension.split('=')[0] for dimension in self.extra_dims] # list of kept dimensions, without the type
        dimension_to_remove = []
        for dimension in las_data.point_format.extra_dimension_names:
            if dimension not in extra_dim_no_type:
                dimension_to_remove.append(dimension)
        
        if not dimension_to_remove:
            return
        
        if len(dimension_to_remove) == 1:
            las_data.remove_extra_dim(dimension_to_remove[0])
            return
        
        las_data.remove_extra_dims(dimension_to_remove)
