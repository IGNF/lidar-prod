import logging
import os
import os.path as osp
from typing import Iterable, Optional, Union
import pdal
import numpy as np
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
        # turn a listconfig into a 'normal' list
        self.extra_dims = [extra_dims] if isinstance(extra_dims, str) else [dimension for dimension in extra_dims]  

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

    def remove_dimensions(self, las_data: laspy.lasdata.LasData):
        """remove dimension from (laspy) data"""
        # if we want to keep all dimension, we do nothing
        if self.extra_dims == ['all']:
            return 
        
        # selecting dimensions to remove
        extra_dim_no_type = [dimension.split('=')[0] for dimension in self.extra_dims]  # removing dimension type (int, float, etc.)
        dimension_to_remove = []
        for dimension in las_data.point_format.extra_dimension_names:
            if dimension not in extra_dim_no_type:
                dimension_to_remove.append(dimension)

        # case 0 dimension to remove
        if not dimension_to_remove:
            return

        # case 1 dimension to remove
        if len(dimension_to_remove) == 1:
            las_data.remove_extra_dim(dimension_to_remove[0])
            return

        # case 2+ dimensions to remove
        las_data.remove_extra_dims(dimension_to_remove)
