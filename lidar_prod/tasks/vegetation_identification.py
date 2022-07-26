"""
Takes vegetation probabilities as input, and defines vegetation

"""
from dataclasses import dataclass
import logging
import os
from typing import List
import numpy as np

import pdal

from lidar_prod.tasks.utils import get_pdal_reader, get_pdal_writer


log = logging.getLogger(__name__)


class VegetationIdentifier:

    def __init__(self, vegetation_thresholds: float,  unclassified_thresholds: float, data_format):
        self.vegetation_thresholds = vegetation_thresholds
        self.unclassified_thresholds = unclassified_thresholds
        self.data_format = data_format

    def run(self, src_las_path: str, target_las_path: str):
        """Application.

        Transform cloud at `src_las_path` following bridge identification logic, and save it to
        `target_las_path`

        Args:
            src_las_path (str): path to input LAS file, with vegetation probabilities
            target_las_path (str): path for saving updated LAS file.

        Returns:
            str: returns `target_las_path` for potential terminal piping.

        """
        log.info(f"Applying Vegetation Identification to file \n{src_las_path}")
        
        # give alias to make things more readable
        las_dim = self.data_format.las_dimensions 
        codes = self.data_format.codes

        # read the LAS, get its points list and add a "groups" dimension, if needed 
        pipeline = pdal.Pipeline() | get_pdal_reader(src_las_path)
        pipeline.execute()
        try:
            points = pipeline.arrays[0]
            points[las_dim.groups] = 0  # if dimension "groups" doesn't exist, will raise a ValueError
        except ValueError:
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{las_dim.groups}") # add "groups" as a new dimension
            pipeline.execute()
            points = pipeline.arrays[0]

        # set the vegetation
        vegetation_mask = points[las_dim.ai_vegetation_proba] >= self.vegetation_thresholds   
        points[las_dim.groups][vegetation_mask] = codes.vegetation
         
        # set the unclassified
        unclassified_mask = points[las_dim.ai_unclassified_proba] >= self.unclassified_thresholds
        points[las_dim.groups][unclassified_mask] = codes.unclassified

        # save points list to the target
        pipeline_test = get_pdal_writer(target_las_path).pipeline(points)
        os.makedirs(os.path.dirname(target_las_path), exist_ok=True)
        pipeline_test.execute()
