"""
Takes vegetation probabilities as input, and defines vegetation

"""

from dataclasses import dataclass
import logging
import os
import numpy as np

import pdal

from lidar_prod.tasks.utils import get_pdal_reader, get_pdal_writer


log = logging.getLogger(__name__)


class VegetationIdentifier:
    """Logic of bridge identification.

    Bridge are complex, diverse objects that are hard to delineate.
    From bridge probabilities predicted by an AI model, the point cloud Classification is updated.
    Process:
    - Get binary bridge/non-bridge label from probabilities with a threshold
    - [TODO]: clusterize bridge-predicted points and discard isolated points
    - Update the classification channel from the point cloud.

    """

    def __init__(self, thresholds: float = 0.5, data_format=None):
        self.thresholds = thresholds
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

        pipeline = pdal.Pipeline() | get_pdal_reader(src_las_path)
        pipeline.execute()
        points = pipeline.arrays[0]
        vegetation_mask = self.identify_vegetation(
            points[self.data_format.las_dimensions.ai_vegetation_proba]
        )
        points["Classification"][vegetation_mask] = self.data_format.codes.vegetation
        pipeline_writer = get_pdal_writer(target_las_path).pipeline(points)
        os.makedirs(os.path.dirname(target_las_path), exist_ok=True)
        pipeline_writer.execute()

    def identify_vegetation(self, ai_vegetation_proba: np.ndarray) -> np.ndarray:
        """Return a mask for identified vegetation points from probabilities."""
        return ai_vegetation_proba >= self.thresholds
