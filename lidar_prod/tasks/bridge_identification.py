"""
Takes bridge probabilities as input, and defines bridge.

"""

import logging
import os
import numpy as np

import pdal

from lidar_prod.tasks.utils import get_pdal_reader, get_pdal_writer


log = logging.getLogger(__name__)


class BridgeIdentifier:
    """Logic of bridge identification.

    Bridge are complex, diverse objects that are hard to delineate.
    From bridge probabilities predicted by an AI model, the point cloud Classification is updated.
    Process:
    - Get binary bridge/non-bridge label from probabilities with a threshold
    - [TODO]: clusterize bridge-predicted points and discard isolated points
    - Update the classification channel from the point cloud.

    """

    def __init__(self, min_bridge_proba: float = 0.5, data_format=None):
        self.min_bridge_proba = min_bridge_proba
        self.data_format = data_format

    def run(self, src_las_path: str, target_las_path: str):
        """Application.

        Transform cloud at `src_las_path` following bridge identification logic, and save it to
        `target_las_path`

        Args:
            src_las_path (str): path to input LAS file, with bridge proabbilities
            target_las_path (str): path for saving updated LAS file.

        Returns:
            str: returns `target_las_path` for potential terminal piping.

        """
        log.info(f"Applying Bridge Identification to file \n{src_las_path}")

        pipeline = pdal.Pipeline() | get_pdal_reader(src_las_path)
        pipeline.execute()
        points = pipeline.arrays[0]
        bridge_mask = self.identify_bridges(
            points[self.data_format.las_dimensions.ai_bridge_proba]
        )
        points["Classification"][bridge_mask] = self.data_format.codes.bridge
        pipeline_writer = get_pdal_writer(target_las_path).pipeline(points)
        os.makedirs(os.path.dirname(target_las_path), exist_ok=True)
        pipeline_writer.execute()

    def identify_bridges(self, ai_bridge_proba: np.ndarray) -> np.ndarray:
        """Return a mask for identified bridge points from probabilities."""
        return ai_bridge_proba >= self.min_bridge_proba
