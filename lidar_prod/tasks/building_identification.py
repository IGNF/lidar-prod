import logging
import os
import os.path as osp
from typing import Union

import pdal

from lidar_prod.tasks.utils import get_pdal_writer, get_pipeline

log = logging.getLogger(__name__)


class BuildingIdentifier:
    """Logic of building validation.

    Points that were not found by rule-based algorithms but which have a high-enough probability of
    being a building are clustered into candidate groups of buildings.

    High enough probability means p>=min_building_proba
    """

    def __init__(
        self,
        min_building_proba: float = 0.5,
        cluster=None,
        data_format=None,
    ):
        self.cluster = cluster
        self.data_format = data_format
        self.min_building_proba = min_building_proba
        self.pipeline: pdal.pipeline.Pipeline = None

    def run(
        self,
        input_values: Union[str, pdal.pipeline.Pipeline],
        target_las_path: str = None,
    ) -> str:
        """Identify potential buildings in a new channel, excluding former candidates as well as already
        confirmed building (confirmed by either Validation or Completion).

        Args:
            input_values (str | pdal.pipeline.Pipeline): path or pipeline to input LAS file with a building probability channel
            target_las_path (str): output LAS

        """

        log.info("Clustering of points with high building proba.")
        self.pipeline = get_pipeline(input_values)

        # Considered for identification:
        non_candidates = f"({self.data_format.las_dimensions.candidate_buildings_flag} == 0)"
        not_already_confirmed = f"{self.data_format.las_dimensions.classification} != {self.data_format.codes.building.final.building}"
        p_heq_threshold = f"(building>={self.min_building_proba})"
        where = f"({non_candidates} && {not_already_confirmed} && {p_heq_threshold})"
        self.pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            is3d=self.cluster.is3d,
            where=where,
        )
        # Always move and reset ClusterID to avoid conflict with later tasks.
        self.pipeline |= pdal.Filter.ferry(
            dimensions=f"{self.data_format.las_dimensions.cluster_id}=>{self.data_format.las_dimensions.ai_building_identified}"
        )
        self.pipeline |= pdal.Filter.assign(value=f"{self.data_format.las_dimensions.cluster_id} = 0")
        if target_las_path:
            self.pipeline |= get_pdal_writer(target_las_path)
            os.makedirs(osp.dirname(target_las_path), exist_ok=True)
        self.pipeline.execute()

        return target_las_path
