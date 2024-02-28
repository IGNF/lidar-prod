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
        # aliases
        _cid = self.data_format.las_dimensions.cluster_id
        _completion_flag = self.data_format.las_dimensions.completion_non_candidate_flag

        log.info("Clustering of points with high building proba.")
        pipeline = get_pipeline(input_values, self.data_format.epsg)

        # Considered for identification:
        non_candidates = (
            f"({self.data_format.las_dimensions.candidate_buildings_flag} == 0)"
        )
        not_already_confirmed = f"({self.data_format.las_dimensions.classification} != {self.data_format.codes.building.final.building})"
        not_a_potential_completion = f"({_completion_flag} != 1)"
        p_heq_threshold = f"(building>={self.min_building_proba})"
        where = f"({non_candidates} && {not_already_confirmed} && {not_a_potential_completion} && {p_heq_threshold})"
        pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            is3d=self.cluster.is3d,
            where=where,
        )
        # Increment ClusterID, so that points from building completion can become cluster 1
        pipeline |= pdal.Filter.assign(
            value=f"{_cid} = {_cid} + 1", where=f"{_cid} != 0"
        )
        pipeline |= pdal.Filter.assign(
            value=f"{_cid} = 1", where=f"{_completion_flag} == 1"
        )
        # Duplicate ClusterID to have an explicit name for it for inspection.
        # Do not reset it to zero to have access to it at human inspection stage.
        pipeline |= pdal.Filter.ferry(
            dimensions=f"{_cid}=>{self.data_format.las_dimensions.ai_building_identified}"
        )
        if target_las_path:
            pipeline |= get_pdal_writer(target_las_path)
            os.makedirs(osp.dirname(target_las_path), exist_ok=True)
        pipeline.execute()

        self.pipeline = pipeline

        return target_las_path
