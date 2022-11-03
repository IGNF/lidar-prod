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

    High enough probability means :
    - p>=min_building_proba
    OR, IF point fall in a building vector from the BDUni:
    - p>=(min_building_proba*min_frac_confirmation_factor_if_bd_uni_overlay).
    """

    def __init__(
        self,
        min_building_proba: float = 0.75,
        min_frac_confirmation_factor_if_bd_uni_overlay: float = 1.0,
        cluster=None,
        data_format=None,
    ):
        self.cluster = cluster
        self.data_format = data_format
        self.min_building_proba = min_building_proba
        self.min_frac_confirmation_factor_if_bd_uni_overlay = (
            min_frac_confirmation_factor_if_bd_uni_overlay
        )
        self.pipeline: pdal.pipeline.Pipeline = None

    def run(
        self,
        input_values: Union[str, pdal.pipeline.Pipeline],
        target_las_path: str = None,
    ) -> str:
        """Application.

        Transform cloud at `src_las_path` following identification logic, and save it to
        `target_las_path`

        Args:
            input_values (str| pdal.pipeline.Pipeline): path or pipeline to input LAS file with a building probability channel
            target_las_path (str): path for saving updated LAS file.

        Returns:
            str:  `target_las_path`

        """
        log.info("Clustering of points with high building proba.")
        self.prepare(input_values, target_las_path)
        return target_las_path

    def prepare(
        self,
        input_values: Union[str, pdal.pipeline.Pipeline],
        target_las_path: str = None,
    ) -> None:
        """Identify potential buildings in a new channel, excluding former candidates from
        search based on their group ID. ClusterID needs to be reset to avoid unwanted merge
        of information from previous VuildingValidation clustering.

        Args:
            input_values (str| pdal.pipeline.Pipeline): path or pipeline to input LAS file with a building probability channel
            target_las_path (str): output LAS
        """
        self.pipeline = get_pipeline(input_values)
        non_candidates = (
            f"({self.data_format.las_dimensions.candidate_buildings_flag} == 0)"
        )
        p_heq_threshold = f"(building>={self.min_building_proba})"
        A = f"(building>={self.min_building_proba * self.min_frac_confirmation_factor_if_bd_uni_overlay})"
        B = f"({self.data_format.las_dimensions.uni_db_overlay} > 0)"
        p_heq_modified_threshold_under_bd_uni = f"({A} && {B})"
        where = f"{non_candidates} && ({p_heq_threshold} || {p_heq_modified_threshold_under_bd_uni})"
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
        self.pipeline |= pdal.Filter.assign(
            value=f"{self.data_format.las_dimensions.cluster_id} = 0"
        )
        if target_las_path:
            self.pipeline |= get_pdal_writer(target_las_path)
            os.makedirs(osp.dirname(target_las_path), exist_ok=True)
        self.pipeline.execute()
