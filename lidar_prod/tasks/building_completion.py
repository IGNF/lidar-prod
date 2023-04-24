import logging
from typing import Union

import pdal
from tqdm import tqdm

from lidar_prod.tasks.utils import get_pipeline, split_idx_by_dim

log = logging.getLogger(__name__)


class BuildingCompletor:
    """Logic of building completion.

    The BuildingValidator only considered points that were 1) candidate, and 2) formed clusers of sufficient size.

    Some points were too isolated, or where not clustered, even though they might have a
    high predicted building probabiliy.
    We assume that we can trust AI probabilities (if high enough) in the neigborhood of large groups (clusters)
    of candidate points already confirmed by the BuildingValidator.

    We will update points classification based on their probability as well as their surrounding:
    - We select points that have p>=0.5 (+ a BDUni factor when applicable)
    - We perform vertical (XY) clustering of A) these points, together with B) confirmed buildings.
    - If the resulting clusters contain confirmed buildings, points with high probability are
    considered to be part of the confirmed building and their class is updated accordingly.

    """

    def __init__(
        self,
        min_building_proba: float = 0.5,
        cluster=None,
        data_format=None,
    ):
        self.cluster = cluster
        self.min_building_proba = min_building_proba
        self.data_format = data_format
        self.pipeline: pdal.pipeline.Pipeline = None

    # @profile()
    def run(self, input_values: Union[str, pdal.pipeline.Pipeline]):
        """Application.

        Transform cloud at `src_las_path` following building completion logic

        Args:
            input_values (str|pdal.pipeline.Pipeline): path to either input LAS file or a pipeline
            target_las_path (str): path for saving updated LAS file.

        Returns:
            str: returns `target_las_path` for potential terminal piping.

        """
        log.info("Completion of building with relatively distant points that have high enough probability")
        pipeline = get_pipeline(input_values)
        self.prepare_for_building_completion(pipeline)
        self.update_classification()

    def prepare_for_building_completion(self, pipeline: pdal.pipeline.Pipeline) -> None:
        """Prepare for building completion.

        Identify candidates that have high enough probability. Then, cluster them together with previously confirmed buildings.
        Cluster parameters are relaxed (2D, with high tolerance).
        If a cluster contains some confirmed points, the others are considered to belong to the same building
        and they will be confirmed as well.

        Args:
            src_las_path (pdal.pipeline.Pipeline): input LAS pipeline
            target_las_path (str): output, prepared LAS.

        """

        # Reset Cluster dim out of safety
        dim_cluster_id_pdal = self.data_format.las_dimensions.cluster_id
        pipeline |= pdal.Filter.assign(value=f"{dim_cluster_id_pdal} = 0")

        # Candidates that where already confirmed by BuildingValidator.
        confirmed_buildings = f"Classification == {self.data_format.codes.building.final.building}"
        p_heq_threshold = f"(building>={self.min_building_proba})"
        where = f"{p_heq_threshold} || {confirmed_buildings}"
        pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            is3d=self.cluster.is3d,
            where=where,
        )
        # Always move then reset ClusterID to avoid conflict with later tasks.
        pipeline |= pdal.Filter.ferry(
            dimensions=f"{self.data_format.las_dimensions.cluster_id}=>{self.data_format.las_dimensions.ClusterID_confirmed_or_high_proba}"
        )
        pipeline |= pdal.Filter.assign(value=f"{self.data_format.las_dimensions.cluster_id} = 0")
        # Create a placeholder dimension that will hold non-candidate points with high enough probas
        pipeline |= pdal.Filter.ferry(dimensions=f"=> {self.data_format.las_dimensions.completion_non_candidate_flag}")
        # Run
        pipeline.execute()

        # set pipeline for access in next operations/tasks.
        self.pipeline = pipeline

    def update_classification(self) -> None:
        """Updates Classification dimension by completing buildings with high probability points."""

        points = self.pipeline.arrays[0]

        _clf = self.data_format.las_dimensions.classification
        _cid = self.data_format.las_dimensions.ClusterID_confirmed_or_high_proba
        _completion_flag = self.data_format.las_dimensions.completion_non_candidate_flag
        _candidate_flag = self.data_format.las_dimensions.candidate_buildings_flag

        # 2) Decide at the group-level
        split_idx = split_idx_by_dim(points[_cid])
        # Isolated/confirmed groups have a cluster index > 0
        split_idx = split_idx[1:]
        # For each group of isolated|confirmed points,
        # Assess if the group already contains confirmed points. If it does, points
        # with high proba may belong to the same building.
        for pts_idx in tqdm(split_idx, desc="Complete buildings with isolated points", unit="grp"):
            if self.data_format.codes.building.final.building in points[_clf][pts_idx]:
                candidates_mask = points[_candidate_flag][pts_idx] == 1
                # (a) If a point is a candidate building, Then confirm it.
                candidates_idx = pts_idx[candidates_mask]
                points[_clf][candidates_idx] = self.data_format.codes.building.final.building
                # (b) If a point is not a candidate building, set a flag to
                # identify it as a potential completion, for future human inspection.
                non_candidates_idx = pts_idx[~candidates_mask]
                points[_completion_flag][non_candidates_idx] = 1
        self.pipeline = pdal.Pipeline(arrays=[points])
