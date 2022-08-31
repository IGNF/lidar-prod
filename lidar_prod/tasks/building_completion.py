import logging
import os
import os.path as osp
from tempfile import TemporaryDirectory
import pdal
from tqdm import tqdm

from lidar_prod.tasks.utils import get_pdal_reader, get_pdal_writer, split_idx_by_dim

log = logging.getLogger(__name__)


class BuildingCompletor:
    """Logic of building completion.

    Some points were too isolated for BuildingValidator to consider them.
    We will update their classification based on their probability as well as their surrounding:
    - We select points that have p>=0.5
    - We perform vertical (XY) clustering including these points as well as confirmed buildings.
    - In the resulting groups, if there are some confirmed buildings, previously isolated points are
    considered to be parts of the same building and their class is updated accordingly.

    """

    def __init__(
        self,
        min_building_proba: float = 0.75,
        min_building_proba_relaxation_if_bd_uni_overlay: float = 1.0,
        cluster=None,
        data_format=None,
    ):
        self.cluster = cluster
        self.min_building_proba = min_building_proba
        self.min_building_proba_relaxation_if_bd_uni_overlay = (
            min_building_proba_relaxation_if_bd_uni_overlay
        )
        self.data_format = data_format
        self.codes = data_format.codes.building  # easier access
        self.pipeline: pdal.pipeline.Pipeline = None

    def run(self, src_las_path: str, target_las_path: str):
        """Application.

        Transform cloud at `src_las_path` following building completion logic, and save it to
        `target_las_path`

        Args:
            src_las_path (str): path to input LAS file, output of BuildingValidator
            target_las_path (str): path for saving updated LAS file.

        Returns:
            str: returns `target_las_path` for potential terminal piping.

        """
        log.info(f"Applying Building Completion to file \n{src_las_path}")
        log.info(
            "Completion of building with relatively distant points that have high enough probability"
        )
        with TemporaryDirectory() as td:
            tmp_las_path = osp.join(td, osp.basename(src_las_path))
            self.prepare_for_building_completion(src_las_path, tmp_las_path)
            self.update_classification(target_las_path)
        return target_las_path

    def prepare_for_building_completion(
        self, src_las_path: str, target_las_path: str
    ) -> None:
        f"""Prepare for building completion.

        Identify candidates that were not clustered together by the BuildingValidator, but that
        have high enough probability. Then, cluster them together with previously confirmed buildings.
        Cluster parameters are relaxed (2D, with high tolerance).
        If a cluster contains some confirmed points, the others are considered to belong to the same building
        and they will be confirmed as well.

        Args:
            src_las_path (str): input LAS
            target_las_path (str): output, prepared LAS with a new `{self.data_format.las_dimensions.ClusterID_isolated_plus_confirmed}`
            dimension.
        """
        if not self.pipeline:
            self.pipeline = pdal.Pipeline()
            self.pipeline |= get_pdal_reader(src_las_path)
        candidates = (
            f"({self.data_format.las_dimensions.candidate_buildings_flag} == 1)"
        )

        where_not_clustered = (
            f"{self.data_format.las_dimensions.ClusterID_candidate_building} == 0"
        )

        # P above threshold
        p_heq_threshold = f"(building>={self.min_building_proba})"

        # P above relaxed threshold when under BDUni
        under_bd_uni = f"({self.data_format.las_dimensions.uni_db_overlay} > 0)"
        p_heq_relaxed_threshold = f"(building>={self.min_building_proba * self.min_building_proba_relaxation_if_bd_uni_overlay})"
        p_heq_threshold_under_bd_uni = f"({p_heq_relaxed_threshold} && {under_bd_uni})"

        # Candidates that where clustered by BuildingValidator but have high enough probability.
        not_clustered_but_with_high_p = f"{candidates} && {where_not_clustered} && ({p_heq_threshold} || {p_heq_threshold_under_bd_uni})"
        confirmed_buildings = (
            f"Classification == {self.data_format.codes.building.final.building}"
        )

        where = f"{not_clustered_but_with_high_p} || {confirmed_buildings}"
        self.pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            is3d=self.cluster.is3d,
            where=where,
        )
        # Always move and reset ClusterID to avoid conflict with later tasks.
        self.pipeline |= pdal.Filter.ferry(
            dimensions=f"{self.data_format.las_dimensions.cluster_id}=>{self.data_format.las_dimensions.ClusterID_isolated_plus_confirmed}"
        )
        self.pipeline |= pdal.Filter.assign(
            value=f"{self.data_format.las_dimensions.cluster_id} = 0"
        )
        self.pipeline |= get_pdal_writer(target_las_path)
        os.makedirs(osp.dirname(target_las_path), exist_ok=True)
        self.pipeline.execute()

    def update_classification(
        self, target_las_path: str
    ) -> None:
        """Updates Classification dimension by completing buildings with high probability points.

        Args:
            target_las_path (str): output LAS, with updated Classification dimension.
        """

        las = self.pipeline.arrays[0]

        # las = laspy.read(prepared_las_path)
        _clf = self.data_format.las_dimensions.classification
        _cid = self.data_format.las_dimensions.ClusterID_isolated_plus_confirmed

        # 2) Decide at the group-level
        split_idx = split_idx_by_dim(las[_cid])
        # Isolated/confirmed groups have a cluster index > 0
        split_idx = split_idx[1:]
        # For each group of isolated|confirmed points,
        # Assess if the group already contains confirmed points.
        # If it does, set all points to confirmed building class so that
        # the isolated points they may contain are also confirmed.
        for pts_idx in tqdm(
            split_idx, desc="Complete buildings with isolated points", unit="grp"
        ):
            # pts = las.points[pts_idx]
            pts = las[pts_idx]
            if self.codes.final.building in pts[_clf]:
                las[_clf][pts_idx] = self.codes.final.building

        # UGLY !!!
        self.pipeline = get_pdal_writer(target_las_path).pipeline(las)
        os.makedirs(osp.dirname(target_las_path), exist_ok=True)
        self.pipeline.execute()

        self.pipeline = pdal.Pipeline()
        self.pipeline |= get_pdal_reader(target_las_path)
        # END OF UGLYNESS
