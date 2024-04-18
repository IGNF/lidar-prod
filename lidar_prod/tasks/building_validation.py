import logging
import os
import os.path as osp
import shutil
from dataclasses import dataclass
from tempfile import TemporaryDirectory, mkdtemp
from typing import Optional, Union

import geopandas
import numpy as np
import pdal
import yaml
from tqdm import tqdm

from lidar_prod.tasks.utils import (
    get_integer_bbox,
    get_pdal_writer,
    get_pipeline,
    request_bd_uni_for_building_shapefile,
    split_idx_by_dim,
)

log = logging.getLogger(__name__)


@dataclass
class BuildingValidationClusterInfo:
    """Elements needed to confirm, refute, or be uncertain about a cluster
    of candidate building points."""

    probabilities: np.ndarray
    overlays: np.ndarray
    entropies: np.ndarray

    # target is based on corrected labels - only needed for optimization of decision thresholds
    target: Optional[int] = None


class BuildingValidator:
    """Logic of building validation.

    The candidate building points identified with a rule-based algorithm are cluster together.
    The BDUni building vectors are overlayed on the points clouds, and points that fall under
    a vector are flagged.
    Then, classification dim is updated on a per-group basis, based on both AI probabilities and
    BDUni flag.

    See `README.md` for the detailed process.
    """

    def __init__(
        self,
        shp_path: str = None,
        bd_uni_connection_params=None,
        cluster=None,
        bd_uni_request=None,
        data_format=None,
        thresholds=None,
        use_final_classification_codes: bool = True,
    ):
        self.shp_path = shp_path
        self.bd_uni_connection_params = bd_uni_connection_params
        self.cluster = cluster
        self.bd_uni_request = bd_uni_request
        self.use_final_classification_codes = use_final_classification_codes
        self.thresholds = thresholds  # default values
        self.data_format = data_format
        # For easier access
        self.codes = data_format.codes.building
        self.candidate_buildings_codes = data_format.codes.building.candidates
        self.pipeline: pdal.pipeline.Pipeline = None
        self.setup()

    def setup(self):
        """Setup. Defines useful variables."""

        self.detailed_to_final_map: dict = {
            detailed: final for detailed, final in self.codes.detailed_to_final
        }

    def run(
        self,
        input_values: Union[str, pdal.pipeline.Pipeline],
        target_las_path: str = None,
        las_metadata: dict = None,
    ) -> dict:
        """Runs application.

        Transforms cloud at `input_values` following building validation logic,
        and saves it to `target_las_path`

        Args:
            input_values (str| pdal.pipeline.Pipeline): path or pipeline to input LAS file with
            a building probability channel
            target_las_path (str): path for saving updated LAS file.
            las_metadata (dict): current pipeline metadata, used to propagate input metadata to the
        application output las (epsg, las version, etc)

        Returns:
            str: returns `las_metadata`: metadata of the input las, which contain
            information to pass to the writer in order for the application to have an output
            with the same header (las version, srs, ...) as the input

        """
        self.pipeline, las_metadata = get_pipeline(input_values, self.data_format.epsg)
        with TemporaryDirectory() as td:
            log.info("Preparation : Clustering of candidates buildings & Import vectors")
            if isinstance(input_values, str):
                log.info(f"Applying Building Validation to file \n{input_values}")
                temp_f = osp.join(td, osp.basename(input_values))
            else:
                temp_f = ""
            las_metadata = self.prepare(input_values, temp_f, las_metadata)
            log.info("Using AI and Databases to update cloud Classification")
            las_metadata = self.update(target_las_path=target_las_path, las_metadata=las_metadata)
        return las_metadata

    def prepare(
        self,
        input_values: Union[str, pdal.pipeline.Pipeline],
        prepared_las_path: str,
        save_result: bool = False,
        las_metadata: dict = None,
    ) -> dict:
        f"""
        Prepare las for later decision process. .
        1. Cluster candidates points, in a new
        `{self.data_format.las_dimensions.ClusterID_candidate_building}`
        dimension where the index of clusters starts at 1 (0 means no cluster).
        2. Identify points overlayed by a BD Uni building, in a new
        `{self.data_format.las_dimensions.uni_db_overlay}` dimension (0/1 flag).

        In the process is created a new dimensions which identifies candidate buildings (0/1 flag)
        `{self.data_format.las_dimensions.candidate_buildings_flag}`, to ignore them in later
        buildings identification.

        Dimension classification should not be modified here, as optimization step needs to
        do this step once before testing multiple decision parameters on the same prepared data.

        Args:
            input_values (str| pdal.pipeline.Pipeline): path or pipeline to input LAS file with
            a building probability channel
            target_las_path (str): path for saving prepared LAS file.
            save_result (bool): True to save a las instead of propagating a pipeline
            las_metadata (dict): current pipeline metadata, used to propagate input metadata to the
        application output las (epsg, las version, etc)

        Returns:
            updated las metadata

        """

        dim_candidate_flag = self.data_format.las_dimensions.candidate_buildings_flag
        dim_cluster_id_pdal = self.data_format.las_dimensions.cluster_id
        dim_cluster_id_candidates = self.data_format.las_dimensions.ClusterID_candidate_building
        dim_overlay = self.data_format.las_dimensions.uni_db_overlay

        self.pipeline, las_metadata = get_pipeline(input_values, self.data_format.epsg)
        # Identify candidates buildings points with a boolean flag
        self.pipeline |= pdal.Filter.ferry(dimensions=f"=>{dim_candidate_flag}")
        _is_candidate_building = (
            "("
            + " || ".join(
                f"Classification == {int(candidate_code)}"
                for candidate_code in self.candidate_buildings_codes
            )
            + ")"
        )
        self.pipeline |= pdal.Filter.assign(
            value=f"{dim_candidate_flag} = 1 WHERE {_is_candidate_building}"
        )
        # Cluster candidates buildings points. This creates a ClusterID dimension (int)
        # in which unclustered points have index 0.
        self.pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            where=f"{dim_candidate_flag} == 1",
        )

        # Copy ClusterID into a new dim and reset it to 0 to avoid conflict with later tasks.
        self.pipeline |= pdal.Filter.ferry(
            dimensions=f"{dim_cluster_id_pdal}=>{dim_cluster_id_candidates}"
        )
        self.pipeline |= pdal.Filter.assign(value=f"{dim_cluster_id_pdal} = 0")
        self.pipeline.execute()
        bbox = get_integer_bbox(self.pipeline, buffer=self.bd_uni_request.buffer)

        self.pipeline |= pdal.Filter.ferry(dimensions=f"=>{dim_overlay}")

        if self.shp_path:
            # no need for a temporay directory to add the shapefile in it, we already have the
            # shapefile
            temp_dirpath = None
            _shp_p = self.shp_path
            log.info(f"Read shapefile\n {_shp_p}")
            gdf = geopandas.read_file(_shp_p)
            buildings_in_bd_topo = not len(gdf) == 0  # check if there are buildings in the shp

        else:
            temp_dirpath = mkdtemp()
            # TODO: extract coordinates from LAS directly using pdal.
            # Request BDUni to get a shapefile of the known buildings in the LAS
            _shp_p = os.path.join(temp_dirpath, "temp.shp")
            log.info("Request Bd Uni")
            buildings_in_bd_topo = request_bd_uni_for_building_shapefile(
                self.bd_uni_connection_params, _shp_p, bbox, self.data_format.epsg
            )

        # Create overlay dim
        # If there are some buildings in the database, create a BDTopoOverlay boolean
        # dimension to reflect it.

        if buildings_in_bd_topo:
            self.pipeline |= pdal.Filter.overlay(
                column="PRESENCE", datasource=_shp_p, dimension=dim_overlay
            )

        if save_result:
            self.pipeline |= get_pdal_writer(prepared_las_path, las_metadata)
            os.makedirs(osp.dirname(prepared_las_path), exist_ok=True)
        self.pipeline.execute()

        if temp_dirpath:
            shutil.rmtree(temp_dirpath)

        return las_metadata

    def update(
        self, src_las_path: str = None, target_las_path: str = None, las_metadata: dict = None
    ) -> dict:
        """Updates point cloud classification channel."""
        if src_las_path:
            self.pipeline, las_metadata = get_pipeline(src_las_path, self.data_format.epsg)

        points = self.pipeline.arrays[0]

        # 1) Map all points to a single "not_building" class
        # to be sure that they will all be modified.

        dim_clf = self.data_format.las_dimensions.classification
        dim_flag = self.data_format.las_dimensions.candidate_buildings_flag
        candidates_mask = points[dim_flag] == 1
        points[dim_clf][candidates_mask] = self.codes.final.not_building

        # 2) Decide at the group-level
        # TODO: check if this can be moved somewhere else.
        # WARNING: use_final_classification_codes may be modified in an unsafe manner during
        # optimization. Consider using a setter that will change decision_func alongside.

        # Decide level of details of classification codes
        decision_func = self._make_detailed_group_decision
        if self.use_final_classification_codes:
            decision_func = self._make_group_decision

        # Get the index of points of each cluster
        # Remove unclustered group that have ClusterID = 0 (i.e. the first "group")
        cluster_id_dim = points[self.data_format.las_dimensions.ClusterID_candidate_building]
        split_idx = split_idx_by_dim(cluster_id_dim)
        split_idx = split_idx[1:]

        # Iterate over groups and update their classification
        for pts_idx in tqdm(split_idx, desc="Update cluster classification", unit="clusters"):
            infos = self._extract_cluster_info_by_idx(points, pts_idx)
            points[dim_clf][pts_idx] = decision_func(infos)

        self.pipeline = pdal.Pipeline(arrays=[points])

        if target_las_path:
            self.pipeline = get_pdal_writer(target_las_path, las_metadata).pipeline(points)
            os.makedirs(osp.dirname(target_las_path), exist_ok=True)
            self.pipeline.execute()

        return las_metadata

    def _extract_cluster_info_by_idx(
        self, las: np.ndarray, pts_idx: np.ndarray
    ) -> BuildingValidationClusterInfo:
        """Extracts all necessary information to make a decision based on points indices.

        Args:
            las (np.ndarray): point cloud of interest
            pts_idx (np.ndarray): indices of points in considered clusters

        Returns:
            BuildingValidationClusterInfo: data necessary to make a decision at cluster level.

        """
        pts = las[pts_idx]
        probabilities = pts[self.data_format.las_dimensions.ai_building_proba]
        overlays = pts[self.data_format.las_dimensions.uni_db_overlay]
        entropies = pts[self.data_format.las_dimensions.entropy]
        targets = pts[self.data_format.las_dimensions.classification]
        return BuildingValidationClusterInfo(probabilities, overlays, entropies, targets)

    def _make_group_decision(self, *args, **kwargs) -> int:
        f"""Wrapper to simplify decision codes during LAS update.
        Signature follows the one of {self._make_detailed_group_decision.__name__}
        Returns:
            int: final classification code for the considered group.
        """
        detailed_code = self._make_detailed_group_decision(*args, **kwargs)
        return self.detailed_to_final_map[detailed_code]

    def _make_detailed_group_decision(self, infos: BuildingValidationClusterInfo) -> int:
        """Decision process at the cluster level.

        Confirm or refute candidate building groups based on fraction of confirmed/refuted points
        and on fraction of points overlayed by a building vector in BDUni.

        See Readme for details of this group-level decision process.

        Args:
            infos (BuildngValidationClusterInfo): arrays describing the cluster of candidate
            builiding points

        Returns:
            int: detailed classification code for the considered group.

        """
        # HIGH ENTROPY

        high_entropy = (
            np.mean(infos.entropies >= self.thresholds.min_entropy_uncertainty)
            >= self.thresholds.min_frac_entropy_uncertain
        )

        # CONFIRMATION - threshold is relaxed under BDUni
        p_heq_threshold = infos.probabilities >= self.thresholds.min_confidence_confirmation

        relaxed_threshold = (
            self.thresholds.min_confidence_confirmation
            * self.thresholds.min_frac_confirmation_factor_if_bd_uni_overlay
        )
        p_heq_relaxed_threshold = infos.probabilities >= relaxed_threshold

        ia_confirmed_flag = np.logical_or(
            p_heq_threshold,
            np.logical_and(infos.overlays, p_heq_relaxed_threshold),
        )

        ia_confirmed = np.mean(ia_confirmed_flag) >= self.thresholds.min_frac_confirmation

        # REFUTATION
        ia_refuted = (
            np.mean((1 - infos.probabilities) >= self.thresholds.min_confidence_refutation)
            >= self.thresholds.min_frac_refutation
        )
        uni_overlayed = np.mean(infos.overlays) >= self.thresholds.min_uni_db_overlay_frac
        # If low entropy, we may trust AI to confirm/refute
        if not high_entropy:
            if ia_refuted:
                if uni_overlayed:
                    return self.codes.detailed.ia_refuted_but_under_db_uni
                return self.codes.detailed.ia_refuted
            if ia_confirmed:
                if uni_overlayed:
                    return self.codes.detailed.both_confirmed
                return self.codes.detailed.ia_confirmed_only
        # Else, we may still use BDUni information
        if uni_overlayed:
            return self.codes.detailed.db_overlayed_only

        # Else: we are uncertain, and we specify why we can specify if entropy was
        # involved to conclude to uncertainty.
        if high_entropy:
            return self.codes.detailed.unsure_by_entropy
        return self.codes.detailed.both_unsure


@dataclass
class thresholds:
    """The decision thresholds for a cluser-level decisions."""

    min_confidence_confirmation: float
    min_frac_confirmation: float
    min_frac_confirmation_factor_if_bd_uni_overlay: float
    min_uni_db_overlay_frac: float
    min_confidence_refutation: float
    min_frac_refutation: float
    min_entropy_uncertainty: float
    min_frac_entropy_uncertain: float

    def dump(self, filename: str):
        with open(filename, "w") as f:
            yaml.safe_dump(self.__dict__, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        return thresholds(**data)
