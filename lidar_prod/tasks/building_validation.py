from dataclasses import dataclass
import logging
import os
import os.path as osp
from typing import Optional
import pickle
import subprocess
import numpy as np
import pdal
from tempfile import TemporaryDirectory
import geopandas
import laspy
from tqdm import tqdm
from lidar_prod.tasks.utils import BDUniConnectionParams, extract_coor, split_idx_by_dim

log = logging.getLogger(__name__)


@dataclass
class BuildingValidationClusterInfo:
    """Elements needed to either confirm, refute, or be uncertain about a cluster of cnadidate building points."""

    probabilities: np.ndarray
    overlays: np.ndarray
    entropies: np.ndarray

    # target is based on corrected labels - only needed for optimization of decision thresholds
    target: Optional[int] = None


class BuildingValidator:
    """Logic of building validation.

    The candidate building points identified with a rule-based algorithm are cluster together.
    The BDUni building vectors are overlayed on the points clouds, and points that fall under a vector are flagged.
    Then, classification dim is updated on a per-group basis, based on both AI probabilities and BDUni flag.

    See `README.md` for the detailed process.
    """

    def __init__(
        self,
        bd_uni_connection_params=None,
        cluster=None,
        bd_uni_request=None,
        data_format=None,
        thresholds=None,
        building_validation_thresholds_pickle: str = None,
        use_final_classification_codes: bool = True,
    ):
        self.bd_uni_connection_params = bd_uni_connection_params
        self.cluster = cluster
        self.bd_uni_request = bd_uni_request
        self.use_final_classification_codes = use_final_classification_codes
        self.thresholds = thresholds  # default values
        self.data_format = data_format
        # For easier access
        self.codes = data_format.codes.building
        self.candidate_buildings_codes = data_format.codes.building.candidates

        self.setup(building_validation_thresholds_pickle)

    def setup(self, building_validation_thresholds_pickle):
        """Setup, loading optimized thresholds if available."""
        if osp.exists(building_validation_thresholds_pickle):
            self._set_thresholds_from_pickle(building_validation_thresholds_pickle)
            log.info(
                f"Using optimized thresholds from: {building_validation_thresholds_pickle}"
            )
        else:
            log.warning(
                "Using default decision thresholds - specify "
                "'building_validation.application.building_validation_thresholds_pickle' "
                "to use thresholds of an ad-hoc optimization step"
            )

        self.detailed_to_final = {
            self.codes.detailed.unclustered: self.codes.final.not_building,
            self.codes.detailed.ia_refuted: self.codes.final.not_building,
            self.codes.detailed.ia_refuted_and_db_overlayed: self.codes.final.unsure,
            self.codes.detailed.both_unsure: self.codes.final.unsure,
            self.codes.detailed.unsure_by_entropy: self.codes.final.unsure,
            self.codes.detailed.ia_confirmed_only: self.codes.final.building,
            self.codes.detailed.db_overlayed_only: self.codes.final.building,
            self.codes.detailed.both_confirmed: self.codes.final.building,
        }
        self.detailed_to_final_mapper = np.vectorize(
            lambda detailed_code: self.detailed_to_final.get(detailed_code),
            otypes=[np.int],
        )

    def run(
        self,
        in_f: str,
        out_f: str,
    ):
        """Application.

        Transform cloud at `in_f` following building validation logic,
        and save it to `out_f`

        Args:
            in_f (str): path to input LAS file with a building probability channel
            out_f (str): path for saving updated LAS file.

        Returns:
            _type_: returns `out_f` for potential terminal piping.

        """
        with TemporaryDirectory() as td:
            log.info(f"Applying Building Validation to file \n{in_f}")
            log.info(
                "Preparation : Clustering of candidates buildings & Requesting BDUni"
            )
            temp_f = osp.join(td, osp.basename(in_f))
            self.prepare(in_f, temp_f)
            log.info("Using AI and Databases to update cloud Classification")
            self.update(temp_f, out_f)
        return out_f

    def prepare(self, in_f: str, out_f: str):
        f"""
        Prepare las for later decision process. .
        1. Cluster candidates points, in a new `{self.data_format.las_dimensions.ClusterID_candidate_building}`
        dimension where the index of clusters starts at 1 (0 means no cluster).
        2. Identify points overlayed by a BD Uni building, in a new
        `{self.data_format.las_dimensions.uni_db_overlay}` dimension (0/1 flag).

        In the process is created a new dimensions which identifies candidate buildings (0/1 flag)
        `{self.data_format.las_dimensions.candidate_buildings_flag}`, to ignore them in later
        buildings identification.

        Dimension classification should not be modified here, as optimization step needs unmo

        """

        _candidate_flag = self.data_format.las_dimensions.candidate_buildings_flag
        _cluster_id = self.data_format.las_dimensions.cluster_id
        _groups = self.data_format.las_dimensions.ClusterID_candidate_building
        _overlay = self.data_format.las_dimensions.uni_db_overlay

        # We use a temporary directory to clean intermediary files automatically
        with TemporaryDirectory() as td:
            pipeline = pdal.Pipeline()
            pipeline |= pdal.Reader(
                in_f,
                type="readers.las",
                nosrs=True,
                override_srs=self.data_format.crs_prefix + str(self.data_format.crs),
            )
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{_candidate_flag}")
            _is_candidate_building = (
                "("
                + " || ".join(
                    f"Classification == {int(candidate_code)}"
                    for candidate_code in self.candidate_buildings_codes
                )
                + ")"
            )
            pipeline |= pdal.Filter.assign(
                value=f"{_candidate_flag} = 1 WHERE {_is_candidate_building}"
            )
            pipeline |= pdal.Filter.cluster(
                min_points=self.cluster.min_points,
                tolerance=self.cluster.tolerance,
                where=f"{_candidate_flag} == 1",
            )
            # Always move and reset ClusterID to avoid conflict with later tasks.
            pipeline |= pdal.Filter.ferry(dimensions=f"{_cluster_id}=>{_groups}")
            pipeline |= pdal.Filter.assign(value=f"{_cluster_id} = 0")

            # TODO: extract coordinates from LAS directly using pdal.
            _shp_p = os.path.join(td, "temp.shp")
            buildings_in_bd_topo = request_bd_uni_for_building_shapefile(
                self.bd_uni_connection_params,
                *extract_coor(
                    os.path.basename(in_f),
                    self.data_format.tile_size_meters,
                    self.data_format.tile_size_meters,
                    self.bd_uni_request.buffer,
                ),
                self.data_format.crs,
                _shp_p,
            )

            # Channel is always created even if there are no buildings in database.
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{_overlay}")
            if buildings_in_bd_topo:
                pipeline |= pdal.Filter.overlay(
                    column="PRESENCE", datasource=_shp_p, dimension=_overlay
                )
            pipeline |= pdal.Writer(
                type="writers.las", filename=out_f, forward="all", extra_dims="all"
            )
            os.makedirs(osp.dirname(out_f), exist_ok=True)
            pipeline.execute()

    def update(self, prepared_f: str, out_f: str):
        """Update point cloud classification channel."""

        las = laspy.read(prepared_f)
        # 1) Map all points to a single class in case there was multiple codes to flag candidate buildings.
        # TODO: perform this at preparation step.

        _clf = self.data_format.las_dimensions.classification
        _flag = self.data_format.las_dimensions.candidate_buildings_flag
        candidates_idx = las[_flag] == 1
        las[_clf][candidates_idx] = self.codes.detailed.unclustered

        # 2) Decide at the group-level
        # TODO: check if this can be moved somewhere else. WARNING: use_final_classification_codes may be modified in
        # an unsafe manner during optimization. Consider using a setter that will change decision_func alongside.

        decision_func = self._make_group_decision
        if self.use_final_classification_codes:
            decision_func = self._make_detailed_group_decision

        split_idx = split_idx_by_dim(
            las[self.data_format.las_dimensions.ClusterID_candidate_building]
        )
        START_IDX_OF_CLUSTERS = 1
        split_idx = split_idx[
            START_IDX_OF_CLUSTERS:
        ]  # removes unclustered group that have ClusterID = 0
        for pts_idx in tqdm(
            split_idx, desc="Update cluster classification", unit="clusters"
        ):
            infos = self._extract_cluster_info_by_idx(las, pts_idx)
            las[_clf][pts_idx] = decision_func(infos)

        os.makedirs(osp.dirname(out_f), exist_ok=True)
        las.write(out_f)

    def _extract_cluster_info_by_idx(
        self, las: laspy.LasData, pts_idx: np.ndarray
    ) -> BuildingValidationClusterInfo:
        """Extract all necessary information to make a decision based on points indices.

        Args:
            las (laspy.LasData): point cloud of interest
            pts_idx (np.ndarray): indices of points in considered clusters

        Returns:
            _type_: _description_
        """
        pts = las.points[pts_idx]
        probabilities = pts[self.data_format.las_dimensions.ai_building_proba]
        overlays = pts[self.data_format.las_dimensions.uni_db_overlay]
        entropies = pts[self.data_format.las_dimensions.entropy]
        targets = pts[self.data_format.las_dimensions.classification]
        return BuildingValidationClusterInfo(
            probabilities, overlays, entropies, targets
        )

    def _make_group_decision(self, *args, **kwargs) -> int:
        f"""Wrapper to simplify decision codes during LAS update.
        Signature follows the one of {self._make_detailed_group_decision.__name__}
        Returns:
            int: final classification code for the considered group.
        """
        detailed_code = self._make_detailed_group_decision(*args, **kwargs)
        return self.detailed_to_final[detailed_code]

    def _make_detailed_group_decision(self, infos: BuildingValidationClusterInfo):
        """Decision process at the cluster level.

        Confirm or refute candidate building groups based on fraction of confirmed/refuted points and
        on fraction of points overlayed by a building vector in BDUni.

        See Readme for details of this group-level decision process.

        Args:
            infos (BuildngValidationClusterInfo): arrays describing the cluster of candidate builiding points

        Returns:
            int: detailed classification code for the considered group.
        """
        # HIGH ENTROPY

        high_entropy = (
            np.mean(infos.entropies >= self.thresholds.min_entropy_uncertainty)
            >= self.thresholds.min_frac_entropy_uncertain
        )

        # CONFIRMATION - threshold is relaxed under BDUni
        p_heq_threshold = (
            infos.probabilities >= self.thresholds.min_confidence_confirmation
        )

        relaxed_threshold = (
            self.thresholds.min_confidence_confirmation
            * self.thresholds.min_frac_confirmation_factor_if_bd_uni_overlay
        )
        p_heq_relaxed_threshold = infos.probabilities >= relaxed_threshold

        ia_confirmed_flag = np.logical_or(
            p_heq_threshold, np.logical_and(infos.overlays, p_heq_relaxed_threshold)
        )

        ia_confirmed = (
            np.mean(ia_confirmed_flag) >= self.thresholds.min_frac_confirmation
        )

        # REFUTATION
        ia_refuted = (
            np.mean(
                (1 - infos.probabilities) >= self.thresholds.min_confidence_refutation
            )
            >= self.thresholds.min_frac_refutation
        )
        uni_overlayed = (
            np.mean(infos.overlays) >= self.thresholds.min_uni_db_overlay_frac
        )

        # TODO: decide if entropy is leveraged after refutation, which would be equivalent to saying
        # that high entropy always translates refutation. This could be risky.
        if high_entropy:
            return self.codes.detailed.unsure_by_entropy
        if ia_refuted:
            if uni_overlayed:
                return self.codes.detailed.ia_refuted_and_db_overlayed
            return self.codes.detailed.ia_refuted
        if ia_confirmed:
            if uni_overlayed:
                return self.codes.detailed.both_confirmed
            return self.codes.detailed.ia_confirmed_only
        if uni_overlayed:
            return self.codes.detailed.db_overlayed_only
        return self.codes.detailed.both_unsure

    def _set_thresholds_from_pickle(self, building_validation_thresholds_pickle: str):
        """Specifiy all thresholds from serialized rules.
        This is used in thresholds optimization.

        Args:
            building_validation_thresholds_pickle (str): _description_
        """
        with open(building_validation_thresholds_pickle, "rb") as f:
            self.thresholds: thresholds = pickle.load(f)


def request_bd_uni_for_building_shapefile(
    bd_params: BDUniConnectionParams,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    srid: int,
    shapefile_path: str,
):
    """BD Uni request.

    Create a shapefile with non destructed building on the area of interest
    and saves it.
    Also add a "PRESENCE" column filled with 1 for later use by pdal.

    """
    sql_request = f"SELECT st_setsrid(batiment.geometrie,{srid}) AS geometry, 1 as presence  FROM batiment WHERE batiment.geometrie && ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid}) and not gcms_detruit"
    cmd = [
        "pgsql2shp",
        "-f",
        shapefile_path,
        "-h",
        bd_params.host,
        "-u",
        bd_params.user,
        "-P",
        bd_params.pwd,
        bd_params.bd_name,
        sql_request,
    ]
    # This call may yield
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # In empty zones, pgsql2shp does not create a shapefile
        if (
            e.output
            == b"Initializing... \nERROR: Could not determine table metadata (empty table)\n"
        ):
            return False

    # read & write to avoid unnacepted 3D shapefile format.
    gdf = geopandas.read_file(shapefile_path)
    gdf[["PRESENCE", "geometry"]].to_file(shapefile_path)

    return True


@dataclass
class thresholds:
    """The deciison threshold for cluser-level decisions."""

    min_confidence_confirmation: float
    min_frac_confirmation: float
    min_frac_confirmation_factor_if_bd_uni_overlay: float
    min_uni_db_overlay_frac: float
    min_confidence_refutation: float
    min_frac_refutation: float
    min_entropy_uncertainty: float
    min_frac_entropy_uncertain: float
