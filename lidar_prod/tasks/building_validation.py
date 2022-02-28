from dataclasses import dataclass
import logging
import os
import os.path as osp
import json
import pickle
import subprocess
import numpy as np
import pdal
import geopandas
import laspy
from tqdm import tqdm
from lidar_prod.tasks.utils import (
    BDUniConnectionParams,
    extract_coor,
    split_idx_by_dim,
    tempdir,
)

log = logging.getLogger(__name__)


class BuildingValidator:
    """Logic of building validation."""

    def __init__(
        self,
        bd_uni_connection_params=None,
        cluster=None,
        bd_uni_request=None,
        rules=None,
        building_validation_thresholds_pickle: str = None,
        codes=None,
        candidate_buildings_codes: int = [202],
        use_final_classification_codes: bool = True,
        data_format=None,
    ):
        self.bd_uni_connection_params = bd_uni_connection_params
        self.cluster = cluster
        self.bd_uni_request = bd_uni_request
        self.candidate_buildings_codes = candidate_buildings_codes
        self.use_final_classification_codes = use_final_classification_codes
        self.rules = rules  # default values
        self.codes = codes
        self.data_format = data_format

        self.setup(building_validation_thresholds_pickle)

    def setup(self, building_validation_thresholds_pickle):
        """Setup, loading optimized thresholds if available."""
        if osp.exists(building_validation_thresholds_pickle):
            self._set_rules_from_pickle(building_validation_thresholds_pickle)
            log.info(f"Using best trial from: {building_validation_thresholds_pickle}")
        else:
            log.warning(
                "Using config decision thresholds - specify "
                "'building_validation.application.building_validation_thresholds_pickle' "
                "to use optimized threshold"
            )

        self.detailed_to_final = {
            self.codes.detailed.unclustered: self.codes.final.not_building,
            self.codes.detailed.ia_refuted: self.codes.final.not_building,
            self.codes.detailed.ia_refuted_and_db_overlayed: self.codes.final.unsure,
            self.codes.detailed.both_unsure: self.codes.final.unsure,
            self.codes.detailed.ia_confirmed_only: self.codes.final.building,
            self.codes.detailed.db_overlayed_only: self.codes.final.building,
            self.codes.detailed.both_confirmed: self.codes.final.building,
        }
        self.detailed_to_final_mapper = np.vectorize(
            lambda detailed_code: self.detailed_to_final.get(detailed_code),
            otypes=[np.int],
        )

    @tempdir()
    def run(
        self,
        in_f: str,
        out_f: str,
        tempdir: str = "for_prepared_las_and_given_by_decorator",
    ):
        """Application.

        Transform cloud at `in_f` following validation logic, and save it to
        `out_f`

        Args:
            in_f (str): path to input LAS file with a building probability channel
            out_f (str): path for saving updated LAS file.
            tempdir (str, optional): This is a path to a temporary directory created
        by the decorator, which is automatically deleted afterward. Used to store intermediary,
        prepared LAS file.

        Returns:
            _type_: returns `out_f` for potential terminal piping.

        """
        log.info(f"Applying Building Validation to file \n{in_f}")
        log.info("Preparation - Clustering + Requesting Building database")
        temp_f = osp.join(tempdir, osp.basename(in_f))
        self.prepare(in_f, temp_f)
        log.info("Using AI and Databases to update cloud Classification")
        self.update(temp_f, out_f)
        log.info(f"Saved to\n{out_f}")
        return out_f

    @tempdir()
    def prepare(
        self,
        in_f: str,
        out_f: str,
        tempdir: str = "for_shapefile_and_given_by_decorator",
    ):
        """
        Prepare las for later decision process.

        1. Cluster candidates points.
        2. Identify points overlayed by a BD Uni building.


        """

        _flag = self.data_format.las_channel_names.candidate_buildings_flag
        _cluster_id = self.data_format.las_channel_names.cluster_id
        _groups = self.data_format.las_channel_names.macro_candidate_building_groups
        _overlay = self.data_format.las_channel_names.uni_db_overlay

        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader(
            in_f,
            type="readers.las",
            nosrs=True,
            override_srs=self.data_format.crs_prefix + str(self.data_format.crs),
        )
        pipeline |= pdal.Filter.ferry(dimensions=f"=>{_flag}")
        conditional_expression = (
            "("
            + " || ".join(
                f"Classification == {int(candidat_code)}"
                for candidat_code in self.candidate_buildings_codes
            )
            + ")"
        )
        pipeline |= pdal.Filter.assign(
            value=f"{_flag} = 1 WHERE {conditional_expression}"
        )
        pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            where=f"{_flag} == 1",
        )
        # TODO: might be removed if we do not keep ClusterID.
        pipeline |= pdal.Filter.ferry(dimensions=f"{_cluster_id}=>{_groups}")
        # reset to avoid crash with future clustering
        pipeline |= pdal.Filter.assign(value=f"{_cluster_id} = 0")

        shapefile_path = os.path.join(tempdir, "temp.shp")
        buildings_in_bd_topo = request_bd_uni_for_building_shapefile(
            self.bd_uni_connection_params,
            *extract_coor(
                os.path.basename(in_f),
                self.data_format.tile_size_meters,
                self.data_format.tile_size_meters,
                self.bd_uni_request.buffer,
            ),
            self.data_format.crs,
            shapefile_path,
        )

        # Channel is always created even if there are no buildings in database.
        pipeline |= pdal.Filter.ferry(dimensions=f"=>{_overlay}")
        if buildings_in_bd_topo:
            pipeline |= pdal.Filter.overlay(
                column="PRESENCE", datasource=shapefile_path, dimension=_overlay
            )
        pipeline |= pdal.Writer(
            type="writers.las", filename=out_f, forward="all", extra_dims="all"
        )
        os.makedirs(osp.dirname(out_f), exist_ok=True)
        pipeline.execute()

    def update(self, prepared_f: str, out_f: str):
        """Update point cloud classification channel."""

        las = laspy.read(prepared_f)
        # 1) Set all candidates points to a single class
        _clf = self.data_format.las_channel_names.classification
        _flag = self.data_format.las_channel_names.candidate_buildings_flag
        candidates_idx = las[_flag] == 1
        las[_clf][candidates_idx] = self.codes.detailed.unclustered

        # 2) Decide at the group-level
        split_idx = split_idx_by_dim(
            las[self.data_format.las_channel_names.macro_candidate_building_groups]
        )
        split_idx = split_idx[1:]  # removes unclustered group with ClusterID = 0
        for pts_idx in tqdm(
            split_idx, desc="Update groups of candidate buildings", unit="grp"
        ):
            pts = las.points[pts_idx]
            detailed_code = self._make_detailed_group_decision(
                pts[self.data_format.las_channel_names.ai_building_proba],
                pts[self.data_format.las_channel_names.uni_db_overlay],
            )
            las[_clf][pts_idx] = detailed_code

        if self.use_final_classification_codes:
            las[_clf][candidates_idx] = self.detailed_to_final_mapper(
                las[_clf][candidates_idx]
            )
        os.makedirs(osp.dirname(out_f), exist_ok=True)
        las.write(out_f)

    def _make_group_decision(self, *args, **kwargs):
        detailed_code = self._make_detailed_group_decision(*args, **kwargs)
        return self.detailed_to_final[detailed_code]

    def _make_detailed_group_decision(self, probas_arr, overlay_bools_arr):
        """Decision process at the cluster level.

        Confirm or refute candidate building shape based on fraction of confirmed/refuted points and
        on fraction of points overlayed by a building shape in a database.

        """
        ia_confirmed = (
            np.mean(probas_arr >= self.rules.min_confidence_confirmation)
            >= self.rules.min_frac_confirmation
        )
        ia_refuted = (
            np.mean((1 - probas_arr) >= self.rules.min_confidence_refutation)
            >= self.rules.min_frac_refutation
        )
        uni_overlayed = np.mean(overlay_bools_arr) >= self.rules.min_uni_db_overlay_frac

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

    def _set_rules_from_pickle(self, building_validation_thresholds_pickle):
        with open(building_validation_thresholds_pickle, "rb") as f:
            self.rules: rules = pickle.load(f)


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
class rules:
    """The deciison threshold for cluser-level decisions."""

    min_confidence_confirmation: float
    min_frac_confirmation: float
    min_uni_db_overlay_frac: float
    min_confidence_refutation: float
    min_frac_refutation: float
