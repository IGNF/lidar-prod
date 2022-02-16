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
    # TODO: replace data_format with a dataclass object .
    def __init__(
        self,
        bd_uni_connection_params=None,
        cluster=None,
        bd_uni_request=None,
        rules=None,
        building_validation_thresholds_pickle: str = None,
        codes=None,
        candidate_buildings_codes: int = [6],
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
        if osp.exists(building_validation_thresholds_pickle):
            self.set_rules_from_pickle(building_validation_thresholds_pickle)
            log.info(f"Using best trial from: {building_validation_thresholds_pickle}")
        else:
            log.info(f"Using config decision thresholds")

        self.codes.detailed_to_final = {
            self.codes.detailed.unclustered: self.codes.final.not_building,
            self.codes.detailed.ia_refuted: self.codes.final.not_building,
            self.codes.detailed.ia_refuted_and_db_overlayed: self.codes.final.unsure,
            self.codes.detailed.both_unsure: self.codes.final.unsure,
            self.codes.detailed.ia_confirmed_only: self.codes.final.building,
            self.codes.detailed.db_overlayed_only: self.codes.final.building,
            self.codes.detailed.both_confirmed: self.codes.final.building,
        }

    @tempdir()
    def prepare(
        self,
        input_filepath: str,
        output_filepath: str,
        temporary_dir: str = "given_by_decorator",
    ):
        """
        Prepare las for later decision process.
        Will:
        - Cluster candidates points, thus creating a ClusterId channel (default cluster: 0).
        - Identify points overlayed by a BDTopo shape, thus creating a BDTopoOverlay channel (no overlap: 0).
        """

        shapefile_path = os.path.join(temporary_dir, "temp.shp")

        buildings_in_bd_topo = request_bd_uni_for_building_shapefile(
            self.bd_uni_connection_params,
            *extract_coor(
                os.path.basename(input_filepath),
                self.data_format.tile_size_meters,
                self.data_format.tile_size_meters,
                self.bd_uni_request.buffer,
            ),
            self.data_format.crs,
            shapefile_path,
        )

        _reader = [
            {
                "type": "readers.las",
                "filename": input_filepath,
                "override_srs": self.data_format.crs_prefix + str(self.data_format.crs),
                "nosrs": True,
            }
        ]
        which_points_to_cluster = (
            "("
            + " || ".join(
                f"Classification == {int(candidat_code)}"
                for candidat_code in self.candidate_buildings_codes
            )
            + ")"
        )
        _cluster = [
            {
                "type": "filters.cluster",
                "min_points": self.cluster.min_points,
                "tolerance": self.cluster.tolerance,
                "where": which_points_to_cluster,
            }
        ]
        _topo_overlay = [
            {
                "type": "filters.ferry",
                "dimensions": f"=>{self.data_format.las_channel_names.uni_db_overlay}",
            }
        ]
        if buildings_in_bd_topo:
            _topo_overlay.append(
                {
                    "column": "PRESENCE",
                    "datasource": shapefile_path,
                    "dimension": f"{self.data_format.las_channel_names.uni_db_overlay}",
                    "type": "filters.overlay",
                },
            )
        _writer = [
            {
                "type": "writers.las",
                "filename": output_filepath,
                "forward": "all",  # keep all dimensions based on input format
                "extra_dims": "all",  # keep all extra dims as well
            }
        ]
        pipeline = {"pipeline": _reader + _cluster + _topo_overlay + _writer}
        pipeline = json.dumps(pipeline)
        pipeline = pdal.Pipeline(pipeline)
        pipeline.execute()

    def make_group_decision(self, *args, **kwargs):
        detailed_code = self.make_detailed_group_decision(*args, **kwargs)
        return self.codes.detailed_to_final[detailed_code]

    def make_detailed_group_decision(self, probas_arr, overlay_bools_arr):
        """
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

    def update(self, prepared_las_path: str, output_las_path: str):
        """
        Update point cloud classification channel.
        Params is a dict-like object with optimized decision thresholds.
        """
        las = laspy.read(prepared_las_path)
        # 1) Set to default all candidats points
        # TODO: check if that logic is ok in new production process.
        candidate_building_points_mask = (
            las[self.data_format.las_channel_names.classification]
            == self.candidate_buildings_codes
        )
        las[self.data_format.las_channel_names.classification][
            candidate_building_points_mask
        ] = self.data_format.codes.unclassified

        # 2) Decide at the group-level
        split_idx = split_idx_by_dim(las[self.data_format.las_channel_names.cluster_id])
        # TODO: make it more robust ? Assurance that it is ordered?
        split_idx = split_idx[1:]  # remove unclustered group with ClusterID = 0
        for pts_idx in tqdm(split_idx, desc="Updating LAS."):
            pts = las.points[pts_idx]
            detailed_code = self.make_detailed_group_decision(
                pts[self.data_format.las_channel_names.ai_building_proba],
                pts[self.data_format.las_channel_names.uni_db_overlay],
            )
            if self.use_final_classification_codes:
                las[self.data_format.las_channel_names.classification][
                    pts_idx
                ] = self.codes.detailed_to_final[detailed_code]
            else:
                las[self.data_format.las_channel_names.classification][
                    pts_idx
                ] = detailed_code
        las.write(output_las_path)
        return las

    def set_rules_from_pickle(self, building_validation_thresholds_pickle):
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
    """
    Create a shapefile with non destructed building on
    the area and saves it. Also add a column "presence" with only 1 in it
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
    min_confidence_confirmation: float
    min_frac_confirmation: float
    min_uni_db_overlay_frac: float
    min_confidence_refutation: float
    min_frac_refutation: float
