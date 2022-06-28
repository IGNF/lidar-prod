"""
Takes bridge probabilities as input, and defines bridge.

"""

import logging
from numbers import Number
import tempfile
from typing import Dict, List
from glob import glob
import os.path as osp
import geopandas
import optuna

from shapely.geometry import Polygon
import numpy as np
import pdal
from lidar_prod.tasks.bridge_identification import BridgeIdentifier, thresholds
from lidar_prod.tasks.utils import (
    LAMBERT_93_EPSG_STR,
    LAMBERT_93_SRID,
    gdal_polygonize,
    get_a_las_to_las_pdal_pipeline,
    get_pdal_reader,
)

log = logging.getLogger(__name__)


def compute_bridge_iou(json_path_target, json_path_predicted) -> float:
    # load the jsons with geopandas
    # compute IoU
    target = load_json_unary_union(json_path_target)
    predicted = load_json_unary_union(json_path_predicted)
    target_area = target.area
    predicted_area = predicted.area
    if target_area == 0 and predicted_area == 0:
        # There are no bridge in this area and no false positive
        return 1.0

    return target.intersection(predicted).area / target.union(predicted).area


def save_geometries_to_geodataframe(geometry_list: List[Polygon], out_json: str):
    "Save a list of geometries to a geojson file.."
    s = geopandas.GeoDataFrame({"geometry": geometry_list})
    s.to_file(out_json)


def load_json_unary_union(json_path_target):
    """Get the unary union of a geojson in a way that is robust to null values.

    Note: filling na values is needed because empty polygons saved by geopandas are
    read as null value, which is unexpected according to
    https://geopandas.org/en/stable/docs/user_guide/missing_empty.html.

    """
    return geopandas.read_file(json_path_target).fillna(value=Polygon([])).unary_union


class BridgeIdentificationOptimizer:
    def __init__(
        self,
        paths: Dict[str, str],
        bridge_identifier: BridgeIdentifier,
        optimization_design: Dict,
        study: optuna.Study,
        gdal_writer_window_size: Number = 0.25,
        gdal_writer_resolution: Number = 0.25,
    ):
        self.paths = paths
        self.bri = bridge_identifier
        self.optimization_design = optimization_design
        self.study = study
        self.gdal_writer_window_size = gdal_writer_window_size
        self.gdal_writer_resolution = gdal_writer_resolution

    def optimize(self):
        """Optimize decision thresholds for bridge identification.

        Runs the genetic algorithm for N generations.
        For each set of decision thresholds, computes the iou on all files.
        Finally, select the optimal thresholds.

        """
        self.study.optimize(self._optuna_objective_func, n_trials=self.design.n_trials)
        best_thresholds = self._select_best_thresholds(self.study)
        log.info(f"Best_trial thresholds: \n{best_thresholds}")
        # TODO: save thresholds to a pickle ?
        # Perform an evaluation step wit the best thresholds to get results files for inspection.
        self.bri.thresholds = best_thresholds
        iou = self.bri.evaluate()
        log.info(f"Maximized vector IoU is {iou}")

    def _optuna_objective_func(self, trial):
        """Sets decision threshold for the trial and computes resulting vector IoU."""
        self.bri.thresholds = thresholds(
            min_bridge_proba=trial.suggest_float(
                "min_confidence_confirmation", 0.0, 1.0
            )
        )
        return self.bri.evaluate()

    def evaluate_mean_bridge_iou_across_data(self) -> None:
        """Iterates through las_filepaths to perform bridge identification and evaluate resulting vector IoU."""
        ious = []
        for input_las_path in glob(osp.join(self.paths.input_las_dir, "*.las")):
            output_las_path = osp.join(
                self.paths.output_las_dir, osp.basename(input_las_path)
            )
            iou = self.evaluate_one_iou(input_las_path, output_las_path)
            print(iou)
            ious += [iou]
        return np.mean(ious)

    def evaluate_one_iou(self, input_las: str, output_las_path: str):
        """Performs bridge vector IoU evaluation on a single file."""
        # Set Classification to 0.0 in a temporary file to avoid conflicts at evaluation time.
        tmp_las = tempfile.NamedTemporaryFile(suffix=".las").name
        self.nullify_classification(input_las, tmp_las)
        # run the bridge identification and save LAS with updated Classification channel
        self.bri.run(tmp_las, output_las_path)
        # vectorize bridge points from input and output LAS files
        out_json_target = output_las_path.replace(".las", ".target.json")
        out_json_predicted = output_las_path.replace(".las", ".predicted.json")
        self.vectorize_bridge(input_las, out_json_target)
        self.vectorize_bridge(output_las_path, out_json_predicted)
        # compare results to the input labels
        iou = compute_bridge_iou(out_json_target, out_json_predicted)
        return iou

    def vectorize_bridge(self, las_path, out_json) -> None:
        """Vectorizes bridge points into a geojson."""
        out_tif = out_json.replace(".json", ".tif")

        bridge_code = self.bri.data_format.codes.bridge
        pipeline = get_pdal_reader(las_path) | pdal.Filter.range(
            limits=f"Classification[{bridge_code}:{bridge_code}]"
        )
        pipeline.execute()
        points = pipeline.arrays[0]
        if len(points) == 0:
            # no building points in this LAS, so we create an empty geometry
            save_geometries_to_geodataframe([Polygon([])], out_json)
            return
        # else we rasterize the bridge points inot a TIF that we then vectorize
        pipeline = pdal.Writer.gdal(
            filename=out_tif,
            dimension="Classification",
            data_type="uint",
            output_type="max",
            window_size=self.gdal_writer_window_size,
            resolution=self.gdal_writer_resolution,
            nodata=0,
            override_srs=LAMBERT_93_EPSG_STR,
        ).pipeline(points)
        # pdal.Writer.las(
        #     filename=out_laz,
        #     extra_dims="all",
        #     minor_version=4,
        #     dataformat_id=8,
        #     a_srs=LAMBERT_93_EPSG_STR,
        # ),
        pipeline.execute()
        gdal_polygonize(out_tif, out_json, epsg_out=LAMBERT_93_SRID)

    def nullify_classification(self, input_las, output_las):
        """Saves the LAS with a nullified Classification dim to avoid conflict when updating Classification."""
        pipeline = get_a_las_to_las_pdal_pipeline(
            input_las,
            output_las,
            [pdal.Filter.assign(value="Classification = 0")],
        )
        pipeline.execute()

    def _select_best_thresholds(self, study):
        """Gets the trial that maximizes IoU."""
        trials = sorted(study.best_trials, key=lambda x: x.values[0], reverse=True)
        return trials[0]
