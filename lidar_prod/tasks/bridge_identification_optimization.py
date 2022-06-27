"""
Takes bridge probabilities as input, and defines bridge.

"""

import logging
from numbers import Number
import tempfile
from typing import Any, Dict
from glob import glob
import os.path as osp
import geopandas

# from shapely.geometry import Polygon
from shapely import wkt
import numpy as np
import pdal
from lidar_prod.tasks.bridge_identification import BridgeIdentifier
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


def load_json_unary_union(json_path_target):
    return (
        geopandas.read_file(json_path_target)
        .fillna(value=wkt.loads("POLYGON EMPTY"))
        .unary_union
    )


class BridgeIdentificationOptimizer:
    def __init__(
        self,
        paths: Dict[str, str],
        bridge_identifier: BridgeIdentifier,
        gdal_writer_window_size: Number = 0.25,
        gdal_writer_resolution: Number = 0.25,
    ):
        self.paths = paths
        self.bri = bridge_identifier
        self.gdal_writer_window_size = gdal_writer_window_size
        self.gdal_writer_resolution = gdal_writer_resolution

    def evaluate_one_iou(self, input_las: str, output_las_path: str):
        """Performe bridge evaluation on one file and evaluate vector IoU."""
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

    def evaluate(self) -> None:
        """Iterate through las_filepaths to perform bridge identification and evaluate resulting vector IoU."""
        # TODO: more detailed statistics by file in a dataframe
        ious = []
        for input_las_path in glob(osp.join(self.paths.input_las_dir, "*.las")):
            output_las_path = osp.join(
                self.paths.output_las_dir, osp.basename(input_las_path)
            )
            iou = self.evaluate_one_iou(input_las_path, output_las_path)
            print(iou)
            ious += [iou]
        return np.mean(ious)

    def vectorize_bridge(self, las_path, out_json) -> None:
        out_tif = out_json.replace(".json", ".tif")

        bridge_code = self.bri.data_format.codes.bridge
        pipeline = get_pdal_reader(las_path) | pdal.Filter.range(
            limits=f"Classification[{bridge_code}:{bridge_code}]"
        )
        pipeline.execute()
        points = pipeline.arrays[0]
        if len(points) == 0:
            # no building points in this LAS, so we create an empty geometry
            s = geopandas.GeoDataFrame({"geometry": [wkt.loads("POLYGON EMPTY")]})
            s.to_file(out_json)
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
        """Save the LAS with a nullified Classification dim to avoid conflict when updating Classification."""
        pipeline = get_a_las_to_las_pdal_pipeline(
            input_las,
            output_las,
            [pdal.Filter.assign(value="Classification = 0")],
        )
        pipeline.execute()
