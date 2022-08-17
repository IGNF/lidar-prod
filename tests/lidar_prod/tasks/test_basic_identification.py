import numpy as np
import pytest

from lidar_prod.tasks.basic_identification import BasicIdentifier
from lidar_prod.tasks.utils import get_las_data_from_las

LAS_SUBSET_FILE_VEGETATION = "tests/files/436000_6478000.subset.postIA.las"


def test_basic_identifier(vegetation_unclassifed_hydra_cfg):
    las_data = get_las_data_from_las(LAS_SUBSET_FILE_VEGETATION)

    basic_identifier = BasicIdentifier(
        vegetation_unclassifed_hydra_cfg["basic_identification"]["vegetation_threshold"],
        vegetation_unclassifed_hydra_cfg["data_format"]["las_dimensions"]["ai_vegetation_proba"],
        vegetation_unclassifed_hydra_cfg["data_format"]["las_dimensions"]["ai_vegetation_unclassified_groups"],
        vegetation_unclassifed_hydra_cfg["data_format"]["codes"]["vegetation"],
        True,
        vegetation_unclassifed_hydra_cfg["data_format"]["las_dimensions"]["classification"],
        vegetation_unclassifed_hydra_cfg["data_format"]["codes"]["vegetation_truth"],
        )
    basic_identifier.identify(las_data)
    vegetation_count = np.count_nonzero(
        las_data.points[vegetation_unclassifed_hydra_cfg["data_format"]["las_dimensions"]["ai_vegetation_unclassified_groups"]] ==
        vegetation_unclassifed_hydra_cfg["data_format"]["codes"]["vegetation"])
    assert vegetation_count == 65824
    assert 0.99 < basic_identifier.iou.iou
