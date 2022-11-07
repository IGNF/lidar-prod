import numpy as np
import pytest
from hydra.experimental import compose, initialize

from lidar_prod.tasks.utils import pdal_read_las_array


@pytest.fixture
def vegetation_unclassifed_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(
            config_name="config",
            overrides=[
                "data_format=vegetation_unclassified.yaml",
                "basic_identification=for_testing.yaml",
                "paths.src_las=tests/files/436000_6478000.subset.postIA.las",
            ],
        )


@pytest.fixture
def hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config", overrides=["data_format=default.yaml"])


def check_las_invariance(las_path1, las_path2):
    TOLERANCE = 0.0001

    array1 = pdal_read_las_array(las_path1)
    array2 = pdal_read_las_array(las_path2)
    key_dims = ["X", "Y", "Z", "Infrared", "Red", "Blue", "Green", "Intensity"]
    assert array1.shape == array2.shape  # no loss of points
    assert all(
        dim in array2.dtype.fields.keys() for dim in key_dims
    )  # key dimensions are here

    # order of points is allowed to change, so we assess a relaxed equality.
    for dim in key_dims:
        assert pytest.approx(np.min(array2[dim]), TOLERANCE) == np.min(array1[dim])
        assert pytest.approx(np.max(array2[dim]), TOLERANCE) == np.max(array1[dim])
        assert pytest.approx(np.mean(array2[dim]), TOLERANCE) == np.mean(array1[dim])
        assert pytest.approx(np.sum(array2[dim]), TOLERANCE) == np.sum(array1[dim])


def check_las_contains_dims(las1, dims_to_check=[]):
    a1 = pdal_read_las_array(las1)
    for d in dims_to_check:
        assert d in a1.dtype.fields.keys()
