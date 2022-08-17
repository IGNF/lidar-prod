import numpy as np
import pytest
import laspy
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
                "paths.src_las=tests/files/436000_6478000.subset.postIA.las"])


@pytest.fixture
def legacy_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config", overrides=["data_format=legacy.yaml"])


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


@pytest.fixture
def las_data():
    points_count = 100
    classification_possibilities = [0, 3, 6]
    length_x = 1000
    length_y = 1000
    length_z = 100

    # data:
    x_data = np.random.rand(points_count) * length_x
    y_data = np.random.rand(points_count) * length_y
    z_data = np.random.rand(points_count) * length_z

    # header
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="classification", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="entropy", type=np.float))
    header.add_extra_dim(laspy.ExtraBytesParams(name="vegetation", type=np.float))
    header.add_extra_dim(laspy.ExtraBytesParams(name="unclassified", type=np.float))
    header.offsets = [np.floor(np.min(x_data)), np.floor(np.min(y_data)), np.floor(np.min(z_data))]
    header.scales = np.array([0.1, 0.1, 0.1])

    # las
    las_data = laspy.LasData(header)
    las_data.x = x_data
    las_data.y = y_data
    las_data.z = z_data
    las_data.classification = np.random.choice(classification_possibilities, points_count)
    las_data.entropy = np.random.rand(points_count)
    las_data.vegetation = np.random.rand(points_count)
    las_data.unclassified = np.random.rand(points_count)

    return las_data
