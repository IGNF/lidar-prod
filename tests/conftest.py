import numpy as np
import pyproj
import pytest
from hydra import compose, initialize

from lidar_prod.tasks.utils import get_pipeline, pdal_read_las_array


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
        return compose(
            config_name="config",
            overrides=[
                "data_format=default.yaml",
                "building_validation/optimization=pytest.yaml",
            ],
        )


def check_las_invariance(las_path1, las_path2, epsg):
    TOLERANCE = 0.0001

    array1, _ = pdal_read_las_array(las_path1, epsg)
    array2, _ = pdal_read_las_array(las_path2, epsg)
    key_dims = ["X", "Y", "Z", "Infrared", "Red", "Blue", "Green", "Intensity"]
    assert array1.shape == array2.shape  # no loss of points
    assert all(dim in array2.dtype.fields.keys() for dim in key_dims)  # key dimensions are here

    # order of points is allowed to change, so we assess a relaxed equality.
    for dim in key_dims:
        assert pytest.approx(np.min(array2[dim]), TOLERANCE) == np.min(array1[dim])
        assert pytest.approx(np.max(array2[dim]), TOLERANCE) == np.max(array1[dim])
        assert pytest.approx(np.mean(array2[dim]), TOLERANCE) == np.mean(array1[dim])
        assert pytest.approx(np.sum(array2[dim]), TOLERANCE) == np.sum(array1[dim])


def check_las_contains_dims(las1, epsg, dims_to_check=[]):
    a1, _ = pdal_read_las_array(las1, epsg)
    for d in dims_to_check:
        assert d in a1.dtype.fields.keys()


def check_las_format_versions_and_srs(input_path: str, epsg: int | str):
    _, metadata = get_pipeline(input_path, epsg=None)  # do not enforce epsg when reading the data
    assert metadata["minor_version"] == 4
    assert metadata["dataformat_id"] == 8
    # Ensure that the final spatial reference is the same as in the config (if provided)
    metadata_crs = metadata["srs"]["compoundwkt"]
    assert metadata_crs, f"Non-empty CRS string expected, got {metadata_crs}"
    if epsg:
        expected_crs = pyproj.crs.CRS(epsg)
        assert expected_crs.equals(metadata_crs)


def check_expected_classification(output_las_path: str, expected_codes: set):
    """Check classification codes of output

    Args:
        output_las_path (str): path of output LAS
        expected_codes (dict): set of expected classification codes.

    """
    arr1, _ = pdal_read_las_array(output_las_path)
    actual_codes = {*np.unique(arr1["Classification"])}
    assert actual_codes.issubset(
        expected_codes
    ), f"Expected classification: {expected_codes}, got: {actual_codes}"
