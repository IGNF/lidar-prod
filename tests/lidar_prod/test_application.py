import pdal
import pytest
import tempfile
import numpy as np

from lidar_prod.application import apply, just_clean, detect_vegetation_unclassified
from lidar_prod.tasks.utils import get_a_las_to_las_pdal_pipeline, get_las_metadata, get_las_data_from_las
from tests.conftest import (
    check_las_contains_dims,
    check_las_invariance,
    pdal_read_las_array,
)

"""We test the application against a LAS subset (~2500m²) with a few buildings and a few
classification mistakes. The data contains the necessary fields (building probability, entropy)
for validation.
We apply different "mutations" to the data in order to test for multiple scenarii.
"""
LAS_SUBSET_FILE_BUILDING = "tests/files/870000_6618000.subset.postIA.las"
SHAPE_FILE = "tests/files/870000_6618000.subset.postIA.shp"
LAS_SUBSET_FILE_VEGETATION = "tests/files/436000_6478000.subset.postIA.las"


@pytest.mark.parametrize(
    "las_mutation, query_db_Uni",
    [
        ([], True),  # identity
        ([pdal.Filter.assign(value="building = 0.0")], True),  # low proba everywhere
        ([pdal.Filter.assign(value="Classification = 1")], False),  # no candidate buildings
        ([pdal.Filter.assign(value="Classification = 202")], False),  # only candidate buildings
    ],  # if query_db_Uni = True, will query database to get a shapefile, otherwise use a prebuilt one
)
def test_application_data_invariance_and_data_format(legacy_hydra_cfg, las_mutation, query_db_Uni):
    # Expected classification codes after application run are either default=0, unclassified=1, or
    # one of the decision codes.
    _fc = legacy_hydra_cfg.data_format.codes.building.final
    expected_codes = {
        1,
        2,
        _fc.building,
        _fc.not_building,
        _fc.unsure,
    }
    # Run application on the data subset
    with tempfile.TemporaryDirectory() as legacy_hydra_cfg.paths.output_dir:
        # Copy the data and apply the "mutation"
        mutated_copy: str = tempfile.NamedTemporaryFile().name
        get_a_las_to_las_pdal_pipeline(
            LAS_SUBSET_FILE_BUILDING, mutated_copy, las_mutation
        ).execute()
        legacy_hydra_cfg.paths.src_las = mutated_copy
        if not query_db_Uni:    # we don't request db_uni, we use a shapefile instead
            legacy_hydra_cfg.building_validation.application.shp_path = SHAPE_FILE
        updated_las_path_list: str = apply(legacy_hydra_cfg)
        # Check output
        check_las_invariance(mutated_copy, updated_las_path_list[0])
        check_format_of_application_output_las(updated_las_path_list[0], expected_codes)


def check_format_of_application_output_las(output_las_path: str, expected_codes: dict):
    """Check LAS format, dimensions, and classification codes of output

    Args:
        output_las_path (str): path of output LAS
        expected_codes (dict): set of expected classification codes.

    """
    # Check that we contain extra_dims that production needs
    check_las_contains_dims(
        output_las_path, dims_to_check=["Group", "building", "entropy"]
    )

    # Ensure that the format versions are as expected
    check_las_format_versions_and_srs(output_las_path)

    # Check that we have either 1/2 (ground/unclassified),
    # or one of the three final classification code of the module
    arr1 = pdal_read_las_array(output_las_path)
    actual_codes = {*np.unique(arr1["Classification"])}
    assert actual_codes.issubset(expected_codes)


def check_las_format_versions_and_srs(las_path):
    metadata = get_las_metadata(las_path)
    assert metadata["minor_version"] == 4
    assert metadata["dataformat_id"] == 8
    # Ensure that the final spatial reference is French CRS Lambert-93
    assert "Lambert-93" in metadata["spatialreference"]


def test_just_clean(vegetation_unclassifed_hydra_cfg):
    destination_path = tempfile.NamedTemporaryFile().name
    just_clean(vegetation_unclassifed_hydra_cfg, LAS_SUBSET_FILE_VEGETATION, destination_path)
    las_data = get_las_data_from_las(destination_path)
    assert [dim for dim in las_data.point_format.extra_dimension_names] == ['entropy', 'vegetation', 'unclassified']


def test_detect_vegetation_unclassified(vegetation_unclassifed_hydra_cfg):
    destination_path = tempfile.NamedTemporaryFile().name
    detect_vegetation_unclassified(
        vegetation_unclassifed_hydra_cfg,
        LAS_SUBSET_FILE_VEGETATION,
        destination_path)
    las_data = get_las_data_from_las(destination_path)
    vegetation_count = np.count_nonzero(las_data.points.classification == vegetation_unclassifed_hydra_cfg.data_format.codes.vegetation)
    unclassified_count = np.count_nonzero(las_data.points.classification == vegetation_unclassifed_hydra_cfg.data_format.codes.unclassified)
    assert vegetation_count == 17
    assert unclassified_count == 23222
