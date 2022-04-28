import pdal
import pytest
import tempfile
import numpy as np

from lidar_prod.application import apply
from lidar_prod.tasks.utils import get_a_las_to_las_pdal_pipeline, get_las_metadata
from tests.conftest import (
    check_las_contains_dims,
    check_las_invariance,
    pdal_read_las_array,
)

"""We test the application against a LAS subset (~50m²) with a few buildings and a few
classification mistakes. The data contains the necessary fields (building probability, entropy)
for validation.
We apply different "mutations" to the data in order to test for multiple scenarii.

"""
LAS_SUBSET_FILE = "tests/files/870000_6618000.subset.postIA.las"


@pytest.mark.parametrize(
    "las_mutation",
    [
        [],  # identity
        [pdal.Filter.assign(value="building = 0.0")],  # low proba everywhere
        [pdal.Filter.assign(value="Classification = 1")],  # no candidate buildings
        [pdal.Filter.assign(value="Classification = 202")],  # only candidate buildings
    ],
)
def test_application_data_invariance_and_data_format(default_hydra_cfg, las_mutation):
    # Expected classification codes after application run are either default=0, unclassified=1, or
    # one of the decision codes.
    _fc = default_hydra_cfg.data_format.codes.building.final
    expected_codes = {
        1,
        2,
        _fc.building,
        _fc.not_building,
        _fc.unsure,
    }
    # Run application on the data subset
    with tempfile.TemporaryDirectory() as default_hydra_cfg.paths.output_dir:
        # Copy the data and apply the "mutation"
        mutated_copy: str = tempfile.NamedTemporaryFile().name
        get_a_las_to_las_pdal_pipeline(
            LAS_SUBSET_FILE, mutated_copy, las_mutation
        ).execute()
        default_hydra_cfg.paths.src_las = mutated_copy
        updated_las_path: str = apply(default_hydra_cfg)
        # Check output
        check_las_invariance(mutated_copy, updated_las_path)
        check_format_of_application_output_las(updated_las_path, expected_codes)


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
