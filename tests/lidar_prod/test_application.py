import pytest
import tempfile
import numpy as np

from lidar_prod.application import apply
from lidar_prod.tasks.utils import get_las_metadata
from tests.conftest import (
    _isolate_and_copy,
    _isolate_and_have_null_probability_everywhere,
    _isolate_and_remove_all_candidate_buildings_points,
    assert_las_contains_dims,
    check_las_invariance,
    pdal_read_las_array,
)

"""We test 

Returns:
    _type_: _description_
"""
LAS_SUBSET_FILE = "tests/files/870000_6618000.subset.postIA.las"


@pytest.mark.parametrize(
    "las_mutation",
    [
        _isolate_and_have_null_probability_everywhere,
        _isolate_and_remove_all_candidate_buildings_points,
        _isolate_and_copy,
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
        # We copy the data, and in the process we apply a "mutation" in order
        # to test for multiple scenarii.
        mutated_copy = las_mutation(LAS_SUBSET_FILE)
        default_hydra_cfg.paths.src_las = mutated_copy
        out_f = apply(default_hydra_cfg)
        check_las_invariance(mutated_copy, out_f)
        check_format_of_application_output_las(out_f, expected_codes)


def check_format_of_application_output_las(out_f: str, expected_codes: dict):
    """Check LAS format, dimensions, and classification codes of output

    Args:
        out_f (str): path of output LAS
        expected_codes (dict): set of expected classification codes.

    """
    # Check that we contain extra_dims that production needs
    assert_las_contains_dims(out_f, dims_to_check=["Group", "building", "entropy"])

    # Ensure that the format versions are as expected
    metadata = get_las_metadata(out_f)
    assert metadata["minor_version"] == 4
    assert metadata["dataformat_id"] == 8
    # Ensure that the final spatial reference is French CRS Lambert-93
    assert "Lambert-93" in metadata["spatialreference"]

    # Check that we have either 1/2 (ground/unclassified),
    # or one of the three final classification code of the module
    arr1 = pdal_read_las_array(out_f)
    actual_codes = {*np.unique(arr1["Classification"])}
    assert actual_codes.issubset(expected_codes)
