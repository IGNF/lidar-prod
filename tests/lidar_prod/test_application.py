import tempfile
import numpy as np
import pdal
import pytest
import shutil

from lidar_prod.application import apply
from lidar_prod.tasks.utils import get_las_metadata
from tests.conftest import (
    assert_las_contains_dims,
    assert_las_invariance,
    get_a_format_preserving_pdal_pipeline,
    pdal_read_las_array,
)


IN_F = "tests/files/870000_6618000.subset.postIA.las"

# With another file of country area without any building <50m.
def _isolate_and_copy(in_f: str):
    """Copy this file to be sure that the test is isolated."""
    isolated_f_copy = tempfile.NamedTemporaryFile().name
    shutil.copy(in_f, isolated_f_copy)
    return isolated_f_copy


def _isolate_and_remove_all_candidate_buildings_points(in_f: str):
    """Set Classification to 1 for all points, thus mimicking a LAS without candidates
    Consequence: no candidate groups. Nothing to update. No building completion.

    Args:
        in_f (str): input LAS path

    """
    isolated_f_copy = tempfile.NamedTemporaryFile().name
    ops = [pdal.Filter.assign(value=f"Classification = 1")]
    pipeline = get_a_format_preserving_pdal_pipeline(in_f, isolated_f_copy, ops)
    pipeline.execute()
    return isolated_f_copy


def _isolate_and_have_null_probability_everywhere(in_f: str):
    """Set building probability to 0 for all points, thus mimicking a low confidence everywhere.
    Consequences : no building in building identification. Only refuted or unsure elements.

    Args:
        in_f (str): input LAS path

    """
    isolated_f_copy = tempfile.NamedTemporaryFile().name
    ops = [pdal.Filter.assign(value=f"building = 0.0")]
    pipeline = get_a_format_preserving_pdal_pipeline(in_f, isolated_f_copy, ops)
    pipeline.execute()
    return isolated_f_copy


@pytest.mark.parametrize(
    "las_mutation",
    [
        _isolate_and_have_null_probability_everywhere,
        _isolate_and_remove_all_candidate_buildings_points,
        _isolate_and_copy,
    ],
)
def test_apply_on_subset(default_hydra_cfg, las_mutation):
    # Expected classification codes after application run.
    _fc = default_hydra_cfg.data_format.codes.building.final
    expected_codes = {
        1,
        2,
        _fc.building,
        _fc.not_building,
        _fc.unsure,
    }
    # Run application on the input data
    with tempfile.TemporaryDirectory() as default_hydra_cfg.paths.output_dir:
        in_f_isolated_copy = las_mutation(IN_F)
        default_hydra_cfg.paths.src_las = in_f_isolated_copy
        out_f = apply(default_hydra_cfg)
        assert_las_invariance(in_f_isolated_copy, out_f)
        assert_format_of_application_output_las(out_f, expected_codes)


def assert_format_of_application_output_las(out_f: str, expected_codes: dict):
    """Check LAS format, dimensions, and classification codes of output

    Args:
        out_f (str): path of output LAS
        expected_codes (dict): expected classification codes.

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
