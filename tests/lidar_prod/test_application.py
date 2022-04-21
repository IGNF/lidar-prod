import tempfile
import pytest

from lidar_prod.application import apply
from lidar_prod.tasks.utils import get_las_metadata
from tests.conftest import assert_las_contains_dims, assert_las_invariance

# test a full run with no error

IN_F = "tests/files/870000_6618000.subset.postIA.las"


def test_run_on_subset(default_hydra_cfg):
    default_hydra_cfg.paths.src_las = IN_F
    # TODO: ignore the warning here.
    with tempfile.TemporaryDirectory() as td:
        default_hydra_cfg.paths.output_dir = td
        out_f = apply(default_hydra_cfg)

        # Check that key dimensions are unchanged
        assert_las_invariance(IN_F, out_f)

        # Check that we contain extra_dims that production needs
        assert_las_contains_dims(out_f, dims_to_check=["Group", "building", "entropy"])

        # Ensure that the data format is the right one.
        metadata = get_las_metadata(out_f)
        assert metadata["minor_version"] == 4
        assert metadata["dataformat_id"] == 8

        # Ensure that the final spatial reference is French CRS Lambert-93
        assert "Lambert-93" in metadata["spatialreference"]
