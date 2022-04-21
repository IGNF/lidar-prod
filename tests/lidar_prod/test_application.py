import tempfile
import pytest

from lidar_prod.application import apply

# test a full run with no error

IN_F = "tests/files/870000_6618000.subset.postIA.las"


def test_run_on_subset(default_hydra_cfg):
    default_hydra_cfg.paths.src_las = IN_F
    # TODO: ignore the warning here.
    with tempfile.TemporaryDirectory() as td:
        default_hydra_cfg.paths.output_dir = td
        apply(default_hydra_cfg)
