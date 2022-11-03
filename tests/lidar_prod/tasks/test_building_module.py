import os
import tempfile

import hydra
import pytest

from lidar_prod.tasks.building_validation import BuildingValidator

INVALID_OVERLAY_LAZ_PATH = "tests/files/large/842_6521_invalid_band.las"
VALID_SHAPE_FILE = "tests/files/invalid_overlay_data/842_6524_invalid_band/842_6521_dissolve.shp"
# INVALID_SHAPE_FILE = "tests/files/invalid_overlay_data/842_6524_invalid_band_shapefile/dalle2_doublons.shp"


@pytest.mark.parametrize("shape_file", [VALID_SHAPE_FILE, None])
# @pytest.mark.parametrize("shape_file", [None])  # remove once it is corrected?
# If a regression occurs, the pdal execution will hang and we need to stop this test.
# Normal execution lasts ~ 2 min on a local machine.
@pytest.mark.timeout(120 + 60)
def test_shapefile_overlay_in_building_module(legacy_hydra_cfg, shape_file):
    """We test the application against a LAS subset for which the BDUni shapefile shows overlapping vectors.

    These overlaps caused a bug in the overlay, and we test that adding a union operation when requesting
    the shapefile removes the error.

    We start with a valid version to ensure that there is no bug with the test itself.
    Then we request the BD Uni.

    """
    # Run application on the data subset for which vector data is expected to be invalid.
    with tempfile.TemporaryDirectory() as legacy_hydra_cfg.paths.output_dir:
        if shape_file:
            legacy_hydra_cfg.building_validation.application.shp_path = shape_file
        target_las_path = os.path.join(
            legacy_hydra_cfg.paths.output_dir,
            os.path.basename(INVALID_OVERLAY_LAZ_PATH),
        )
        bv: BuildingValidator = hydra.utils.instantiate(legacy_hydra_cfg.building_validation.application)
        bv.prepare(INVALID_OVERLAY_LAZ_PATH, target_las_path)
