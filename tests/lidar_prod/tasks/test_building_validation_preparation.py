import os
import tempfile

import hydra

from lidar_prod.tasks.building_validation import BuildingValidator

INVALID_OVERLAY_LAZ_PATH = "tests/files/invalid_overlay_data/842_6521-invalid_shapefile_area-borders.las"


# We try to reduce size of LAZ to isolate the problem first to make it quick to test when it is ok.

# Normal execution on subset of LAZ lasts ~ 3sec.
# If a regression occurs, the pdal execution will hang and a timeout would make it more apparent.
# However, pytest-timeout does not stop pdal for some reasons. For now this should be sufficient.
def test_shapefile_overlay_in_building_module(hydra_cfg):
    """We test the application against a LAS subset for which the BDUni shapefile shows overlapping vectors.

    We only need points at the borders of the area in order to request the error-generating shapefile.

    These overlaps caused a hanging overlay fiter, and we added a dissolve operation on the requested
    shapefile to remove this bug.

    """
    # Run application on the data subset for which vector data is expected to be invalid.
    with tempfile.TemporaryDirectory() as hydra_cfg.paths.output_dir:
        target_las_path = os.path.join(
            hydra_cfg.paths.output_dir,
            os.path.basename(INVALID_OVERLAY_LAZ_PATH),
        )
        bv: BuildingValidator = hydra.utils.instantiate(hydra_cfg.building_validation.application)
        bv.prepare(INVALID_OVERLAY_LAZ_PATH, target_las_path)
