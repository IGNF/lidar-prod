import shutil
from pathlib import Path

import hydra
import numpy as np
import pytest

from lidar_prod.tasks.building_validation import BuildingValidator
from lidar_prod.tasks.utils import BDUniConnectionParams, get_las_data_from_las

TMP_DIR = Path("tmp/lidar_prod/tasks/building_validation_preparation")


def setup_module(module):
    try:
        shutil.rmtree(TMP_DIR)
    except FileNotFoundError:
        pass
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def test_shapefile_overlay_in_building_module(hydra_cfg):
    """Check that that the prepare function does not add any presence data if the laz geometry does not intersect the
    BDUni territoire corresponding with the configured epsg"""
    # Run application on the data subset for which vector data is expected to be invalid.
    laz_input_file = "tests/files/St_Barth_RGAF09_UTM20N_IGN_1988_SB_subset_100m.laz"
    epsg = 5490

    target_las_path = str(TMP_DIR / "St_Barth_RGAF09_UTM20N_IGN_1988_SB_subset_100m_prepared.laz")
    cfg = hydra_cfg.copy()
    cfg.data_format.epsg = epsg

    bd_uni_connection_params: BDUniConnectionParams = hydra.utils.instantiate(
        cfg.bd_uni_connection_params
    )

    bv = BuildingValidator(
        shp_path=cfg.building_validation.application.shp_path,
        bd_uni_connection_params=bd_uni_connection_params,
        cluster=cfg.building_validation.application.cluster,
        bd_uni_request=cfg.building_validation.application.bd_uni_request,
        data_format=cfg.building_validation.application.data_format,
        thresholds=cfg.building_validation.application.thresholds,
        use_final_classification_codes=cfg.building_validation.application.use_final_classification_codes,
    )

    bv.prepare(laz_input_file, target_las_path, save_result=True)
    data = get_las_data_from_las(target_las_path)
    overlay = data[cfg.data_format.las_dimensions.uni_db_overlay]
    assert np.any(overlay == 1)  # assert some points are marked
    assert np.any(overlay == 0)  # assert not all points are marked


def test_shapefile_overlay_in_building_module_fail(hydra_cfg):
    """Check that that the prepare function fails if the laz geometry does not intersect the
    BDUni territoire corresponding with the configured epsg"""
    # Run application on the data subset for which vector data is expected to be invalid.
    laz_input_file = "tests/files/St_Barth_RGAF09_UTM20N_IGN_1988_SB_subset_100m.laz"
    wrong_epsg = 2154

    target_las_path = str(
        TMP_DIR / "St_Barth_RGAF09_UTM20N_IGN_1988_SB_subset_100m_wrong_epsg.laz"
    )
    cfg = hydra_cfg.copy()
    cfg.data_format.epsg = wrong_epsg

    bd_uni_connection_params: BDUniConnectionParams = hydra.utils.instantiate(
        cfg.bd_uni_connection_params
    )

    bv = BuildingValidator(
        shp_path=cfg.building_validation.application.shp_path,
        bd_uni_connection_params=bd_uni_connection_params,
        cluster=cfg.building_validation.application.cluster,
        bd_uni_request=cfg.building_validation.application.bd_uni_request,
        data_format=cfg.building_validation.application.data_format,
        thresholds=cfg.building_validation.application.thresholds,
        use_final_classification_codes=cfg.building_validation.application.use_final_classification_codes,
    )

    with pytest.raises(ValueError):
        bv.prepare(laz_input_file, target_las_path, save_result=True)


# We try to reduce size of LAZ to isolate the problem first to make it quick to test when it is ok.


# Normal execution on subset of LAZ lasts ~ 3sec.
# If a regression occurs, the pdal execution will hang and a timeout would make it more apparent.
# However, pytest-timeout does not stop pdal for some reasons. For now this should be sufficient.
def test_shapefile_overlay_in_building_module_invalid_overlay(hydra_cfg):
    """We test the application against a LAS subset for which the BDUni shapefile shows overlapping vectors.

    We only need points at the borders of the area in order to request the error-generating shapefile.

    These overlaps caused a hanging overlay fiter, and we added a dissolve operation on the requested
    shapefile to remove this bug.

    """
    invalid_overlay_laz_path = (
        "tests/files/invalid_overlay_data/842_6521-invalid_shapefile_area-borders.las"
    )
    # Run application on the data subset for which vector data is expected to be invalid.
    target_las_path = str(TMP_DIR / "invalid_overlay.laz")

    bd_uni_connection_params: BDUniConnectionParams = hydra.utils.instantiate(
        hydra_cfg.bd_uni_connection_params
    )

    bv = BuildingValidator(
        shp_path=hydra_cfg.building_validation.application.shp_path,
        bd_uni_connection_params=bd_uni_connection_params,
        cluster=hydra_cfg.building_validation.application.cluster,
        bd_uni_request=hydra_cfg.building_validation.application.bd_uni_request,
        data_format=hydra_cfg.building_validation.application.data_format,
        thresholds=hydra_cfg.building_validation.application.thresholds,
        use_final_classification_codes=hydra_cfg.building_validation.application.use_final_classification_codes,
    )

    bv.prepare(invalid_overlay_laz_path, target_las_path)
