import shutil
from pathlib import Path

from lidar_prod.tasks.building_completion import BuildingCompletor
from lidar_prod.tasks.utils import get_pdal_writer
from tests.conftest import (
    check_expected_classification,
    check_las_contains_dims,
    check_las_format_versions_and_srs,
)

TMP_DIR = Path("tmp/lidar_prod/tasks/building_completion")


def setup_module(module):
    try:
        shutil.rmtree(TMP_DIR)
    except FileNotFoundError:
        pass
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def test_run(hydra_cfg):
    input_las_path = "tests/files/870000_6618000.subset.postValidation.laz"
    dest_dir = TMP_DIR / "run"
    dest_dir.mkdir(parents=True)
    dest_las_path = str(dest_dir / "output.laz")

    _fc = hydra_cfg.data_format.codes.building.final
    expected_codes = {
        1,
        2,
        _fc.building,
        _fc.not_building,
        _fc.unsure,
    }

    bc_cfg = hydra_cfg.building_completion
    bc = BuildingCompletor(
        min_building_proba=bc_cfg.min_building_proba,
        cluster=bc_cfg.cluster,
        data_format=bc_cfg.data_format,
    )
    las_metadata = bc.run(input_las_path)
    pipeline = bc.pipeline
    pipeline |= get_pdal_writer(dest_las_path, las_metadata)
    pipeline.execute()

    check_las_format_versions_and_srs(dest_las_path, hydra_cfg.data_format.epsg)
    check_expected_classification(dest_las_path, expected_codes)
    dims = hydra_cfg.data_format.las_dimensions
    check_las_contains_dims(
        dest_las_path,
        None,
        dims_to_check=[
            dims.classification,
            dims.ClusterID_confirmed_or_high_proba,
            dims.completion_non_candidate_flag,
            dims.candidate_buildings_flag,
        ],
    )
