import os
import shutil
from pathlib import Path

import hydra

from lidar_prod.tasks.building_validation import thresholds
from lidar_prod.tasks.building_validation_optimization import (
    BuildingValidationOptimizer,
)
from lidar_prod.tasks.utils import BDUniConnectionParams

TMP_DIR = Path("tmp/lidar_prod/tasks/building_validation_optimization")


def setup_module(module):
    try:
        shutil.rmtree(TMP_DIR)
    except FileNotFoundError:
        pass
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def test_BuildingValidationOptimizer_run(hydra_cfg):
    config = hydra_cfg.copy()
    opt_cfg = config.building_validation.optimization
    opt_cfg.paths.input_las_dir = "tests/files/building_optimization_data/preds"
    opt_cfg.paths.results_output_dir = str(TMP_DIR / "run")

    bvo: BuildingValidationOptimizer = hydra.utils.instantiate(
        config.building_validation.optimization
    )

    bd_uni_connection_params: BDUniConnectionParams = hydra.utils.instantiate(
        hydra_cfg.bd_uni_connection_params
    )
    bvo.bv.bd_uni_connection_params = bd_uni_connection_params
    bvo.run()

    th_yaml = opt_cfg.paths.building_validation_thresholds

    assert os.path.isfile(th_yaml)
    assert isinstance(thresholds.load(th_yaml), thresholds)

    for filename in os.listdir(opt_cfg.paths.input_las_dir):
        assert (TMP_DIR / "run" / "prepared" / filename).is_file()
        assert (TMP_DIR / "run" / "updated" / filename).is_file()
