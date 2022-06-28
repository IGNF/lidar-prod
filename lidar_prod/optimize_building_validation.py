import logging
import os.path as osp
import sys
import hydra
from omegaconf import DictConfig
from lidar_prod.tasks.building_validation_optimization import (
    BuildingValidationOptimizer,
)

from lidar_prod.commons import commons

log = logging.getLogger(__name__)


@commons.eval_time
@hydra.main(config_path="../configs/", config_name="config.yaml")
def optimize_building_validation(config: DictConfig):
    """Runs a multi-objectives hyperparameters optimization of the decision
    thresholds, to maximize recall and precision directly while
    also maximizing automation.

    Args:
        config (DictConfig): Hydra config passed from run.py

    """
    commons.extras(config)

    bvo: BuildingValidationOptimizer = hydra.utils.instantiate(
        config.building_validation.optimization
    )
    bvo.run()


if __name__ == "__main__":
    sys.path.append(osp.dirname(osp.dirname(__file__)))
    optimize_building_validation()
