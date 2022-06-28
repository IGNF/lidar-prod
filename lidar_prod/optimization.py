import logging
import hydra
from omegaconf import DictConfig
from lidar_prod.tasks.building_validation_optimization import (
    BuildingValidationOptimizer,
)

from lidar_prod.commons import commons

log = logging.getLogger(__name__)


@commons.eval_time
def optimize(config: DictConfig):
    """
    Runs a multi-objectives hyperparameters optimization of the decision
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
    # TODO: optimization logic should be splitted by task and put into the main() of each script.
