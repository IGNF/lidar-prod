import logging
import hydra
from omegaconf import DictConfig
from typing import Optional
from lidar_prod.tasks.building_validation_optimization import (
    BuildingValidationOptimizer,
)

from lidar_prod.utils import utils

log = logging.getLogger(__name__)


@utils.eval_time
def optimize(config: DictConfig) -> Optional[float]:
    utils.extras(config)

    bv: BuildingValidationOptimizer = hydra.utils.instantiate(
        config.building_validation.optimization
    )
    bv.run()
