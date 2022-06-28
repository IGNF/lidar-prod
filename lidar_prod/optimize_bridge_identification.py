"""
Takes bridge probabilities as input, and defines bridge.

"""

import logging
import sys
import os.path as osp
import hydra
from omegaconf import DictConfig
from lidar_prod.commons import commons
from lidar_prod.tasks.bridge_identification_optimization import (
    BridgeIdentificationOptimizer,
)


log = logging.getLogger(__name__)


@commons.eval_time
@hydra.main(config_path=".../configs/", config_name="config.yaml")
def optimize_bridge_identification(config: DictConfig):
    """Runs hyperparameters optimization of decision thresholds to maximize bridge vector IoU."""

    commons.extras(config)

    brio: BridgeIdentificationOptimizer = hydra.utils.instantiate(
        config.bridge_identification.optimization
    )
    brio.optimize()


if __name__ == "__main__":
    sys.path.append(osp.dirname(osp.dirname(__file__)))
    optimize_bridge_identification()
