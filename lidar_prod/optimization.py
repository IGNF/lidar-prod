import logging

import hydra
from omegaconf import DictConfig

from lidar_prod.commons import commons
from lidar_prod.tasks.basic_identification_optimization import BasicIdentifierOptimizer
from lidar_prod.tasks.building_validation_optimization import BuildingValidationOptimizer
from lidar_prod.tasks.utils import BDUniConnectionParams

log = logging.getLogger(__name__)


@commons.eval_time
def optimize_building(
    config: DictConfig,
):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
    """
    Run a multi-objectives hyperparameters optimization of the decision
    thresholds, to maximize recall and precision directly while
    also maximizing automation.

    Args:
        config (DictConfig): Hydra config passed from run.py

    """
    commons.extras(config)

    bvo: BuildingValidationOptimizer = hydra.utils.instantiate(
        config.building_validation.optimization
    )
    bd_uni_connection_params: BDUniConnectionParams = hydra.utils.instantiate(
        config.bd_uni_connection_params
    )
    bvo.bv.bd_uni_connection_params = bd_uni_connection_params
    bvo.run()


def optimize_vegetation(
    config: DictConfig,
):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
    log.info("Starting optimizing vegetation identifier")
    data_format = config["data_format"]
    vegetation_identification_optimiser = BasicIdentifierOptimizer(
        config,
        data_format["las_dimensions"]["ai_vegetation_proba"],
        data_format["las_dimensions"]["ai_vegetation_unclassified_groups"],
        data_format["codes"]["vegetation"],
        data_format["las_dimensions"]["classification"],
        config["basic_identification"]["vegetation_nb_trials"],
        list(data_format["codes"]["vegetation_target"].values()),
    )
    vegetation_identification_optimiser.optimize()


def optimize_unclassified(
    config: DictConfig,
):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
    log.info("Starting optimizing unclassifier identifier")
    data_format = config["data_format"]
    unclassified_identification_optimiser = BasicIdentifierOptimizer(
        config,
        data_format["las_dimensions"]["ai_unclassified_proba"],
        data_format["las_dimensions"]["ai_vegetation_unclassified_groups"],
        data_format["codes"]["unclassified"],
        data_format["las_dimensions"]["classification"],
        config["basic_identification"]["unclassified_nb_trials"],
    )
    unclassified_identification_optimiser.optimize()
