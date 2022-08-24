import sys
import os
import logging
import hydra
from omegaconf import DictConfig
from enum import Enum


class POSSIBLE_TASK(Enum):
    CLEANING = "cleaning"
    ID_VEGETATION_UNCLASSIFIED = "identify_vegetation_unclassified"
    OPT_VEGETATION = "optimize_veg_id"
    OPT_UNCLASSIFIED = "optimize_unc_id"
    OPT_BUIlDING = "optimize_building"


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):  # pragma: no cover
    """Main entry point to either apply or optimize thresholds.

    Check the configurations files for usage.

    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lidar_prod.commons.commons import extras
    from lidar_prod.application import apply, identify_vegetation_unclassified, just_clean
    from lidar_prod.optimization import optimize_building, optimize_vegetation, optimize_unclassified

    log = logging.getLogger(__name__)

    extras(config)

    assert os.path.exists(config.paths.src_las)

    if config["task"] == POSSIBLE_TASK.OPT_VEGETATION.value:
        optimize_vegetation(config)

    elif config["task"] == POSSIBLE_TASK.OPT_UNCLASSIFIED.value:
        optimize_unclassified(config)

    elif config["task"] == POSSIBLE_TASK.ID_VEGETATION_UNCLASSIFIED.value:
        apply(config, identify_vegetation_unclassified)

    elif config["task"] == POSSIBLE_TASK.CLEANING.value:
        apply(config, just_clean)

    elif config["task"] == POSSIBLE_TASK.OPT_BUIlDING.value:
        optimize_building(config)

    else:
        log.info("Starting applying the default process")
        optimize_building(config)


if __name__ == "__main__":  # pragma: no cover
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    # OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
