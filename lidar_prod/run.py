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
    APPLY_BUILDING = "apply_on_building"
    OPT_BUIlDING = "optimize_building"
    GET_SHAPEFILE = "get_shapefile"


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):  # pragma: no cover
    """Main entry point to either apply or optimize thresholds.

    Check the configurations files for usage.

    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lidar_prod.commons.commons import extras
    from lidar_prod.application import apply, identify_vegetation_unclassified, just_clean, apply_building_module, get_shapefile
    from lidar_prod.optimization import optimize_building, optimize_vegetation, optimize_unclassified

    log = logging.getLogger(__name__)

    extras(config)

    if config.get("task") == POSSIBLE_TASK.OPT_VEGETATION.value:
        optimize_vegetation(config)

    elif config.get("task") == POSSIBLE_TASK.OPT_UNCLASSIFIED.value:
        optimize_unclassified(config)

    elif config.get("task") == POSSIBLE_TASK.ID_VEGETATION_UNCLASSIFIED.value:
        apply(config, identify_vegetation_unclassified)

    elif config.get("task") == POSSIBLE_TASK.CLEANING.value:
        apply(config, just_clean)

    elif config.get("task") == POSSIBLE_TASK.APPLY_BUILDING.value:
        apply(config, apply_building_module)

    elif config.get("task") == POSSIBLE_TASK.OPT_BUIlDING.value:
        optimize_building(config)

    elif config.get("task") == POSSIBLE_TASK.GET_SHAPEFILE.value:
        apply(config, get_shapefile)

    else:
        log.info("WARNING! Starting applying the default process, is this really what you want?")
        apply(config, apply_building_module)


if __name__ == "__main__":  # pragma: no cover
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    # OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
