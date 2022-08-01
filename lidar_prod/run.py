import sys
import os.path as osp
import logging
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    """Main entry point to either apply or optimize thresholds.

    Check the configurations files for usage.

    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lidar_prod.commons.commons import extras
    from lidar_prod.application import apply, apply_veg, apply_cleaning
    from lidar_prod.optimization import optimize
    from lidar_prod.tasks.vegetation_identification import BasicIdentifier
    from lidar_prod.tasks.vegetation_identification_optimization import BasicIdentifierOptimizer

    log = logging.getLogger(__name__)

    extras(config)

    if config.get("task") == "optimize_veg_id":
        log.info("Starting optimizing vegetation identifier")
        data_format = config["data_format"]
        vegetation_identification_optimiser = BasicIdentifierOptimizer(
            config,  
            data_format.las_dimensions.ai_vegetation_proba,
            data_format.las_dimensions.ai_vegetation_unclassified_groups,
            data_format.codes.vegetation,
            data_format.las_dimensions.classification
            )
        vegetation_identification_optimiser.optimize()
        # return optimize(config)
    if config.get("task") == "identify_vegetation":
        log.info("Starting identifying vegetation")
        apply_veg(config)

    elif config.get("task") == "cleaning":
        log.info("Starting cleaning")
        apply_cleaning(config)

    else:
        log.info("Starting applying the default process")
        apply(config)


if __name__ == "__main__":
    sys.path.append(osp.dirname(osp.dirname(__file__)))
    # OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
