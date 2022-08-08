import sys
import os 
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
    from lidar_prod.application import apply, applying, detect_vegetation_unclassified, just_clean
    from lidar_prod.optimization import optimize
    from lidar_prod.tasks.vegetation_identification_optimization import BasicIdentifierOptimizer

    log = logging.getLogger(__name__)

    extras(config)

    assert os.path.exists(config.paths.src_las)

    if config.get("task") == "optimize_veg_id":
        log.info("Starting optimizing vegetation identifier")
        data_format = config["data_format"]
        vegetation_identification_optimiser = BasicIdentifierOptimizer(
            config,  
            data_format.las_dimensions.ai_vegetation_proba,
            data_format.las_dimensions.ai_vegetation_unclassified_groups,
            data_format.codes.vegetation,
            data_format.las_dimensions.classification,
            25
            )
        vegetation_identification_optimiser.optimize()

    elif config.get("task") == "optimize_unc_id":
        log.info("Starting optimizing unclassifier identifier")
        data_format = config["data_format"]
        vegetation_identification_optimiser = BasicIdentifierOptimizer(
            config,  
            data_format.las_dimensions.ai_unclassified_proba,
            data_format.las_dimensions.ai_vegetation_unclassified_groups,
            data_format.codes.unclassified,
            data_format.las_dimensions.classification,
            50
            )
        vegetation_identification_optimiser.optimize()
    
    elif config.get("task") == "identify_vegetation":
        logic = detect_vegetation_unclassified
        applying(config, logic)

    elif config.get("task") == "cleaning":
        logic = just_clean
        applying(config, logic)

    else:
        log.info("Starting applying the default process")
        apply(config)

    
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    # OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
