import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    # from lidar_prod.optimize import optimize
    from lidar_prod.utils import utils
    from lidar_prod.application import apply
    from lidar_prod.optimization import optimize

    utils.extras(config)

    if config.get("task") == "optimize":
        """Optimization of decision threshold applied to predictions of the NN."""
        return optimize(config)
    else:
        """Automate semantic segmentation decisions"""
        return apply(config)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
