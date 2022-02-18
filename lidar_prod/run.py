import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os.path as osp


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lidar_prod.utils.utils import extras
    from lidar_prod.application import apply
    from lidar_prod.optimization import optimize

    extras(config)
    print(config.paths.output_dir)
    if config.get("task") == "optimize":
        return optimize(config)
    else:
        return apply(config)


if __name__ == "__main__":
    sys.path.append(osp.dirname(osp.dirname(__file__)))
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
