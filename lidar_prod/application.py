import logging
import os
import os.path as osp
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from lidar_prod.utils import utils

from lidar_prod.tasks.building_validation import BuildingValidator


log = logging.getLogger(__name__)


@utils.eval_time
def apply(config: DictConfig) -> Optional[float]:
    assert os.path.exists(config.paths.src_las)
    out_f = osp.join(config.paths.output_dir, osp.basename(config.paths.src_las))
    bv: BuildingValidator = hydra.utils.instantiate(
        config.building_validation.application
    )
    bv.run(config.paths.src_las, out_f)


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    utils.extras(config)

    if config.get("print_config"):
        utils.print_config(config, resolve=False)

    return apply(config)


if __name__ == "__main__":
    # cf. https://github.com/facebookresearch/hydra/issues/1283
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
