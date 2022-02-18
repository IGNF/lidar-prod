import logging
import os
import os.path as osp
import hydra
from omegaconf import DictConfig
from typing import Optional

from lidar_prod.utils import utils
from lidar_prod.tasks.building_validation import BuildingValidator


log = logging.getLogger(__name__)


@utils.eval_time
def apply(config: DictConfig):
    """
    Augment rule-based classification of a point cloud with deep learning
    probabilities and vector building database.

    Args:
        config (DictConfig): Hydra config passed from run.py

    """
    assert os.path.exists(config.paths.src_las)
    out_f = osp.join(config.paths.output_dir, osp.basename(config.paths.src_las))
    bv: BuildingValidator = hydra.utils.instantiate(
        config.building_validation.application
    )
    bv.run(config.paths.src_las, out_f)
