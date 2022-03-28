import logging
import os
import os.path as osp
from tempfile import TemporaryDirectory
import hydra
from omegaconf import DictConfig
from typing import Optional
from lidar_prod.tasks.building_completion import BuildingCompletor
from lidar_prod.tasks.cleaning import Cleaner

from lidar_prod.utils import utils
from lidar_prod.tasks.building_validation import BuildingValidator
from lidar_prod.tasks.building_identification import BuildingIdentifier


log = logging.getLogger(__name__)

# TODO: intermediary out_f should be in a tempr dir instead to avoid unfinised business


@utils.eval_time
def apply(config: DictConfig):
    """
    Augment rule-based classification of a point cloud with deep learning
    probabilities and vector building database.

    Args:
        config (DictConfig): Hydra config passed from run.py

    """
    assert os.path.exists(config.paths.src_las)
    IN_F = config.paths.src_las
    OUF_F = osp.join(config.paths.output_dir, osp.basename(IN_F))

    with TemporaryDirectory() as td:
        # Temporary LAS file for intermediary results.
        temp_f = osp.join(td, osp.basename(IN_F))

        # Removes unnecessary input dimensions to reduce memory usage
        cl: Cleaner = hydra.utils.instantiate(config.data_format.cleaning.input)
        cl.run(IN_F, temp_f)

        # Validate buildings (unsure/confirmed/refuted) on a per-group basis.
        bv: BuildingValidator = hydra.utils.instantiate(
            config.building_validation.application
        )
        bv.run(temp_f, temp_f)

        # Complete buildings with non-candidates that were nevertheless confirmed
        bc: BuildingCompletor = hydra.utils.instantiate(config.building_completion)
        bc.run(temp_f, temp_f)

        # Define groups of confirmed building points among non-candidates
        bi: BuildingIdentifier = hydra.utils.instantiate(config.building_identification)
        bi.run(temp_f, temp_f)

        # Remove unnecessary intermediary dimensions
        cl: Cleaner = hydra.utils.instantiate(config.data_format.cleaning.output)
        cl.run(temp_f, OUF_F)
