import logging
import os
import os.path as osp
from tempfile import TemporaryDirectory
import hydra
from omegaconf import DictConfig
from lidar_prod.tasks.building_completion import BuildingCompletor
from lidar_prod.tasks.cleaning import Cleaner

from lidar_prod.commons import commons
from lidar_prod.tasks.building_validation import BuildingValidator
from lidar_prod.tasks.building_identification import BuildingIdentifier


log = logging.getLogger(__name__)


@commons.eval_time
def apply(config: DictConfig):
    """
    Augment rule-based classification of a point cloud with deep learning
    probabilities and vector building database.

    Args:
        config (DictConfig): Hydra config passed from run.py

    """
    assert os.path.exists(config.paths.src_las)
    in_f = config.paths.src_las
    out_f = osp.join(config.paths.output_dir, osp.basename(in_f))

    with TemporaryDirectory() as td:
        # Temporary LAS file for intermediary results.
        temp_f = osp.join(td, osp.basename(in_f))

        # Removes unnecessary input dimensions to reduce memory usage
        cl: Cleaner = hydra.utils.instantiate(config.data_format.cleaning.input)
        cl.run(in_f, temp_f)

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
        cl.run(temp_f, out_f)
    return out_f
