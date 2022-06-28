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
    src_las_path = config.paths.src_las
    target_las_path = osp.join(config.paths.output_dir, osp.basename(src_las_path))

    with TemporaryDirectory() as td:
        # Temporary LAS file for intermediary results.
        tmp_las_path = osp.join(td, osp.basename(src_las_path))

        # Removes unnecessary input dimensions to reduce memory usage
        cl: Cleaner = hydra.utils.instantiate(config.data_format.cleaning.input)
        cl.run(src_las_path, tmp_las_path)

        # TODO: add a bridge proba dim to test LAS subset before uncommenting
        # Identify bridge from a specialized model predictions
        # bi: BridgeIdentifier = hydra.utils.instantiate(config.bridge_identification)
        # bi.run(tmp_las_path, tmp_las_path)

        # Validate buildings (unsure/confirmed/refuted) on a per-group basis.
        bv: BuildingValidator = hydra.utils.instantiate(
            config.building_validation.application
        )
        bv.run(tmp_las_path, tmp_las_path)

        # Complete buildings with non-candidates that were nevertheless confirmed
        bc: BuildingCompletor = hydra.utils.instantiate(config.building_completion)
        bc.run(tmp_las_path, tmp_las_path)

        # Define groups of confirmed building points among non-candidates
        bi: BuildingIdentifier = hydra.utils.instantiate(config.building_identification)
        bi.run(tmp_las_path, tmp_las_path)

        # Remove unnecessary intermediary dimensions
        cl: Cleaner = hydra.utils.instantiate(config.data_format.cleaning.output)
        cl.run(tmp_las_path, target_las_path)
    return target_las_path
