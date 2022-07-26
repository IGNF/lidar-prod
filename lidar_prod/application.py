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
from lidar_prod.tasks.vegetation_identification import VegetationIdentifier


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

    # src_las is a unique file
    if os.path.isfile(config.paths.src_las):
        src_las_path = config.paths.src_las
        whole_process(config, src_las_path)

    # src_las is a directory, we process all LAS files in it
    if os.path.isdir(config.paths.src_las):
        for (root, _, files) in os.walk(config.paths.src_las):
            for file in files:
                _, file_extension = os.path.splitext(file)
                if file_extension.lower() != ".las":    # only LAS files are processed (the extension might be in uppercase)
                    continue
                
                src_las_path = os.path.join(root,file)
                whole_process(config, src_las_path)

@commons.eval_time    
def whole_process(config: DictConfig, src_las_path: str):
    log.info(f"Processing {src_las_path}")
    target_las_path = osp.join(config.paths.output_dir, osp.basename(src_las_path))

    with TemporaryDirectory() as td:
        # Temporary LAS file for intermediary results.
        tmp_las_path = osp.join(td, osp.basename(src_las_path))

        # Removes unnecessary input dimensions to reduce memory usage
        cl: Cleaner = hydra.utils.instantiate(config.data_format.cleaning.input)
        cl.run(src_las_path, tmp_las_path)

        # # Validate buildings (unsure/confirmed/refuted) on a per-group basis.
        # bv: BuildingValidator = hydra.utils.instantiate(
        #     config.building_validation.application
        # )
        # bv.run(tmp_las_path, tmp_las_path)

        # # Complete buildings with non-candidates that were nevertheless confirmed
        # bc: BuildingCompletor = hydra.utils.instantiate(config.building_completion)
        # bc.run(tmp_las_path, tmp_las_path)

        # # Define groups of confirmed building points among non-candidates
        # bi: BuildingIdentifier = hydra.utils.instantiate(config.building_identification)
        # bi.run(tmp_las_path, tmp_las_path)

        # identify vegetation
        vegetation_identifier: VegetationIdentifier = hydra.utils.instantiate(
            config.vegetation_identification
        )
        vegetation_identifier.run(tmp_las_path, tmp_las_path)


        # Remove unnecessary intermediary dimensions
        cl: Cleaner = hydra.utils.instantiate(config.data_format.cleaning.output)
        cl.run(tmp_las_path, target_las_path)
