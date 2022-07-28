import logging
import os
import os.path as osp
from enum import Enum
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

class Step(Enum):
    """set a list of values to designate each step for the process"""
    cleaner_beginning = 1
    building_validator = 2
    building_completor = 3
    building_identifier = 4
    vegetation_detection = 5
    cleaner_ending = 6

@commons.eval_time
def apply(config: DictConfig):
    """
    Augment rule-based classification of a point cloud with deep learning
    probabilities and vector building database.

    Args:
        config (DictConfig): Hydra config passed from run.py

    """
    assert os.path.exists(config.paths.src_las)

    for src_las_path in get_list_las_path_from_src(config.paths.src_las):
        # stepper = {cleaner_beginning: hydra.utils.instantiate(config.data_format.cleaning.input)}
        target_las_path = osp.join(config.paths.output_dir, osp.basename(src_las_path))
        stepper = {
            Step.cleaner_beginning : hydra.utils.instantiate(config.data_format.cleaning.input),
            Step.vegetation_detection : hydra.utils.instantiate(config.vegetation_identification),
            Step.cleaner_ending: hydra.utils.instantiate(config.data_format.cleaning.output)
        }
        process_one_file(stepper, src_las_path, target_las_path)

def get_list_las_path_from_src(src_path: str):
    """get a list of las from a path. 
    If the path is a single file, that file will be the only one in the returned list
    if the path is a directory, all the .las will be in the returned list"""
    # src_path is a unique file
    if os.path.isfile(src_path):
        return [src_path]

    # src_path is a directory
    if os.path.isdir(src_path):
        src_las_path = []
        for (root, _, files) in os.walk(src_path):
            for file in files:
                _, file_extension = os.path.splitext(file)
                if file_extension.lower() != ".las":    # only LAS files are selected (the extension might be in uppercase)
                    continue
                src_las_path.append(os.path.join(root,file))
        return src_las_path

@commons.eval_time    
def process_one_file(stepper: dict, src_las_path: str, dest_las_path: str = None):
    """call every desired step to process a las
    Args:
        stepper: a dictionary containing the instantiated objects of the needed classes
        for each step
        src_las_path: the path of the source las
        dest_las_path: the path to save the result (optional)
    """
    log.info(f"Processing {src_las_path}")
    with TemporaryDirectory() as td:
        # Temporary LAS file for intermediary results.
        tmp_las_path = osp.join(td, osp.basename(src_las_path))

        # Removes unnecessary input dimensions to reduce memory usage
        try:
            cl = stepper[Step.cleaner_beginning]
            cl.run(src_las_path, tmp_las_path)
        except KeyError:
            # if that key isn't in stepper we assume that step isn't wanted for this process
            pass

        # # Validate buildings (unsure/confirmed/refuted) on a per-group basis.
        try:
            bv = stepper[Step.building_validator]
            bv.run(tmp_las_path, tmp_las_path)
        except KeyError:
            # if that key isn't in stepper we assume that step isn't wanted for this process
            pass

        # # Complete buildings with non-candidates that were nevertheless confirmed
        try:
            bc = stepper[Step.building_completor]
            bc.run(tmp_las_path, tmp_las_path)
        except KeyError:
            # if that key isn't in stepper we assume that step isn't wanted for this process
            pass

        # # Define groups of confirmed building points among non-candidates
        try:
            bc = stepper[Step.building_identifier]
            bc.run(tmp_las_path, tmp_las_path)
        except KeyError:
            # if that key isn't in stepper we assume that step isn't wanted for this process
            pass

        # identify vegetation
        try:
            vegetation_identifier = stepper[Step.vegetation_detection]
            vegetation_identifier.identify(tmp_las_path, tmp_las_path)
        except KeyError:
            # if that key isn't in stepper we assume that step isn't wanted for this process
            pass

        # Remove unnecessary intermediary dimensions
        try:
            cl = stepper[Step.cleaner_ending]
            cl.run(tmp_las_path, dest_las_path)
        except KeyError:
            # if that key isn't in stepper we assume that step isn't wanted for this process
            pass
