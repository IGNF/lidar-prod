import logging
import os
from typing import Callable
from enum import Enum
from tempfile import TemporaryDirectory
import hydra
from omegaconf import DictConfig
from lidar_prod.tasks.building_completion import BuildingCompletor
from lidar_prod.tasks.cleaning import Cleaner

from lidar_prod.commons import commons
from lidar_prod.tasks.building_validation import BuildingValidator
from lidar_prod.tasks.building_identification import BuildingIdentifier
from lidar_prod.tasks.vegetation_identification import BasicIdentifier

import pdal
from lidar_prod.tasks.utils import get_points_from_las, set_points_to_las, get_pdal_reader, get_a_las_to_las_pdal_pipeline, get_pdal_writer

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
    for src_las_path in get_list_las_path_from_src(config.paths.src_las):
        target_las_path = os.path.join(config.paths.output_dir, os.path.basename(src_las_path))
        stepper = {
            Step.cleaner_beginning: hydra.utils.instantiate(config.data_format.cleaning.input),
            Step.vegetation_detection: hydra.utils.instantiate(config.vegetation_identification),
            Step.cleaner_ending: hydra.utils.instantiate(config.data_format.cleaning.output)
        }
        process_one_file(stepper, src_las_path, target_las_path)


@commons.eval_time
def applying(config: DictConfig, logic: Callable):
    for src_las_path in get_list_las_path_from_src(config.paths.src_las):
        target_las_path = os.path.join(config.paths.output_dir, os.path.basename(src_las_path))
        logic(config, src_las_path, target_las_path)


@commons.eval_time
def apply_veg(config: DictConfig):
    """
    Augment rule-based classification of a point cloud with deep learning
    probabilities and vector building database.

    Args:
        config (DictConfig): Hydra config passed from run.py

    """
    for src_las_path in get_list_las_path_from_src(config.paths.src_las):
        target_las_path = os.path.join(config.paths.output_dir, os.path.basename(src_las_path))
        detect_vegetation_unclassified(config, src_las_path, target_las_path)


@commons.eval_time
def apply_cleaning(config: DictConfig):
    for src_las_path in get_list_las_path_from_src(config.paths.src_las):
        target_las_path = os.path.join(config.paths.output_dir, os.path.basename(src_las_path))
        just_clean(config, src_las_path, target_las_path)


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
                src_las_path.append(os.path.join(root, file))
        return src_las_path


@commons.eval_time
def detect_vegetation_unclassified(config, src_las_path: str, dest_las_path: str = None):

    # pipeline = pdal.Pipeline() | get_pdal_reader(src_las_path)
    # pipeline.execute()

    log.info(f"Detecting on {src_las_path}")
    data_format = config["data_format"]

    # pipeline = get_a_las_to_las_pdal_pipeline(src_las_path, dest_las_path, [])
    # pipeline = pdal.Pipeline()
    # pipeline |= get_pdal_reader(src_las_path)
    # pipeline.execute()
    # points = pipeline.arrays[0]
    

    # pipeline = get_pdal_writer(dest_las_path).pipeline(pipeline.arrays[0])

    # pipeline |= get_pdal_writer(dest_las_path)
    # pipeline.execute()

    

    # formatting extra_dims to be readable by pdal
    # extra_dims = ",".join(data_format.cleaning.input.extra_dims)
    # extra_dims = extra_dims if extra_dims else []

    points = get_points_from_las(src_las_path)

    cleaner = hydra.utils.instantiate(data_format.cleaning.input)
    # points = cleaner.remove_unwanted_dimensions(points)

    # cleaner.add_column(src_las_path, dest_las_path, [data_format.las_dimensions.ai_vegetation_unclassified_groups])
    points = get_points_from_las(src_las_path)


    # detect vegetation
    vegetation_identifier = BasicIdentifier(
        config["vegetation_identification"]["vegetation_threshold"],
        data_format.las_dimensions.ai_vegetation_proba,
        data_format.las_dimensions.ai_vegetation_unclassified_groups,
        data_format.codes.vegetation,
        data_format
    )
    points = vegetation_identifier.identify(points)

    # detect unclassified
    unclassified_identifier = BasicIdentifier(
        config["vegetation_identification"]["unclassified_threshold"],
        data_format.las_dimensions.ai_unclassified_proba,
        data_format.las_dimensions.ai_vegetation_unclassified_groups,
        data_format.codes.unclassified,
        data_format
    )
    points = unclassified_identifier.identify(points)

    # keeping only the wanted dimensions for the result las
    cleaner = hydra.utils.instantiate(data_format.cleaning.output)
    # points = cleaner.remove_unwanted_dimensions(points)

    # save points array to the target
    # set_points_to_las(dest_las_path, points)

    pipeline = get_pdal_writer(dest_las_path).pipeline(points)
    os.makedirs(os.path.dirname(dest_las_path), exist_ok=True)
    pipeline.execute()

    # pipeline = get_a_las_to_las_pdal_pipeline(dest_las_path, dest_las_path, [])
    # pipeline.execute()

@commons.eval_time
def just_clean(config, src_las_path: str, dest_las_path: str = None):
    log.info(f"Cleaning {src_las_path}")
    points = get_points_from_las(src_las_path)

    # remove unwanted dimensions
    cleaner = hydra.utils.instantiate(config.data_format.cleaning.input)
    points = cleaner.remove_unwanted_dimensions(points)

    # save points array to the target
    set_points_to_las(dest_las_path, points)


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
        tmp_las_path = os.path.join(td, os.path.basename(src_las_path))

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
