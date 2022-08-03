"""
Takes vegetation probabilities as input, and defines vegetation

"""
import logging
import os
from typing import List
import sys
import pickle

import numpy as np
import numpy.lib.recfunctions as rfn

import pdal

from lidar_prod.tasks.utils import get_pdal_reader, get_pdal_writer


log = logging.getLogger(__name__)

class IoU:
    """contains an IoU and its relative values """
    true_positive: int
    false_negative: int
    false_positive: int
    iou: float

    def __init__(self, true_positive: int, false_negative: int, false_positive: int):
        self.true_positive = true_positive
        self.false_negative = false_negative
        self.false_positive = false_positive
        self.iou = true_positive / (true_positive + false_negative + false_positive)
    
    def __str__(self):
        return "IoU: {:0.3f} |  true positive: {:,} | false negative: {:,} | false positive: {:,}"\
            .format(self.iou, self.true_positive, self.false_negative, self.false_positive)
    
    @staticmethod
    def combine_iou(iou_list: list):
        """combine several IoUs to make an average/total IoU"""
        return IoU( sum(iou.true_positive for iou in iou_list),
                    sum(iou.false_negative for iou in iou_list),
                    sum(iou.false_positive for iou in iou_list)
                )

class BasicIdentifier:
    def __init__(self, threshold: float, proba_column: str, result_column: str, result_code: int, data_format, evaluate_iou: bool=False, truth_column: str=None):
        self.threshold = threshold
        self.proba_column = proba_column
        self.result_column = result_column
        self.result_code = result_code
        self.data_format = data_format
        self.evaluate_iou = evaluate_iou
        self.truth_column = truth_column

    # def identify(self, src_las_path: str, target_las_path: str):
    def identify(self, points: np.ndarray):
        # read the LAS, get its points list and add a dimension, if needed 
        # pipeline = pdal.Pipeline() | get_pdal_reader(src_las_path)
        # pipeline.execute()
        # points = pipeline.arrays[0]

        # add the result column if not yet in points
        if self.result_column not in points.dtype.names:    
            points = rfn.append_fields(points, self.result_column, np.empty(points.shape[0], dtype='uint')) # adding the result column

        # get the mask listing the points above the threshold
        threshold_mask = points[self.proba_column] >= self.threshold   
        points[self.result_column][threshold_mask] = self.result_code
     
        # save points list to the target
        # pipeline = get_pdal_writer(target_las_path).pipeline(points)
        # os.makedirs(os.path.dirname(target_las_path), exist_ok=True)
        # pipeline.execute()

        # calculate ious if necessary
        if self.evaluate_iou:
            self.iou = self.calculate_iou(points[self.truth_column], self.result_code, threshold_mask)

        return points

    def calculate_iou(self, truth_array, value_truth_should_have, evaluated_mask):
        true_positive = np.count_nonzero(
            np.logical_and(truth_array == value_truth_should_have, evaluated_mask)
            )
        false_negative = np.count_nonzero(
            np.logical_and(truth_array == value_truth_should_have, ~evaluated_mask)
            )
        false_positive = np.count_nonzero(
            np.logical_and(truth_array != value_truth_should_have, evaluated_mask)
            )
        return IoU(true_positive, false_negative, false_positive)


class VegetationIdentifier:

    def __init__(self, vegetation_threshold: float,  unclassified_threshold: float, evaluate_ious: bool, data_format):
        self.vegetation_threshold = vegetation_threshold
        self.unclassified_threshold = unclassified_threshold
        self.evaluate_ious = evaluate_ious
        self.data_format = data_format

    def run(self, src_las_path: str, target_las_path: str):
        """Application.

        Args:
            src_las_path (str): path to input LAS file, with vegetation probabilities
            target_las_path (str): path for saving updated LAS file.

        Returns:
            str: returns `target_las_path` for potential terminal piping.

        """
        # give alias to make things more readable
        las_dim = self.data_format.las_dimensions 
        codes = self.data_format.codes

        # read the LAS, get its points list and add a dimension, if needed 
        pipeline = pdal.Pipeline() | get_pdal_reader(src_las_path)
        pipeline.execute()

        # points = rfn.append_fields(points, 'USNG', np.empty(points.shape[0], dtype='uint'))

        # test = np.zeros((points.shape[0] ), dtype=points.dtype)
        # test = np.core.records.fromarrays(test, names = 'toto')
        # print("shape of test: ", test.shape)
        # np.hstack((points, test))

        # weight_pipeline = pickle.dumps(pipeline.arrays[0])
        # print("size entry ", sys.getsizeof(weight_pipeline))

        try:
            points = pipeline.arrays[0]
            points[las_dim.ai_vegetation_unclassified_groups] = 0  # if that dimension doesn't exist, will raise a ValueError
        except ValueError:
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{las_dim.ai_vegetation_unclassified_groups}") # add the new dimension
            pipeline.execute()
            points = pipeline.arrays[0]

        # set the vegetation
        vegetation_mask = points[las_dim.ai_vegetation_proba] >= self.vegetation_threshold   
        points[las_dim.ai_vegetation_unclassified_groups][vegetation_mask] = codes.vegetation

        # set the unclassified
        unclassified_mask = points[las_dim.ai_unclassified_proba] >= self.unclassified_threshold
        points[las_dim.ai_vegetation_unclassified_groups][unclassified_mask] = codes.unclassified

        # save points list to the target
        pipeline_test = get_pdal_writer(target_las_path).pipeline(points)
        os.makedirs(os.path.dirname(target_las_path), exist_ok=True)
        pipeline_test.execute()

        # calculate ious if necessary
        if self.evaluate_ious:
            self.vegetation_iou = self.calculate_iou(points[las_dim.classification], codes.vegetation, vegetation_mask)
            self.unclassified_iou = self.calculate_iou(points[las_dim.classification], codes.unclassified, unclassified_mask)
            self.total_iou = IoU.combine_iou([self.vegetation_iou, self.unclassified_iou])

    def calculate_iou(self, truth_array, value_truth_should_have, evaluated_mask):
        true_positive = np.count_nonzero(
            np.logical_and(truth_array == value_truth_should_have, evaluated_mask)
            )
        false_negative = np.count_nonzero(
            np.logical_and(truth_array == value_truth_should_have, ~evaluated_mask)
            )
        false_positive = np.count_nonzero(
            np.logical_and(truth_array != value_truth_should_have, evaluated_mask)
            )
        return IoU(true_positive, false_negative, false_positive)
        
