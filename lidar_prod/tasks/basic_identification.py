"""
Takes vegetation probabilities as input, and defines vegetation

"""
from __future__ import (
    annotations,
)  # to recognize IoU as a type by itself (in __add__())

import logging
from typing import Union

import laspy
import numpy as np

log = logging.getLogger(__name__)


class IoU:
    """Contains an IoU and its associated values."""

    true_positive: int
    false_negative: int
    false_positive: int
    iou: float

    def __init__(self, true_positive: int, false_negative: int, false_positive: int):
        self.true_positive = true_positive
        self.false_negative = false_negative
        self.false_positive = false_positive
        if true_positive + false_negative + false_positive > 0:
            self.iou = true_positive / (true_positive + false_negative + false_positive)
        else:
            self.iou = 1

    def __add__(self, other_iou: IoU):
        return IoU(
            self.true_positive + other_iou.true_positive,
            self.false_negative + other_iou.false_negative,
            self.false_positive + other_iou.false_positive,
        )

    def __str__(self):
        return "IoU: {:0.3f} |  true positive: {:,} | false negative: {:,} | false positive: {:,}".format(
            self.iou, self.true_positive, self.false_negative, self.false_positive
        )

    @staticmethod
    def combine_iou(iou_list: list):
        """Combine several IoUs to return an average/total IoU."""
        return IoU(
            sum(iou.true_positive for iou in iou_list),
            sum(iou.false_negative for iou in iou_list),
            sum(iou.false_positive for iou in iou_list),
        )

    @staticmethod
    def iou_by_mask(preds_mask: np.ndarray, target_mask: np.ndarray):
        """return an IoU from a mask we want to evaluate and a mask containing the truth"""
        true_positive = np.count_nonzero(np.logical_and(target_mask, preds_mask))
        false_negative = np.count_nonzero(np.logical_and(target_mask, ~preds_mask))
        false_positive = np.count_nonzero(np.logical_and(~target_mask, preds_mask))
        return IoU(true_positive, false_negative, false_positive)


class BasicIdentifier:
    def __init__(
        self,
        threshold: float,
        proba_column: str,
        result_column: str,
        result_code: int,
        evaluate_iou: bool = False,
        target_column: str = None,
        target_result_code: Union[int, list] = None,
    ) -> None:
        """
        BasicIdentifier set all points with a value from a column above a threshold to another value in another column

        args:
            threshold: above the threshold, a point is set
            proba_column: the column the treshold is compared against
            result_column: the column to store the result
            result_code: the value the point will be set to
            evaluate_iou: True if we want to evaluate the IoU of that selection
            target_column: if we want to evaluate the IoU, this is the column with the real results to compare againt
            target_result_code: if we want to evaluate the IoU, this is/are the code(s) of the target.
                                Can be an int of a list of int, if we want an IoU but target_result_code
                                is not provided then result_code is used instead.
        """
        self.threshold = threshold
        self.proba_column = proba_column
        self.result_column = result_column
        self.result_code = result_code
        self.evaluate_iou = evaluate_iou
        self.target_column = target_column
        self.target_result_code = (
            target_result_code if target_result_code else result_code
        )

    def identify(self, las_data: laspy.lasdata.LasData) -> None:
        """Identify the points above the threshold and set them to the wanted value."""
        # if the result column doesn't exist, we add it
        if self.result_column not in [
            dim for dim in las_data.point_format.extra_dimension_names
        ]:
            las_data.add_extra_dim(
                laspy.ExtraBytesParams(name=self.result_column, type="uint32")
            )

        # get the mask listing the points above the threshold
        threshold_mask = las_data.points[self.proba_column] >= self.threshold

        # set the selected points to the wanted value
        las_data.points[self.result_column][threshold_mask] = self.result_code

        # calculate ious if necessary
        if self.evaluate_iou:
            if isinstance(self.target_result_code, int):
                target_mask = (
                    las_data.points[self.target_column] == self.target_result_code
                )
            else:  # if not an int, truth_mask should be a list
                target_mask = np.isin(
                    las_data.points[self.target_column], self.target_result_code
                )
            self.iou = IoU.iou_by_mask(threshold_mask, target_mask)

        # MONKEY PATCHING !!! for debugging
        # if self.result_code == 1:   # unclassified
        #     truth_mask = las_data.points["classification"] == 1
        # else:   # vegetation
        #     truth_mask = np.isin(las_data.points["classification"], [3, 4, 5])

        # print("threshold_mask size: ", np.count_nonzero(threshold_mask), "truth_mask size: ", np.count_nonzero(truth_mask))

        # self.result_code = self.result_code if self.result_code ==1 else 11

        # las_data.points[self.result_column][np.logical_and(truth_mask, threshold_mask)] = self.result_code # correct values
        # las_data.points[self.result_column][np.logical_and(truth_mask, ~threshold_mask)] = self.result_code+1 # false positive
        # las_data.points[self.result_column][np.logical_and(~truth_mask, threshold_mask)] = self.result_code+2 # false negative
        # print(
        #     "true positive: ", np.count_nonzero(np.logical_and(truth_mask, threshold_mask)),
        #     "False positive: ", np.count_nonzero(np.logical_and(truth_mask, ~threshold_mask)),
        #     "False negative: ", np.count_nonzero(np.logical_and(~truth_mask, threshold_mask))
        # )
