"""
Takes vegetation probabilities as input, and defines vegetation

"""
import logging

import numpy as np
import laspy


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
        return IoU(
            sum(iou.true_positive for iou in iou_list),
            sum(iou.false_negative for iou in iou_list),
            sum(iou.false_positive for iou in iou_list)
            )


class BasicIdentifier:
    def __init__(
            self,
            threshold: float,
            proba_column: str,
            result_column: str,
            result_code: int,
            data_format,
            evaluate_iou: bool = False,
            truth_column: str = None):
        self.threshold = threshold
        self.proba_column = proba_column
        self.result_column = result_column
        self.result_code = result_code
        self.data_format = data_format
        self.evaluate_iou = evaluate_iou
        self.truth_column = truth_column

    def identify(self, las_data: laspy.lasdata.LasData):

        if self.result_column not in [dim for dim in las_data.point_format.extra_dimension_names]:
            las_data.add_extra_dim(laspy.ExtraBytesParams(name=self.result_column, type="uint32"))

        # get the mask listing the points above the threshold
        threshold_mask = las_data.points[self.proba_column] >= self.threshold
        las_data.points[self.result_column][threshold_mask] = self.result_code

        # calculate ious if necessary
        if self.evaluate_iou:
            self.iou = self.calculate_iou(las_data.points[self.truth_column], self.result_code, threshold_mask)

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
