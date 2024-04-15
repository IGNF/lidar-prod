import logging
from typing import Union

import optuna
from omegaconf import DictConfig

from lidar_prod.application import get_list_las_path_from_src
from lidar_prod.tasks.basic_identification import BasicIdentifier, IoU
from lidar_prod.tasks.utils import get_las_data_from_las

log = logging.getLogger(__name__)


class BasicIdentifierOptimizer:
    def __init__(
        self,
        config: DictConfig,
        proba_column: str,
        result_column: str,
        result_code: int,
        target_column: str,
        n_trials: int,
        target_result_code: Union[int, list] = None,
    ) -> None:
        """
        Search the best threshold for BasicIdentifier

        args:
            config: the hydra config dictionnary
            proba_column: the column the treshold is compared against
            result_column: the column that store the result of the comparison
            result_code: the value the point will be set to
            target_column: the column with the target results to compare againt
            n_trials: number of trials to get the best IoU
            target_result_code: the code(s) defining the points with the target results.
                Can be an int of a list of int, if we want an IoU but
                target_result_code is not provided then result_code is used instead.
        """

        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler()
        )
        self.config = config
        self.proba_column = proba_column
        self.result_column = result_column
        self.result_code = result_code
        self.truth_column = target_column
        self.n_trials = n_trials
        self.truth_result_code = target_result_code if target_result_code else result_code

    def optimize(self) -> None:
        """Search the best threshold."""
        self.study.optimize(self._optuna_objective_func, self.n_trials)
        for key, value in self.study.best_trial.params.items():
            print(f"    {key}: {value}")

    def _optuna_objective_func(self, trial) -> IoU:
        """Get the best IoU"""
        threshold = trial.suggest_float("min_threshold_proba", 0.0, 1.0)

        iou_list = []
        for src_las_path in get_list_las_path_from_src(self.config.paths.src_las):
            las_data = get_las_data_from_las(src_las_path)
            basic_identifier = BasicIdentifier(
                threshold,
                self.proba_column,
                self.result_column,
                self.result_code,
                True,
                self.truth_column,
                self.truth_result_code,
            )
            basic_identifier.identify(las_data)
            iou_list.append(basic_identifier.iou)
        return IoU.combine_iou(iou_list).iou  # return the combined IoU of all the .las
