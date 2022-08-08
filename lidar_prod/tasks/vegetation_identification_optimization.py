"""
Takes bridge probabilities as input, and defines bridge.

"""

import logging
import hydra
import optuna
from omegaconf import DictConfig
from lidar_prod.tasks.vegetation_identification import BasicIdentifier, IoU

from lidar_prod.tasks.utils import get_points_from_las
from application import get_list_las_path_from_src, process_one_file

log = logging.getLogger(__name__)

class BasicIdentifierOptimizer:
    def __init__(
        self,
        config: DictConfig,
        proba_column: str,
        result_column: str,
        result_code: int,
        truth_column: str,
        n_trials: int,
        ):
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        self.config = config
        self.proba_column = proba_column
        self.result_column = result_column
        self.result_code = result_code
        self.truth_column = truth_column
        self.n_trials = n_trials

    def optimize(self):
        """search the best threshold"""
        self.study.optimize(
            self._optuna_objective_func, self.n_trials
        )
        trial = self.study.best_trial
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))    

    def _optuna_objective_func(self, trial):
        """Sets decision threshold for the trial"""
        threshold = trial.suggest_float("min_threshold_proba", 0.0, 1.0)

        iou_list = []
        for src_las_path in get_list_las_path_from_src(self.config.paths.src_las):
            points = get_points_from_las(src_las_path)

            basic_identifier = BasicIdentifier(threshold, self.proba_column, self.result_column, self.result_code, self.config["data_format"], True, self.truth_column)
            basic_identifier.identify(points)
            iou_list.append(basic_identifier.iou)
        return IoU.combine_iou(iou_list).iou    # return the combined IoU of all the .las
