import logging
import os
import os.path as osp
import hydra
from omegaconf import DictConfig
from typing import Optional

from lidar_prod.utils import utils

from lidar_prod.tasks.building_validation import BuildingValidator


log = logging.getLogger(__name__)


@utils.eval_time
def apply(config: DictConfig) -> Optional[float]:
    assert os.path.exists(config.paths.src_las)
    # TODO: add support for decision threhsold when all the rest is stable.
    # assert os.path.exists(
    #     config.building_validation.application.building_validation_thresholds_pickle
    # )

    src_las = config.paths.src_las
    out_dir = config.paths.output_dir

    bv: BuildingValidator = hydra.utils.instantiate(
        config.building_validation.application
    )
    log.info("Prepare LAS...")
    prepared_las_path = osp.join(out_dir, "prepared", osp.basename(src_las))
    bv.prepare(src_las, prepared_las_path)
    log.info("Update LAS...")

    output_las_path = osp.join(out_dir, "final", osp.basename(src_las))
    bv.update(prepared_las_path, output_las_path)
    log.info(f"Updated LAS saved to : {output_las_path}")
