import pytest
import hydra

from lidar_prod.tasks.bridge_identification_optimization import (
    BridgeIdentificationOptimizer,
)

# Directory with labeled test bridge data
BRIDGE_TEST_INPUT_LAS_DIRNAME = "tests/files/bridges/"
BRIDGE_TEST_OUTPUT_LAS_DIRNAME = "tests/tmp/bridge/opti/"
# name of dim containing perfect bridge probas in all bridge test data
DIM_BRIDGE_PROBAS = "bridge_accurate"


def test_bridge_identification_optimizer(default_hydra_cfg):
    # Use manually crafted bridge probabilities stored in a different dim than default "bridge" dim
    default_hydra_cfg.data_format.las_dimensions.ai_bridge_proba = DIM_BRIDGE_PROBAS
    default_hydra_cfg.bridge_identification.optimization.paths.input_las_dir = (
        BRIDGE_TEST_INPUT_LAS_DIRNAME
    )
    default_hydra_cfg.bridge_identification.optimization.paths.output_dir = (
        BRIDGE_TEST_OUTPUT_LAS_DIRNAME
    )
    default_hydra_cfg.bridge_identification.optimization.optimization_design.n_trials = (
        10
    )
    brio: BridgeIdentificationOptimizer = hydra.utils.instantiate(
        default_hydra_cfg.bridge_identification.optimization
    )
    best_mean_iou = brio.optimize()
    assert best_mean_iou == pytest.approx(1.0, 0.01)
