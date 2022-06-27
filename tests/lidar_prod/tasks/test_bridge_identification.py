import os
import shutil
import hydra
import pytest

from lidar_prod.tasks.bridge_identification_optimization import (
    BridgeIdentificationOptimizer,
)
from lidar_prod.tasks.utils import stem

OUTPUT_PATH = "tests/tmp/bridge/"
OUTPUT_PATTERN = OUTPUT_PATH + "{stem}-{ai_bridge_proba_name}.las"
BRIDGE_LAS_TEST_FILES_PATTERN = "tests/files/bridges/{basename}"


def setup_module(module):
    """Setup. Create (erase if exists) output path."""
    try:
        shutil.rmtree(OUTPUT_PATH)
    except FileNotFoundError as e:
        pass
    os.makedirs(os.path.dirname(OUTPUT_PATH))


cases = [
    ("prio4_bridge_with_probas.las", "bridge_accurate", 1.0),
    ("prio4_bridge_with_probas.las", "bridge_too_large", 0.35),
    ("prio4_bridge_with_probas.las", "bridge_half", 0.56),
    ("prio4_bridge_with_probas.las", "bridge_with_a_false_positive", 0.5),
    ("prio4_no_bridge_patch_with_probas.las", "bridge_false_positive", 0.0),
    ("prio4_two_bridges_with_probas.las", "bridge_accurate", 1.0),
    ("prio4_two_bridges_with_probas.las", "bridge_one_out_of_two", 0.58),
    ("prio4_two_bridges_with_probas.las", "bridge_no_detection", 0.0),
]


@pytest.mark.parametrize("basename, ai_bridge_proba_name, expected_iou", cases)
def test_bri_compute_IoU(
    default_hydra_cfg, basename, ai_bridge_proba_name, expected_iou
):
    input_las = BRIDGE_LAS_TEST_FILES_PATTERN.format(basename=basename)
    output_las = OUTPUT_PATTERN.format(
        stem=stem(basename), ai_bridge_proba_name=ai_bridge_proba_name
    )
    # Use manually crafted bridge probabilities stored in a different dim than default "bridge" dim
    default_hydra_cfg.data_format.las_dimensions.ai_bridge_proba = ai_bridge_proba_name
    brio: BridgeIdentificationOptimizer = hydra.utils.instantiate(
        default_hydra_cfg.bridge_identification.optimization
    )
    iou = brio.evaluate_one_iou(input_las, output_las)

    assert expected_iou == pytest.approx(iou, abs=0.05)
    print("iou")
