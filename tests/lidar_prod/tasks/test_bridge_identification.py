import os
import shutil
import hydra
import pytest
from shapely.geometry import Polygon

from lidar_prod.tasks.bridge_identification_optimization import (
    BridgeIdentificationOptimizer,
    compute_bridge_iou,
    save_geometries_to_geodataframe,
)
from lidar_prod.tasks.utils import stem

OUTPUT_PATH = "tests/tmp/bridge/"
OUTPUT_LAS_AFTER_IOU_COMPUTATION_PATTERN = (
    OUTPUT_PATH + "{stem}-{ai_bridge_proba_name}.las"
)
JSON_TEST_FILE_FOR_IOU_CALCULATION_PATTERN = (
    OUTPUT_PATH + "{test_case_name}-{target_or_predicted}.json"
)
BRIDGE_LAS_TEST_FILES_PATTERN = "tests/files/bridges/{basename}"


def setup_module(module):
    """Setup. Create (erase if exists) output path."""
    try:
        shutil.rmtree(OUTPUT_PATH)
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(OUTPUT_PATH))


bri_cases = [
    ("prio4_bridge_with_probas.las", "bridge_accurate", 1.0),
    ("prio4_bridge_with_probas.las", "bridge_too_large", 0.35),
    ("prio4_bridge_with_probas.las", "bridge_half", 0.56),
    ("prio4_bridge_with_probas.las", "bridge_with_a_false_positive", 0.5),
    ("prio4_no_bridge_patch_with_probas.las", "bridge_false_positive", 0.0),
    ("prio4_two_bridges_with_probas.las", "bridge_accurate", 1.0),
    ("prio4_two_bridges_with_probas.las", "bridge_one_out_of_two", 0.58),
    ("prio4_two_bridges_with_probas.las", "bridge_no_detection", 0.0),
]


@pytest.mark.parametrize("basename, ai_bridge_proba_name, expected_iou", bri_cases)
def test_bridge_identifier_by_evaluation_of_iou(
    default_hydra_cfg, basename, ai_bridge_proba_name, expected_iou
):
    input_las = BRIDGE_LAS_TEST_FILES_PATTERN.format(basename=basename)
    output_las = OUTPUT_LAS_AFTER_IOU_COMPUTATION_PATTERN.format(
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


square = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
half_square = Polygon([(0, 0), (0, 1), (1 / 2, 1), (1 / 2, 0)])
empty_polygon = Polygon([])
iou_computation_cases = [
    ("both_empty", [empty_polygon], [empty_polygon], 1.0),
    ("identity", [square], [square], 1.0),
    ("doubled_target", [square, square], [square], 1.0),
    ("doubled_predicted", [square], [square, square], 1.0),
    ("no_effect_of_additional_empty_target", [square, empty_polygon], [square], 1.0),
    ("no_effect_of_additional_empty_predicted", [square], [square, empty_polygon], 1.0),
    ("empty_target", [empty_polygon], [square], 0.0),
    ("empty_predicted", [square], [empty_polygon], 0.0),
    ("target_is_half_predicted", [half_square], [square], 0.5),
    ("predicted_is_half_target", [square], [half_square], 0.5),
]


@pytest.mark.parametrize(
    "test_case_name, target_geometry, predicted_geometry, expected_iou",
    iou_computation_cases,
)
def test_bri_compute_bridge_iou(
    test_case_name, target_geometry, predicted_geometry, expected_iou
):
    # Create fake json files based on input
    json_path_target = JSON_TEST_FILE_FOR_IOU_CALCULATION_PATTERN.format(
        test_case_name=test_case_name, target_or_predicted="target"
    )
    json_path_predicted = JSON_TEST_FILE_FOR_IOU_CALCULATION_PATTERN.format(
        test_case_name=test_case_name, target_or_predicted="predicted"
    )
    save_geometries_to_geodataframe(target_geometry, json_path_target)
    save_geometries_to_geodataframe(predicted_geometry, json_path_predicted)

    iou = compute_bridge_iou(json_path_target, json_path_predicted)
    assert iou == expected_iou
