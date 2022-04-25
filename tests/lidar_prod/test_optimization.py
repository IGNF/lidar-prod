import pytest
import os
import os.path as osp
import tempfile
import hydra
import numpy as np


from lidar_prod.tasks.building_validation_optimization import (
    BuildingValidationOptimizer,
)
from tests.conftest import (
    get_a_copy_pdal_pipeline,
    get_a_format_preserving_pdal_pipeline,
    pdal_read_las_array,
)

"""We test the building validation optimizer against two LAS:

These datasets must have the right classification codes, i.e. the ones defined in labels_from_20211001_building_val.
The large LAS cannot be versionned by git. If it is absent from environment, the test is expected to fail.

"""

# Small LAS, for which we optimize thresholds and reach perfect validation, 
# to quickly check optimization logic. 
IN_F = "tests/files/870000_6618000.subset.postIA.corrected.las"
IN_F_EXPECTED = {
    "exact": {
        "groups_count": 15,
        "group_no_buildings": 0.4,
    },
    "min": {
        "p_auto": 1.0,
        "recall": 1.0,
        "precision": 1.0,
    },
}
# Large LAS, for which we evaluate performance, to control that there was no regression in terms of
# automation/precision/recall of building validation.
IN_F_LARGE = "tests/files/V0.5_792000_6272000.las"
IN_F_LARGE_EXPECTED = {
    "exact": {
        "groups_count": 1493,
        "group_no_buildings": 0.149,
        "group_building": 0.847,
    },
    "min": {
        "p_auto": 0.889,
        "recall": 0.98,
        "precision": 0.98,
    },
}

# Relative tolerance when comparing metrics to their expected value for large LAS.
# i.e. resulting metrics are >= (1-tolerance) * expected metrics for performance indicators.
RELATIVE_MIN_TOLERANCE_OF_EXPECTED_METRICS = 0.01


def test_BVOptimization_on_subset(default_hydra_cfg):
    with tempfile.TemporaryDirectory() as td:
        # Optimization output (thresholds and prepared/updated LASfiles) saved to td
        default_hydra_cfg.building_validation.optimization.paths.results_output_dir = td

        # We isolate the input file in a subdir, and prepare it for optimization
        input_las_dir = osp.join(td, "inputs/")
        default_hydra_cfg.building_validation.optimization.paths.input_las_dir = (
            input_las_dir
        )
        os.makedirs(input_las_dir, exist_ok=False)
        in_f_copy = osp.join(input_las_dir, "copy.las")
        pipeline = get_a_format_preserving_pdal_pipeline(IN_F, in_f_copy, [])
        pipeline.execute()

        # Check that a full optimization run can pass successfully
        bvo: BuildingValidationOptimizer = hydra.utils.instantiate(
            default_hydra_cfg.building_validation.optimization
        )
        bvo.run()

        # Assert that a prepared and an updated file are generated in td
        assert os.path.isfile(osp.join(td, "prepared", osp.basename(in_f_copy)))
        out_f = osp.join(td, "updated", osp.basename(in_f_copy))
        assert os.path.isfile(out_f)

        # Check the output of the evaluate method. Note that it uses the
        # prepared data obtained from the full run just above
        metrics_dict = bvo.evaluate()
        assert IN_F_EXPECTED["exact"].items() <= metrics_dict.items()
        for k, v in IN_F_EXPECTED["min"].items():
            assert (
                (1 - RELATIVE_MIN_TOLERANCE_OF_EXPECTED_METRICS) * v
            ) <= metrics_dict[k]
        # Update with final codes and check if they are the right ones.
        bvo.bv.use_final_classification_codes = True
        bvo.update()
        assert os.path.isfile(out_f)
        arr = pdal_read_las_array(out_f)
        # Check that we have either 1/2 (ground/unclassified), or one of
        # the final classification code of the module.
        final_codes = default_hydra_cfg.data_format.codes.building.final
        expected_codes = {
            1,
            2,
            final_codes.building,
            final_codes.not_building,
            final_codes.unsure,
        }
        actual_codes = {*np.unique(arr["Classification"])}
        assert actual_codes.issubset(expected_codes)


@pytest.mark.slow()
def test_BVOptimization_on_large_file(default_hydra_cfg):

    if not os.path.isfile(IN_F_LARGE):
        pytest.xfail(reason=f"File {IN_F_LARGE} is not present in environment.")

    with tempfile.TemporaryDirectory() as td:
        # Optimization output (thresholds and prepared/updated LASfiles) saved to td
        default_hydra_cfg.building_validation.optimization.paths.results_output_dir = td

        # We isolate the input file in a subdir, and prepare it for optimization
        input_las_dir = osp.join(td, "inputs/")
        default_hydra_cfg.building_validation.optimization.paths.input_las_dir = (
            input_las_dir
        )
        os.makedirs(input_las_dir, exist_ok=False)
        in_f_copy = osp.join(input_las_dir, "copy.las")
        pipeline = get_a_copy_pdal_pipeline(IN_F_LARGE, in_f_copy)
        pipeline.execute()

        # Check that a full optimization run can pass successfully
        bvo: BuildingValidationOptimizer = hydra.utils.instantiate(
            default_hydra_cfg.building_validation.optimization
        )
        bvo.prepare()
        metrics_dict = bvo.evaluate()

        exact_expected_val = IN_F_LARGE_EXPECTED["exact"]
        for k in exact_expected_val:
            assert pytest.approx(exact_expected_val[k], RELATIVE_MIN_TOLERANCE_OF_EXPECTED_METRICS) == metrics_dict[k]
        min_expected_val = IN_F_LARGE_EXPECTED["min"]
        for k in min_expected_val:
            assert (
                (1 - RELATIVE_MIN_TOLERANCE_OF_EXPECTED_METRICS) * min_expected_val[k]
            ) <= metrics_dict[k]


# All expected metrics for reference:
"""
    groups_count=1493
    group_unsure=0.00402
    group_no_buildings=0.149
    group_building=0.847
    p_auto=0.889
    p_unsure=0.111
    p_refute=0.0924
    p_confirm=0.797
    a_refute=0.899
    a_confirm=0.976
    precision=0.98
    recall=0.99
    Confusion Matrix
    [[   2    1    3]
    [  74  124   25]
    [  89   13 1162]]
    Confusion Matrix (normalized)
    [[0.333 0.167 0.5  ]
    [0.332 0.556 0.112]
    [0.07  0.01  0.919]]
"""
