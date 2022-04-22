import os
import hydra
import numpy as np

import pdal
import os.path as osp
import tempfile

from lidar_prod.tasks.building_validation_optimization import (
    BuildingValidationOptimizer,
)
from tests.conftest import get_a_format_preserving_pdal_pipeline, pdal_read_las_array


IN_F = "tests/files/870000_6618000.subset.postIA.corrected.las"


def test_BVOptimization(default_hydra_cfg):
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
        assert metrics_dict["groups_count"] == 15
        assert metrics_dict["group_no_buildings"] == 0.4
        assert metrics_dict["p_auto"] == 1.0
        assert metrics_dict["p_auto"] == 1.0
        assert metrics_dict["recall"] == 1.0
        assert metrics_dict["precision"] == 1.0

        # Update with final codes and check if they are the right ones.
        bvo.bv.use_final_classification_codes = True
        bvo.update()
        assert os.path.isfile(out_f)
        arr = pdal_read_las_array(out_f)
        # Check that we have either 1/2 (ground/unclassified), or one of
        # the final classification code of the module. here there are no unsure groups
        # so we have only 4 codes.
        final_codes = default_hydra_cfg.data_format.codes.building.final
        expected_codes = {1, 2, final_codes.building, final_codes.not_building}
        actual_codes = {*np.unique(arr["Classification"])}
        assert expected_codes == actual_codes


# We may want to use a large las if we manage to download it. we cannot version it with git.
# Here are the perfs for reference, on
# /home/CGaydon/repositories/Validation_Module/lidar-prod-quality-control/inputs/evaluation_las/V0.5_792000_6272000.las

# """
# groups_count=1493
# group_unsure=0.00402
# group_no_buildings=0.149
# group_building=0.847
# p_auto=0.889
# p_unsure=0.111
# p_refute=0.0924
# p_confirm=0.797
# a_refute=0.899
# a_confirm=0.976
# precision=0.98
# recall=0.99
# Confusion Matrix
# [[   2    1    3]
# [  74  124   25]
# [  89   13 1162]]
# Confusion Matrix (normalized)
# [[0.333 0.167 0.5  ]
# [0.332 0.556 0.112]
# [0.07  0.01  0.919]]
# """

# REFERENCE_METRICS = {"precision": 0.98, "recall":0.98,"p_auto":0.889,}
