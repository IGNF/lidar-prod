import os
import os.path as osp
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from lidar_prod.optimization import optimize_building

TMP_DIR = Path("tmp/lidar_prod/optimization")
LAS_SUBSET_FILE = "tests/files/870000_6618000.subset.postIA.corrected.las"


def setup_module(module):
    try:
        shutil.rmtree(TMP_DIR)
    except FileNotFoundError:
        pass
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def test_optimize_building_on_subset(hydra_cfg):
    out_dir = str(TMP_DIR / "subset")
    # Optimization output (thresholds and prepared/updated LASfiles) saved to out_dir
    hydra_cfg.building_validation.optimization.paths.results_output_dir = out_dir

    # We isolate the input file in a subdir, and prepare it for optimization
    input_las_dir = osp.join(out_dir, "inputs/")
    hydra_cfg.building_validation.optimization.paths.input_las_dir = input_las_dir
    hydra_cfg.building_validation.application.thresholds = "NO THRESHOLDS"
    os.makedirs(input_las_dir, exist_ok=False)
    src_las_copy_path = osp.join(input_las_dir, "copy.las")
    shutil.copy(LAS_SUBSET_FILE, src_las_copy_path)

    optimize_building(hydra_cfg)

    # Check that the expected outputs are saved successfully
    th_yaml = hydra_cfg.building_validation.optimization.paths.building_validation_thresholds
    assert os.path.isfile(th_yaml)
    cfg_yaml = hydra_cfg.building_validation.optimization.paths.output_optimized_config
    assert os.path.isfile(cfg_yaml)

    assert os.path.isfile(osp.join(out_dir, "prepared", osp.basename(src_las_copy_path)))
    updated_las_path = osp.join(out_dir, "updated", osp.basename(src_las_copy_path))
    assert os.path.isfile(updated_las_path)

    # Check that thte thresholds are saved correctly in output config file
    out_cfg = OmegaConf.load(cfg_yaml)
    assert out_cfg.building_validation.application.thresholds != "NO THRESHOLDS"
