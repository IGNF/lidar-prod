import numpy as np
import pdal
import shutil
from pathlib import Path
import pytest
import geopandas as gdb

from lidar_prod.tasks.utils import (
    get_pdal_writer,
    split_idx_by_dim,
    request_bd_uni_for_building_shapefile,
)


TMP_DIR = Path("tmp/lidar_prod/tasks/utils")


def setup_module(module):
    try:
        shutil.rmtree(TMP_DIR)
    except FileNotFoundError:
        pass
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_las_data_within_bounds(synthetic_las_path: str, bbox) -> None:
    """Creates a synthetic LAS contained within given bbox.

    Args:
        synthetic_las_path (str): path to save the synthetic LAS.
        bbox (_type_): bounding box (example key: `x_min`).

    """
    bounds = (
        f'([{bbox["x_min"]},{bbox["x_max"]}],[{bbox["y_min"]},{bbox["y_max"]}],[0,100])'
    )
    pipeline = pdal.Pipeline()
    pipeline |= pdal.Reader.faux(
        filename="no_file.las", mode="ramp", count=100, bounds=bounds
    )
    pipeline |= get_pdal_writer(synthetic_las_path)
    pipeline.execute()


def test_split_idx_by_dim():
    d1 = [1, 1, 2, 3]
    d2 = [10, 10, 20, 30]
    dim_array = np.array([d1, d2]).transpose()

    group_idx = split_idx_by_dim(dim_array[:, 0])
    expected_groups = [np.array([0, 1]), np.array([2]), np.array([3])]
    expected_values = [np.array([10, 10]), np.array([20]), np.array([30])]

    assert len(group_idx) == 3
    for i, group in enumerate(group_idx):
        assert np.array_equal(group, expected_groups[i])
        assert np.array_equal(dim_array[group, 1], expected_values[i])


def test_split_idx_by_dim_unordered():
    """
    This also works if split dimension is not sorted.
    The expected values are still sorted.
    """
    d1 = [1, 3, 2, 1]
    d2 = [10, 30, 20, 10]
    dim_array = np.array([d1, d2]).transpose()

    group_idx = split_idx_by_dim(dim_array[:, 0])
    expected_groups = [np.array([0, 3]), np.array([2]), np.array([1])]
    expected_values = [np.array([10, 10]), np.array([20]), np.array([30])]

    assert len(group_idx) == 3
    for i, group in enumerate(group_idx):
        assert np.array_equal(group, expected_groups[i])
        assert np.array_equal(dim_array[group, 1], expected_values[i])


@pytest.mark.parametrize(
    "bbox,epsg,out_shp,has_data",
    [
        (
            # Bbox in Metropolitan France, with correct epsg => output should not be empty
            dict(x_min=870150, y_min=6616950, x_max=870350, y_max=6617200),
            2154,
            "metropolitan_ok.shp",
            True,
        ),
        (
            # Bbox in St Barthelemy with correct epsg => output should not be empty
            dict(x_min=515000, y_min=1981000, x_max=515100, y_max=1981100),
            5490,
            "st_barth_ok.shp",
            True,
        ),
        (
            # Bbox in St Barthelemy with wrong epsg => output should be empty
            dict(x_min=515000, y_min=1981000, x_max=515100, y_max=1981100),
            2154,
            "st_barth_nok.shp",
            False,
        ),
    ],
)
def test_request_bd_uni_for_building_shapefile(
    hydra_cfg, bbox, epsg, out_shp, has_data
):
    out_path = TMP_DIR / out_shp
    request_bd_uni_for_building_shapefile(
        hydra_cfg.bd_uni_connection_params,
        shapefile_path=out_path,
        bbox=bbox,
        epsg=epsg,
    )
    assert out_path.is_file()
    gdf = gdb.read_file(out_path)
    print(gdf)
    assert bool(len(gdf.index)) == has_data
