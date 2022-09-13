import numpy as np
import pdal

from lidar_prod.tasks.utils import get_pdal_writer, split_idx_by_dim


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
