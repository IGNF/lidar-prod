import numpy as np
import pytest

from lidar_prod.tasks.utils import extract_coor, split_idx_by_dim


@pytest.mark.parametrize(
    "las_name, x_span, y_span, buffer, x_min, y_min, x_max, y_max",
    [
        ("922000_6307000.las", 1000, 1000, 50, 921950, 6305950, 923050, 6307050),
        ("5000_4000_stupid_lasname.las", 1000, 1000, 50, 4950, 2950, 6050, 4050),
    ],
)
def test_extract_coor(las_name, x_span, y_span, buffer, x_min, y_min, x_max, y_max):
    assert extract_coor(las_name, x_span, y_span, buffer) == (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def test_split_idx_by_dim():
    d1 = [1, 1, 2, 3]
    d2 = [10, 10, 20, 30]
    dim_array = np.array([d1, d2]).transpose()

    group_idx = split_idx_by_dim(dim_array[:, 0])
    expected_groups = [np.array([0, 1]), np.array([2]), np.array([3])]
    expected_values = [np.array([10, 10]), np.array([20]), np.array([30])]

    assert len(group_idx) == 3
    for i, group in enumerate(group_idx):
        assert (group == expected_groups[i]).all()
        assert (group == expected_groups[i]).all()


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
        assert (group == expected_groups[i]).all()
        assert (group == expected_groups[i]).all()
