import os
import numpy as np
import pdal
import pytest
from hydra.experimental import compose, initialize

TOLERANCE = 0.0001


def assert_las_invariance(las1, las2):
    a1 = pdal_read_las_array(las1)
    a2 = pdal_read_las_array(las2)
    key_dims = ["X", "Y", "Z", "Infrared", "Red", "Blue", "Green", "Intensity"]
    assert a1.shape == a2.shape  # no loss of points
    assert all(d in a2.dtype.fields.keys() for d in key_dims)  # key dims are here

    # order of points is allowed to change, so we assess a relaxed equality.
    for d in key_dims:
        assert pytest.approx(np.min(a2[d]), TOLERANCE) == np.min(a1[d])
        assert pytest.approx(np.max(a2[d]), TOLERANCE) == np.max(a1[d])
        assert pytest.approx(np.mean(a2[d]), TOLERANCE) == np.mean(a1[d])
        assert pytest.approx(np.sum(a2[d]), TOLERANCE) == np.sum(a1[d])


@pytest.fixture
def default_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config")


# This might be extracted in lidar_prod to have modular steps in application.
def pdal_read_las_array(in_f):
    p1 = pdal.Pipeline() | pdal.Reader.las(in_f)
    p1.execute()
    return p1.arrays[0]
