import tempfile
import numpy as np
import pdal
import pytest

from lidar_prod.tasks.cleaning import Cleaner
import os.path as osp

from lidar_prod.tasks.utils import get_las_metadata

IN_F = "tests/files/870000_6618000.subset.postIA.las"


def pdal_read_las_array(in_f):
    p1 = pdal.Pipeline() | pdal.Reader.las(in_f)
    p1.execute()
    return p1.arrays[0]


def assert_las_key_dims_equality(las1, las2):
    a1 = pdal_read_las_array(las1)
    a2 = pdal_read_las_array(las2)
    key_dims = ["X", "Y", "Z", "Infrared", "Red", "Blue", "Green", "Intensity"]
    assert np.array_equal(a1[key_dims], a2[key_dims])


@pytest.mark.parametrize("extra_dims", ([], None, "", 0))
def test_cleaning_no_extra_dims(extra_dims):
    cl = Cleaner(extra_dims=extra_dims)

    with tempfile.TemporaryDirectory() as td:
        out_f = osp.join(td, "no_extra_dims.las")
        cl.run(IN_F, out_f)
        assert_las_key_dims_equality(IN_F, out_f)


@pytest.mark.parametrize("extra_dims", ("entropy=float", "building=float"))
def test_cleaning_float_extra_dim(extra_dims):
    cl = Cleaner(extra_dims=extra_dims)
    with tempfile.TemporaryDirectory() as td:
        out_f = osp.join(td, "float_extra_dim.las")
        cl.run(IN_F, out_f)
        assert_las_key_dims_equality(IN_F, out_f)


def test_cleaning_two_float_extra_dims():
    d1 = "entropy"
    d2 = "building"
    extra_dims = [f"{d1}=float", f"{d2}=float"]
    cl = Cleaner(extra_dims=extra_dims)
    with tempfile.TemporaryDirectory() as td:
        out_f = osp.join(td, "float_extra_dim.las")
        cl.run(IN_F, out_f)
        assert_las_key_dims_equality(IN_F, out_f)
        out_a = pdal_read_las_array(out_f)
        assert d1 in out_a.dtype.fields.keys()
        assert d2 in out_a.dtype.fields.keys()


@pytest.mark.parametrize("extra_dims", ("", "entropy=float", "building=float"))
def test_cleaning_format(extra_dims):
    cl = Cleaner(extra_dims=extra_dims)
    with tempfile.TemporaryDirectory() as td:
        out_f = osp.join(td, "float_extra_dim.las")
        cl.run(IN_F, out_f)
        metadata = get_las_metadata(out_f)
        assert metadata["minor_version"] == 4
        assert metadata["dataformat_id"] == 8
