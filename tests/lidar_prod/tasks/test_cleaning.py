import os.path as osp
import tempfile
import pytest

from lidar_prod.tasks.cleaning import Cleaner
from lidar_prod.tasks.utils import get_las_metadata, pdal_read_las_array
from tests.conftest import check_las_invariance

IN_F = "tests/files/870000_6618000.subset.postIA.las"


@pytest.mark.parametrize("extra_dims", ([], None, "", 0))
def test_cleaning_no_extra_dims(extra_dims):
    cl = Cleaner(extra_dims=extra_dims)

    with tempfile.TemporaryDirectory() as td:
        out_f = osp.join(td, "no_extra_dims.las")
        cl.run(IN_F, out_f)
        check_las_invariance(IN_F, out_f)
        a = pdal_read_las_array(out_f)
        las_dimensions = a.dtype.fields.keys()
        assert all(
            dim not in las_dimensions for dim in ["building", "entropy"]
        )  # key dims were cleaned out


def test_cleaning_float_extra_dim():
    cl = Cleaner(extra_dims="entropy=float")
    with tempfile.TemporaryDirectory() as td:
        out_f = osp.join(td, "float_extra_dim.las")
        cl.run(IN_F, out_f)
        check_las_invariance(IN_F, out_f)
        a = pdal_read_las_array(out_f)
        las_dimensions = a.dtype.fields.keys()
        assert "entropy" in las_dimensions
        assert "building" not in las_dimensions


def test_cleaning_two_float_extra_dims():
    d1 = "entropy"
    d2 = "building"
    extra_dims = [f"{d1}=float", f"{d2}=float"]
    cl = Cleaner(extra_dims=extra_dims)
    with tempfile.TemporaryDirectory() as td:
        out_f = osp.join(td, "float_extra_dim.las")
        cl.run(IN_F, out_f)
        check_las_invariance(IN_F, out_f)
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
