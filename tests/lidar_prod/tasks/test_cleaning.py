import os.path as osp
import tempfile
import pytest

from lidar_prod.tasks.cleaning import Cleaner
from lidar_prod.tasks.utils import get_las_metadata, pdal_read_las_array
from tests.conftest import check_las_invariance
from tests.lidar_prod.test_application import check_las_format_versions_and_srs

SRC_LAS_SUBSET_PATH = "tests/files/870000_6618000.subset.postIA.las"


@pytest.mark.parametrize("extra_dims", ([], ""))
def test_cleaning_no_extra_dims(extra_dims):
    cl = Cleaner(extra_dims=extra_dims)

    with tempfile.TemporaryDirectory() as td:
        clean_las_path = osp.join(td, "no_extra_dims.las")
        cl.run(SRC_LAS_SUBSET_PATH, clean_las_path)
        check_las_invariance(SRC_LAS_SUBSET_PATH, clean_las_path)
        a = pdal_read_las_array(clean_las_path)
        las_dimensions = a.dtype.fields.keys()
        # Check that key dims were cleaned out
        assert all(dim not in las_dimensions for dim in ["building", "entropy"])


def test_cleaning_float_extra_dim():
    cl = Cleaner(extra_dims="entropy=float")
    with tempfile.TemporaryDirectory() as td:
        clean_las_path = osp.join(td, "float_extra_dim.las")
        cl.run(SRC_LAS_SUBSET_PATH, clean_las_path)
        check_las_invariance(SRC_LAS_SUBSET_PATH, clean_las_path)
        a = pdal_read_las_array(clean_las_path)
        las_dimensions = a.dtype.fields.keys()
        assert "entropy" in las_dimensions
        assert "building" not in las_dimensions


def test_cleaning_two_float_extra_dims():
    d1 = "entropy"
    d2 = "building"
    extra_dims = [f"{d1}=float", f"{d2}=float"]
    cl = Cleaner(extra_dims=extra_dims)
    with tempfile.TemporaryDirectory() as td:
        clean_las_path = osp.join(td, "float_extra_dim.las")
        cl.run(SRC_LAS_SUBSET_PATH, clean_las_path)
        check_las_invariance(SRC_LAS_SUBSET_PATH, clean_las_path)
        out_a = pdal_read_las_array(clean_las_path)
        assert d1 in out_a.dtype.fields.keys()
        assert d2 in out_a.dtype.fields.keys()


@pytest.mark.parametrize("extra_dims", ("", "entropy=float", "building=float"))
def test_cleaning_format(extra_dims):
    cl = Cleaner(extra_dims=extra_dims)
    with tempfile.TemporaryDirectory() as td:
        clean_las_path = osp.join(td, "float_extra_dim.las")
        cl.run(SRC_LAS_SUBSET_PATH, clean_las_path)
        check_las_format_versions_and_srs(clean_las_path)
