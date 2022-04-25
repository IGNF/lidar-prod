import os
import shutil
import tempfile
from typing import Any, Iterable
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


def assert_las_contains_dims(las1, dims_to_check=[]):
    a1 = pdal_read_las_array(las1)
    for d in dims_to_check:
        assert d in a1.dtype.fields.keys()


@pytest.fixture
def default_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config")


# This might be extracted in lidar_prod to have modular steps in application.
def pdal_read_las_array(in_f):
    p1 = pdal.Pipeline() | pdal.Reader.las(in_f)
    p1.execute()
    return p1.arrays[0]


# TODO: this could be used in code to avoid pdal boilerplate
def get_a_format_preserving_pdal_pipeline(in_f: str, out_f: str, ops: Iterable[Any]):
    """Create a pdal pipeline, preserving format, forwarding every dimension.

    Args:
        in_f (str): input LAS path
        out_f (str): output LAS path
        ops (Iterable[Any]): list of pdal operation (e.g. Filter.assign(...))

    """
    pipeline = pdal.Pipeline()
    pipeline |= pdal.Reader.las(
        filename=in_f,
        nosrs=True,
        override_srs="EPSG:2154",
    )

    for op in ops:
        pipeline |= op

    pipeline |= pdal.Writer.las(
        filename=out_f,
        forward="all",
        extra_dims="all",
        minor_version=4,
        dataformat_id=8,
    )
    return pipeline


def get_a_copy_pdal_pipeline(in_f: str, out_f: str):
    """Get a pipeline that will copy in_f to out_f, preserving format."""
    return get_a_format_preserving_pdal_pipeline(in_f, out_f, [])


def _isolate_and_copy(in_f: str):
    """Copy this file to be sure that the test is isolated."""
    isolated_f_copy = tempfile.NamedTemporaryFile().name
    shutil.copy(in_f, isolated_f_copy)
    return isolated_f_copy


def _isolate_and_remove_all_candidate_buildings_points(in_f: str):
    """Set Classification to 1 for all points, thus mimicking a LAS without candidates
    Consequence: no candidate groups. Nothing to update. No building completion.

    Args:
        in_f (str): input LAS path

    """
    isolated_f_copy = tempfile.NamedTemporaryFile().name
    ops = [pdal.Filter.assign(value=f"Classification = 1")]
    pipeline = get_a_format_preserving_pdal_pipeline(in_f, isolated_f_copy, ops)
    pipeline.execute()
    return isolated_f_copy


def _isolate_and_have_null_probability_everywhere(in_f: str):
    """Set building probability to 0 for all points, thus mimicking a low confidence everywhere.
    Consequences : no building in building identification. Only refuted or unsure elements.

    Args:
        in_f (str): input LAS path

    """
    isolated_f_copy = tempfile.NamedTemporaryFile().name
    ops = [pdal.Filter.assign(value=f"building = 0.0")]
    pipeline = get_a_format_preserving_pdal_pipeline(in_f, isolated_f_copy, ops)
    pipeline.execute()
    return isolated_f_copy