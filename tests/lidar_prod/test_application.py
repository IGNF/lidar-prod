import os
import tempfile

import numpy as np
import pdal
import pyproj
import pytest
from omegaconf import open_dict

from lidar_prod.application import (
    apply,
    apply_building_module,
    get_shapefile,
    identify_vegetation_unclassified,
    just_clean,
)
from lidar_prod.tasks.utils import (
    get_a_las_to_las_pdal_pipeline,
    get_las_data_from_las,
    get_input_las_metadata,
    get_pipeline,
)
from tests.conftest import (
    check_las_contains_dims,
    check_las_invariance,
    pdal_read_las_array,
)

LAS_SUBSET_FILE_BUILDING = "tests/files/870000_6618000.subset.postIA.las"
SHAPE_FILE = "tests/files/870000_6618000.subset.postIA.shp"
LAS_SUBSET_FILE_VEGETATION = "tests/files/436000_6478000.subset.postIA.las"
LAZ_SUBSET_FILE_VEGETATION = "tests/files/436000_6478000.subset.postIA.laz"
DUMMY_DIRECTORY_PATH = "tests/files/dummy_folder"
DUMMY_FILE_PATH = "tests/files/dummy_folder/dummy_file1.las"


@pytest.mark.parametrize(
    "las_mutation, query_db_Uni",
    [
        ([], True),  # identity
        (
            [pdal.Filter.assign(value="building = 0.0")],
            True,
        ),  # low proba everywhere
        (
            [pdal.Filter.assign(value="Classification = 1")],
            False,
        ),  # no candidate buildings
        (
            [pdal.Filter.assign(value="Classification = 202")],
            False,
        ),  # only candidate buildings
    ],  # if query_db_Uni = True, will query database to get a shapefile, otherwise use a prebuilt one
)
def test_application_data_invariance_and_data_format(hydra_cfg, las_mutation, query_db_Uni):
    """We test the application against a LAS subset (~2500m²).

    Data contains a few buildings, a few classification mistakes, and necessary fields (building probability, entropy)
    for validation.
    We apply different "mutations" to the data in order to test for multiple scenarii.

    Expected classification codes after application run are either default=0, unclassified=1, or
    one of the decision codes.

    """
    _fc = hydra_cfg.data_format.codes.building.final
    expected_codes = {
        1,
        2,
        _fc.building,
        _fc.not_building,
        _fc.unsure,
    }
    # Run application on the data subset
    with tempfile.TemporaryDirectory() as hydra_cfg.paths.output_dir:
        # Copy the data and apply the "mutation"
        mutated_copy: str = tempfile.NamedTemporaryFile().name
        pipeline = get_a_las_to_las_pdal_pipeline(
            LAS_SUBSET_FILE_BUILDING,
            mutated_copy,
            las_mutation,
            hydra_cfg.data_format.epsg,
        )
        pipeline.execute()
        hydra_cfg.paths.src_las = mutated_copy
        if not query_db_Uni:  # we don't request db_uni, we use a shapefile instead
            hydra_cfg.building_validation.application.shp_path = SHAPE_FILE
        updated_las_path_list = apply(hydra_cfg, apply_building_module)
        # Check output
        check_las_invariance(
            mutated_copy, updated_las_path_list[0], hydra_cfg.data_format.epsg
        )
        check_format_of_application_output_las(
            updated_las_path_list[0], hydra_cfg.data_format.epsg, expected_codes
        )


def check_format_of_application_output_las(
    output_las_path: str, epsg: int | str, expected_codes: dict
):
    """Check LAS format, dimensions, and classification codes of output

    Args:
        output_las_path (str): path of output LAS
        epsg (int | str): epsg code for the file (if empty or None: infer
        it from the las metadata). Used to read the data
        expected_codes (dict): set of expected classification codes.

    """
    # Check that we contain extra_dims that production needs
    check_las_contains_dims(output_las_path, epsg, dims_to_check=["Group", "entropy"])

    # Ensure that the format versions are as expected
    check_las_format_versions_and_srs(output_las_path, epsg)

    # Check that we have either 1/2 (ground/unclassified),
    # or one of the three final classification code of the module
    arr1 = pdal_read_las_array(output_las_path, epsg)
    actual_codes = {*np.unique(arr1["Classification"])}
    assert actual_codes.issubset(expected_codes)


def check_las_format_versions_and_srs(input_path: str, epsg: int | str):
    pipeline = get_pipeline(input_path, epsg)
    metadata = get_input_las_metadata(pipeline)
    assert metadata["minor_version"] == 4
    assert metadata["dataformat_id"] == 8
    # Ensure that the final spatial reference is the same as in the config (if provided)
    metadata_crs = metadata["srs"]["compoundwkt"]
    assert metadata_crs
    if epsg:
        expected_crs = pyproj.crs.CRS(epsg)
        assert expected_crs.equals(metadata_crs)


@pytest.mark.parametrize(
    "las_file",
    [LAS_SUBSET_FILE_VEGETATION, LAZ_SUBSET_FILE_VEGETATION],
)
def test_just_clean(vegetation_unclassifed_hydra_cfg, las_file):
    destination_path = tempfile.NamedTemporaryFile().name
    just_clean(vegetation_unclassifed_hydra_cfg, las_file, destination_path)
    las_data = get_las_data_from_las(destination_path)
    assert [dim for dim in las_data.point_format.extra_dimension_names] == [
        "entropy",
        "vegetation",
        "unclassified",
    ]


def test_detect_vegetation_unclassified(vegetation_unclassifed_hydra_cfg):
    destination_path = tempfile.NamedTemporaryFile().name
    identify_vegetation_unclassified(
        vegetation_unclassifed_hydra_cfg,
        LAS_SUBSET_FILE_VEGETATION,
        destination_path,
    )
    las_data = get_las_data_from_las(destination_path)
    vegetation_count = np.count_nonzero(las_data.points.classification == vegetation_unclassifed_hydra_cfg.data_format.codes.vegetation)
    unclassified_count = np.count_nonzero(las_data.points.classification == vegetation_unclassifed_hydra_cfg.data_format.codes.unclassified)
    assert vegetation_count == 17
    assert unclassified_count == 23222


@pytest.mark.parametrize(
    "path, expected",
    [
        (DUMMY_DIRECTORY_PATH, ["dummy_file1.las", "dummy_file2.las"]),
        (DUMMY_FILE_PATH, ["dummy_file1.las"]),
    ],
)
def test_applying(vegetation_unclassifed_hydra_cfg, path, expected):
    def dummy_method(config, src_las_path, target_las_path):
        assert os.path.basename(src_las_path) in config.expected
        assert os.path.basename(target_las_path) in config.expected

    vegetation_unclassifed_hydra_cfg.paths.src_las = path
    with tempfile.TemporaryDirectory() as td:
        vegetation_unclassifed_hydra_cfg.paths.output_dir = td
    with open_dict(vegetation_unclassifed_hydra_cfg):  # needed to open the config dict and add elements
        vegetation_unclassifed_hydra_cfg.expected = expected
    apply(vegetation_unclassifed_hydra_cfg, dummy_method)


def test_get_shapefile(hydra_cfg):
    destination_path = tempfile.NamedTemporaryFile().name
    get_shapefile(hydra_cfg, LAS_SUBSET_FILE_BUILDING, destination_path)
    created_shapefile_path = os.path.join(
        os.path.dirname(destination_path),
        os.path.splitext(os.path.basename(LAS_SUBSET_FILE_BUILDING))[0] + ".shp",
    )
    assert os.path.exists(created_shapefile_path)
