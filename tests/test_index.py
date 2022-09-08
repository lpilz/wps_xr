import copy

import pytest

from wps_xr.config import config
from wps_xr.index import __check_index


@pytest.fixture(scope="session")
def index(request):
    _dict = copy.deepcopy(config.get("index_defaults"))
    _dict.update({"projection": "regular_ll", "type": "continuous"})
    _dict.update(request.param)
    return _dict


@pytest.mark.parametrize(
    "index",
    [
        {"projection": "regular_ll"},
        {"type": "continuous"},
        {"type": "categorical"},
        {"signed": "yes"},
        {"row_order": "top_bottom"},
        {"endian": "little"},
        {"filename_digits": 6},
        {"tile_z_start": 0, "tile_z_end": 1},
        {"category_min": 0, "category_max": 1},
    ],
    indirect=["index"],
)
def test_check_valid_index(index):
    __check_index(index)


@pytest.mark.parametrize(
    "index",
    [
        {"tile_z_end": 1},
        {"tile_z_start": 0},
        {"category_min": 0},
        {"category_max": 1},
        {"tile_z": 2, "tile_z_end": 1},
        {"tile_z": 2, "tile_z_start": 0},
        {"tile_z": 2, "tile_z_start": 0, "tile_z_end": 1},
    ],
    indirect=["index"],
)
def test_check_invalid_index(index):
    with pytest.raises(AssertionError):
        __check_index(index)


@pytest.mark.parametrize(
    "index,projection",
    [
        ({}, "lambert"),
        ({}, "polar"),
        ({}, "mercator"),
        ({}, "albers_nad83"),
        ({}, "polar_wgs84"),
    ],
    indirect=["index"],
)
def test_check_invalid_projection(index, projection):
    index.update({"projection": projection})
    with pytest.raises(NotImplementedError):
        __check_index(index)
