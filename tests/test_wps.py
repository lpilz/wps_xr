import os

import numpy as np
import pytest

from wps_xr.config import config
from wps_xr.wps import _add_latlon_coords, _generate_dtype, open_dataset

test_files = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_files")


@pytest.mark.parametrize(
    "signed,wordsize,endian,expected_dtype",
    [
        ("yes", 1, "big", "i1"),
        ("no", 1, "little", "u1"),
        ("yes", 2, "big", ">i2"),
        ("yes", 2, "little", "<i2"),
        ("no", 2, "big", ">u2"),
        ("no", 2, "little", "<u2"),
        ("yes", 4, "big", ">i4"),
        ("yes", 4, "little", "<i4"),
        ("no", 4, "big", ">u4"),
        ("no", 4, "little", "<u4"),
    ],
)
def test__generate_dtype(signed, wordsize, endian, expected_dtype):
    config.set(
        {"index.signed": signed, "index.wordsize": wordsize, "index.endian": endian}
    )
    assert _generate_dtype() == np.dtype(expected_dtype)


@pytest.fixture(scope="session")
def dataset(request):
    return open_dataset(request.param)


@pytest.mark.parametrize(
    "dataset,sample_size",
    [(f"{os.path.join(test_files,'usgs')}", 100)],
    indirect=["dataset"],
)
def test__add_latlon_coords(dataset, sample_size):
    dataset = _add_latlon_coords(dataset)
    assert "lat" in dataset.variables and "lat" in dataset.coords
    assert "lon" in dataset.variables and "lon" in dataset.coords
    for x, y in zip(
        np.random.randint(dataset.dims["x"], size=sample_size),
        np.random.randint(dataset.dims["y"], size=sample_size),
    ):
        smp = dataset.sel(x=x, y=y)
        assert np.isclose(
            smp["x"],
            (smp["lon"] - smp.attrs["known_lon"]) / smp.attrs["dx"]
            + smp.attrs["known_x"],
        )
        assert np.isclose(
            smp["y"],
            (smp["lat"] - smp.attrs["known_lat"]) / smp.attrs["dy"]
            + smp.attrs["known_y"],
        )
