from pathlib import Path

import numpy as np
import pytest

from wps_xr.config import config
from wps_xr.wps import _generate_dtype_from_config, open_dataset

test_files = Path(__file__).parents[0] / "test_files"


@pytest.mark.parametrize(
    "signed,wordsize,endian,expected_dtype",
    [
        ("yes", 1, "big", "int8"),
        ("no", 1, "little", "uint8"),
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
    assert _generate_dtype_from_config() == np.dtype(expected_dtype)


@pytest.fixture(scope="session")
def dataset(request):
    return open_dataset(request.param)


@pytest.mark.parametrize(
    "dataset,sample_size",
    [(test_files / "usgs", 100)],
    indirect=["dataset"],
)
def test__add_latlon_coords(dataset, sample_size):
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


@pytest.mark.parametrize(
    "dataset,shape,rowgen,dtype",
    [
        (
            test_files / "synthetic2d",
            (10, 10),
            lambda x: -(x + 1),
            "int8",
        ),
        (
            test_files / "synthetic2d_same_startend",
            (10, 10),
            lambda x: -(x + 1),
            "int8",
        ),
        (
            test_files / "synthetic3d",
            (10, 10, 2),
            lambda x: [-(x + 1)] * 2,
            "int8",
        ),
        (
            test_files / "synthetic2d_flipped",
            (10, 10),
            lambda x: x + 1,
            "uint16",
        ),
        (
            test_files / "synthetic3d_flipped",
            (10, 10, 2),
            lambda x: [x + 1] * 2,
            "uint16",
        ),
    ],
    indirect=["dataset"],
)
def test_synthetic_data(dataset, shape, rowgen, dtype):
    da = dataset[list(dataset.data_vars.keys())[0]]
    assert da.shape == shape
    assert da.dtype == np.dtype(dtype)
    for i, row in enumerate(da.values):
        assert (row == rowgen(i)).all()


@pytest.mark.parametrize(
    "dataset,shape,rowgen,dtype",
    [
        (
            test_files / "synthetic2d_scaled",
            (10, 10),
            lambda x: -2.5 * (x + 1),
            "float64",
        ),
    ],
    indirect=["dataset"],
)
def test_scale_factor(dataset, shape, rowgen, dtype):
    da = dataset[list(dataset.data_vars.keys())[0]]
    assert da.shape == shape
    assert da.dtype == np.dtype(dtype)
    for i, row in enumerate(da.values):
        assert (row == rowgen(i)).all()


@pytest.mark.parametrize(
    "dataset,z_val",
    [
        (test_files / "synthetic3d", np.array([1, 2])),
        (test_files / "synthetic3d_flipped", np.array([2, 3])),
    ],
    indirect=["dataset"],
)
def test_tile_z_start_end(dataset, z_val):
    da = dataset[list(dataset.data_vars.keys())[0]]
    assert (da.z.values == z_val).all()
