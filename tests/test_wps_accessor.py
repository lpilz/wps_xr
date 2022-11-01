import math
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wps_xr import config
from wps_xr.wps import open_dataset
from wps_xr.wps_accessor import _pad_data_if_needed, _prepare_wps_directory

test_files = Path(__file__).parents[0] / "test_files"


def get_num_filename(lst):
    return 5 if np.max(lst) <= 99999 else 6 if np.max(lst) <= 999999 else None


def gen_expexcted_filelist(ds, tile_size):
    # get dimensions
    x_ext, y_ext = ds.dims["x"], ds.dims["y"]

    # number of tiles in each direction
    x_num, y_num = [
        math.ceil(ext / size) for ext, size in zip((x_ext, y_ext), tile_size)
    ]

    # get upper bound of x,y indices for files
    def get_tiles(n, s):
        return list(range(s, n * s + 1, s))

    x_idx, y_idx = [get_tiles(n, s) for n, s in zip((x_num, y_num), tile_size)]

    # get list of combined x,y indices
    idx_list = product(x_idx, y_idx)

    num_filename = get_num_filename([*x_idx, *y_idx])

    def _fmt(x):
        return f"{x:0{num_filename}d}"

    return [
        f"{_fmt(x-(tile_size[0]-1))}-{_fmt(x)}.{_fmt(y-(tile_size[1]-1))}-{_fmt(y)}"
        for x, y in idx_list
    ]


@pytest.fixture(scope="session")
def dataset(request):
    return open_dataset(request.param)


@pytest.mark.parametrize(
    "dataset,tile_size",
    [
        (test_files / "usgs", (1200, 1200)),
        (test_files / "usgs", (250, 250)),
        (test_files / "usgs", (200, 200)),
    ],
    indirect=["dataset"],
)
def test_to_disk(tmp_path_factory, dataset, tile_size):
    """Tests WPSAccessor.to_disk file output"""
    missing_val = 127
    config.set({"index.missing_value": missing_val})

    dn = tmp_path_factory.mktemp(f"out_{tile_size[0]}-{tile_size[1]}")
    dataset.wps.to_disk(dn, tile_size=tile_size, force=True)
    indexfile = dn / "index"
    assert indexfile.exists() and indexfile.is_file()

    expected_filelist = gen_expexcted_filelist(dataset, tile_size)
    for _file in expected_filelist:
        __file = dn / _file
        assert __file.exists() and __file.is_file()
        assert __file.stat().st_size == np.prod(tile_size)


@pytest.mark.parametrize(
    "dataset,tile_size,will_need_padding",
    [
        (test_files / "usgs", (1200, 1200), False),
        (test_files / "usgs", (250, 250), True),
    ],
    indirect=["dataset"],
)
def test_to_disk_padding(tmp_path_factory, dataset, tile_size, will_need_padding):
    """Tests WPSAccessor.to_disk padding feature"""
    missing_val = 127
    config.set({"index.missing_value": missing_val})

    dn = tmp_path_factory.mktemp(f"out_{tile_size[0]}-{tile_size[1]}")
    dataset.wps.to_disk(dn, tile_size=tile_size, force=True)

    ds_out = open_dataset(dn)
    var_name = dn.name
    assert ds_out[var_name].isnull().any() == will_need_padding
    if will_need_padding:
        assert ds_out[var_name].isel(x=-1, y=-1).isnull()


@pytest.mark.parametrize(
    "dataset,tile_size",
    [
        (test_files / "usgs", (200, 200)),
    ],
    indirect=["dataset"],
)
def test_to_disk_err(tmp_path_factory, dataset, tile_size):
    dataset["foo"] = xr.Variable(dims="bar", data=np.zeros(1))

    dn = tmp_path_factory.mktemp(f"out_{tile_size[0]}-{tile_size[1]}")
    with pytest.raises(LookupError):
        dataset.wps.to_disk(dn, tile_size=tile_size, force=True)

    with pytest.raises(Exception):
        dataset.wps.to_disk(dn, var=["foo", "usgs"], tile_size=tile_size, force=True)

    with pytest.raises(KeyError):
        del config.config["index"]["tile_x"]
        del config.config["index"]["tile_y"]
        dataset["usgs"] = dataset["usgs"].compute()
        dataset.wps.to_disk(dn, var="usgs", force=True)


def test__prepare_wps_directory(tmp_path_factory):
    pth = tmp_path_factory.mktemp("tmppath")
    with pytest.raises(FileExistsError):
        _prepare_wps_directory(pth)
    _prepare_wps_directory(pth, force=True)


@pytest.mark.parametrize(
    "dataset,tile_size",
    [
        (test_files / "usgs", (1200, 1200)),
        (test_files / "usgs", (250, 250)),
        (test_files / "usgs", (200, 200)),
    ],
    indirect=["dataset"],
)
def test__pad_data_if_needed(dataset, tile_size):
    missing_val = 127
    config.set({"index.missing_value": missing_val})

    da = dataset[list(dataset.data_vars.keys())[0]]
    padded_da = _pad_data_if_needed(da, tile_size)

    assert (np.array(padded_da.shape) % np.array(tile_size) == np.zeros(2)).all()
    if (np.array(da.size) % np.array(tile_size)).all():
        assert padded_da.isel(x=-1, y=-1).values == missing_val


@pytest.mark.parametrize(
    "dataset,tile_size",
    [
        (test_files / "usgs", (250, 250)),
    ],
    indirect=["dataset"],
)
def test__pad_data_if_needed_keyerr(dataset, tile_size):
    da = dataset[list(dataset.data_vars.keys())[0]]
    with pytest.raises(KeyError):
        _pad_data_if_needed(da, tile_size)


@pytest.mark.parametrize(
    "dataset",
    [
        test_files / "usgs",
    ],
    indirect=True,
)
def test_plot(dataset):
    dataset.wps.plot()
