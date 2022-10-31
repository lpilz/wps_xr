from pathlib import Path

import pytest
import xarray as xr

from wps_xr.backend import BinaryBackend
from wps_xr.index import _construct_index

test_files = Path(__file__).parents[0] / "test_files"


@pytest.mark.parametrize(
    "filename,dtype,dims",
    [(Path("usgs") / "00001-01200.00001-01200", "int8", ["x", "y"])],
)
def test_open_dataset(filename, dtype, dims):
    _construct_index(test_files / filename.parents[0])
    ds = xr.open_dataset(test_files / filename, dtype=dtype, engine=BinaryBackend)
    assert "foo" in ds.data_vars
    for dim in dims:
        assert dim in list(ds.dims.keys())
    assert ds.foo.dtype == dtype
    assert (ds.foo.values != 0).any()


@pytest.mark.parametrize(
    "filename,dtype,dims",
    [(Path("usgs") / "01201-02400.00001-01200", "int8", ["x", "y"])],
)
def test_open_dataset_pathlib(filename, dtype, dims):
    _construct_index(test_files / filename.parents[0])
    ds = xr.open_dataset(Path(test_files) / filename, dtype=dtype, engine=BinaryBackend)
    assert "foo" in ds.data_vars
    for dim in dims:
        assert dim in list(ds.dims.keys())
    assert ds.foo.dtype == dtype
    assert (ds.foo.values != 0).any()


@pytest.mark.parametrize(
    "filename,dtype,dims",
    [(Path("usgs") / "00001-01200.00001-01200", "int8", ["x", "y"])],
)
def test_open_dataset_dask(filename, dtype, dims):
    _construct_index(test_files / filename.parents[0])
    ds = xr.open_dataset(
        test_files / filename,
        dtype=dtype,
        engine=BinaryBackend,
        chunks="auto",
    )
    assert "foo" in ds.data_vars
    for dim in dims:
        assert dim in list(ds.dims.keys())
    assert ds.foo.dtype == dtype
    assert ds.foo.chunks
    assert (ds.foo.values != 0).any()
