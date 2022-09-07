import os
from pathlib import Path

import pytest
import xarray as xr

from wps_xr.backend import BinaryBackend
from wps_xr.index import _construct_index

test_files = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_files")


@pytest.mark.parametrize(
    "filename,dtype,dims",
    [(os.path.join("usgs", "00001-01200.00001-01200"), "int8", ["x", "y"])],
)
def test_open_dataset(filename, dtype, dims):
    _construct_index(os.path.join(test_files, os.path.dirname(filename)))
    ds = xr.open_dataset(
        os.path.join(test_files, filename), dtype=dtype, engine=BinaryBackend
    )
    assert "foo" in ds.data_vars
    for dim in dims:
        assert dim in list(ds.dims.keys())
    assert ds.foo.dtype == dtype
    assert (ds.foo.values != 0).any()


@pytest.mark.parametrize(
    "filename,dtype,dims",
    [(Path("usgs") / Path("01201-02400.00001-01200"), "int8", ["x", "y"])],
)
def test_open_dataset_pathlib(filename, dtype, dims):
    _construct_index(os.path.join(test_files, os.path.dirname(filename)))
    ds = xr.open_dataset(Path(test_files) / filename, dtype=dtype, engine=BinaryBackend)
    assert "foo" in ds.data_vars
    for dim in dims:
        assert dim in list(ds.dims.keys())
    assert ds.foo.dtype == dtype
    assert (ds.foo.values != 0).any()


@pytest.mark.parametrize(
    "filename,dtype,dims",
    [(os.path.join("usgs", "00001-01200.00001-01200"), "int8", ["x", "y"])],
)
def test_open_dataset_dask(filename, dtype, dims):
    _construct_index(os.path.join(test_files, os.path.dirname(filename)))
    ds = xr.open_dataset(
        os.path.join(test_files, filename),
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
