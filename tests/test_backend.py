from pathlib import Path

import pytest
import xarray as xr

from wps_xr.backend import BinaryBackend
from wps_xr.index import _construct_index

test_files = Path(__file__).parents[0] / "test_files"


@pytest.mark.parametrize(
    "filename,dtype,dims,dask",
    [
        (Path("usgs") / "00001-01200.00001-01200", "int8", ["x", "y"], True),
        (Path("usgs") / "00001-01200.00001-01200", "int8", ["x", "y"], False),
    ],
)
def test_open_dataset(filename, dtype, dims, dask):
    _construct_index(test_files / filename.parents[0])
    ds = xr.open_dataset(
        test_files / filename,
        dtype=dtype,
        engine=BinaryBackend,
        chunks="auto" if dask else None,
    )
    assert "foo" in ds.data_vars
    for dim in dims:
        assert dim in list(ds.dims.keys())
    assert ds.foo.dtype == dtype
    assert (ds.foo.values != 0).any()
    if dask:
        assert ds.foo.chunks
