import os
import pytest
import glob
import math
import numpy as np
import xarray as xr

from pathlib import Path
from itertools import product

from wps_xr.wps_accessor import WPSAccessor
from wps_xr.wps import open_dataset

test_files = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_files")

def get_num_filename(lst):
    return 5 if np.max(lst)<=99999 else 6 if np.max(lst)<=999999 else None

def gen_expexcted_filelist(ds, tile_size):
    # get dimensions
    x_ext, y_ext = ds.dims['x'], ds.dims['y']

    # number of tiles in each direction
    x_num, y_num = [math.ceil(ext/size) for ext,size in zip((x_ext,y_ext), tile_size)]

    # get upper bound of x,y indices for files
    get_tiles = lambda n,s: list(range(s,n*s+1,s))
    x_idx, y_idx = [get_tiles(n,s) for n,s in zip((x_num, y_num), tile_size)]

    # get list of combined x,y indices
    idx_list = product(x_idx, y_idx)

    num_filename = get_num_filename([*x_idx, *y_idx])

    _fmt = lambda x: f"{x:0{num_filename}d}"
    return [f"{_fmt(x-(tile_size[0]-1))}-{_fmt(x)}.{_fmt(y-(tile_size[1]-1))}-{_fmt(y)}" for x,y in idx_list]

@pytest.fixture(scope="session")
def dataset(request):
    return open_dataset(request.param)

@pytest.mark.parametrize("dataset,tile_size", [(f"{os.path.join(test_files,'usgs')}", (1200,1200)), (f"{os.path.join(test_files,'usgs')}", (200,200))], indirect=["dataset"])
def test_to_wps_data(tmp_path_factory, dataset, tile_size):
    """Tests WPSAccessor.to_wps_data
    """
    dn = tmp_path_factory.mktemp(f"out_{tile_size[0]}-{tile_size[1]}")
    dataset.wps.to_wps_data(dn, tile_size=tile_size, force=True)
    indexfile = os.path.join(dn, "index")
    assert os.path.exists(indexfile) and os.path.isfile(indexfile)

    expected_filelist = gen_expexcted_filelist(dataset, tile_size)
    for _file in expected_filelist:
        __file = os.path.join(dn,_file)
        assert os.path.exists(__file) and os.path.isfile(__file)
